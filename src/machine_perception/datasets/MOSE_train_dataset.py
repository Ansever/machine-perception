from __future__ import division
import os
import json
import random
import cv2
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

cv2.setNumThreads(0)


def _get_images(sample):
    return [sample["ref_img"], sample["prev_img"]] + sample["curr_img"]


def _get_labels(sample):
    return [sample["ref_label"], sample["prev_label"]] + sample["curr_label"]


def _merge_sample(sample1, sample2, min_obj_pixels=100, max_obj_n=10):
    sample1_images = _get_images(sample1)
    sample2_images = _get_images(sample2)

    sample1_labels = _get_labels(sample1)
    sample2_labels = _get_labels(sample2)

    obj_idx = torch.arange(0, max_obj_n * 2 + 1).view(max_obj_n * 2 + 1, 1, 1)
    selected_idx = None
    selected_obj = None

    all_img = []
    all_mask = []
    for idx, (s1_img, s2_img, s1_label, s2_label) in enumerate(
        zip(sample1_images, sample2_images, sample1_labels, sample2_labels)
    ):
        s2_fg = (s2_label > 0).float()
        s2_bg = 1 - s2_fg
        merged_img = s1_img * s2_bg + s2_img * s2_fg
        merged_mask = s1_label * s2_bg.long() + ((s2_label + max_obj_n) * s2_fg.long())
        merged_mask = (merged_mask == obj_idx).float()
        if idx == 0:
            after_merge_pixels = merged_mask.sum(dim=(1, 2), keepdim=True)
            selected_idx = after_merge_pixels > min_obj_pixels
            selected_idx[0] = True
            obj_num = selected_idx.sum().int().item() - 1
            selected_idx = selected_idx.expand(
                -1, s1_label.size()[1], s1_label.size()[2]
            )
            if obj_num > max_obj_n:
                selected_obj = list(range(1, obj_num + 1))
                random.shuffle(selected_obj)
                selected_obj = [0] + selected_obj[:max_obj_n]

        merged_mask = merged_mask[selected_idx].view(
            obj_num + 1, s1_label.size()[1], s1_label.size()[2]
        )
        if obj_num > max_obj_n:
            merged_mask = merged_mask[selected_obj]
        merged_mask[0] += 0.1
        merged_mask = torch.argmax(merged_mask, dim=0, keepdim=True).long()

        all_img.append(merged_img)
        all_mask.append(merged_mask)

    sample = {
        "ref_img": all_img[0],
        "prev_img": all_img[1],
        "curr_img": all_img[2:],
        "ref_label": all_mask[0],
        "prev_label": all_mask[1],
        "curr_label": all_mask[2:],
    }
    sample["meta"] = sample1["meta"]
    sample["meta"]["obj_num"] = min(obj_num, max_obj_n)
    return sample


class VOSTrain(Dataset):
    def __init__(
        self,
        image_root,
        label_root,
        imglistdic,
        transform=None,
        rgb=True,
        repeat_time=1,
        rand_gap=3,
        seq_len=5,
        rand_reverse=True,
        dynamic_merge=True,
        enable_prev_frame=False,
        merge_prob=0.3,
        max_obj_n=10,
    ):
        self.image_root = image_root
        self.label_root = label_root
        self.rand_gap = rand_gap
        self.seq_len = seq_len
        self.rand_reverse = rand_reverse
        self.repeat_time = repeat_time
        self.transform = transform
        self.dynamic_merge = dynamic_merge
        self.merge_prob = merge_prob
        self.enable_prev_frame = enable_prev_frame
        self.max_obj_n = max_obj_n
        self.rgb = rgb
        self.imglistdic = imglistdic
        self.seqs = list(self.imglistdic.keys())
        print("Video Num: {} X {}".format(len(self.seqs), self.repeat_time))

    def __len__(self):
        return int(len(self.seqs) * self.repeat_time)

    def reverse_seq(self, imagelist, lablist):
        if np.random.randint(2) == 1:
            imagelist = imagelist[::-1]
            lablist = lablist[::-1]
        return imagelist, lablist

    def get_ref_index(self, seqname, lablist, objs, min_fg_pixels=200, max_try=5):
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(len(lablist))
            if ref_index in bad_indices:
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index])
            )
            ref_label = np.array(ref_label, dtype=np.uint8)
            ref_objs = list(np.unique(ref_label))
            is_consistent = True
            for obj in ref_objs:
                if obj == 0:
                    continue
                if obj not in objs:
                    is_consistent = False
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels and is_consistent:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_ref_index_v2(
        self, seqname, lablist, min_fg_pixels=200, max_try=20, total_gap=0
    ):
        search_range = len(lablist) - total_gap
        if search_range <= 1:
            return 0
        bad_indices = []
        for _ in range(max_try):
            ref_index = np.random.randint(search_range)
            if ref_index in bad_indices:
                continue
            ref_label = Image.open(
                os.path.join(self.label_root, seqname, lablist[ref_index])
            )
            ref_label = np.array(ref_label, dtype=np.uint8)
            xs, ys = np.nonzero(ref_label)
            if len(xs) > min_fg_pixels:
                break
            bad_indices.append(ref_index)
        return ref_index

    def get_curr_gaps(self, seq_len, max_gap=999, max_try=10):
        for _ in range(max_try):
            curr_gaps = []
            total_gap = 0
            for _ in range(seq_len):
                gap = int(np.random.randint(self.rand_gap) + 1)
                total_gap += gap
                curr_gaps.append(gap)
            if total_gap <= max_gap:
                break
        return curr_gaps, total_gap

    def get_prev_index(self, lablist, total_gap):
        search_range = len(lablist) - total_gap
        if search_range > 1:
            prev_index = np.random.randint(search_range)
        else:
            prev_index = 0
        return prev_index

    def check_index(self, total_len, index, allow_reflect=True):
        if total_len <= 1:
            return 0

        if index < 0:
            if allow_reflect:
                index = -index
                index = self.check_index(total_len, index, True)
            else:
                index = 0
        elif index >= total_len:
            if allow_reflect:
                index = 2 * (total_len - 1) - index
                index = self.check_index(total_len, index, True)
            else:
                index = total_len - 1

        return index

    def get_curr_indices(self, lablist, prev_index, gaps):
        total_len = len(lablist)
        curr_indices = []
        now_index = prev_index
        for gap in gaps:
            now_index += gap
            curr_indices.append(self.check_index(total_len, now_index))
        return curr_indices

    def get_image_label(self, seqname, imagelist, lablist, index):
        image = cv2.imread(os.path.join(self.image_root, seqname, imagelist[index]))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        label = Image.open(os.path.join(self.label_root, seqname, lablist[index]))
        label = np.array(label, dtype=np.uint8)

        return image, label

    def sample_sequence(self, idx):
        idx = idx % len(self.seqs)
        seqname = self.seqs[idx]
        imagelist, lablist = self.imglistdic[seqname]
        frame_num = len(imagelist)
        if self.rand_reverse:
            imagelist, lablist = self.reverse_seq(imagelist, lablist)

        is_consistent = False
        max_try = 5
        try_step = 0
        while is_consistent is False and try_step < max_try:
            try_step += 1

            # generate random gaps
            curr_gaps, total_gap = self.get_curr_gaps(self.seq_len - 1)

            if self.enable_prev_frame:  # prev frame is randomly sampled
                # get prev frame
                prev_index = self.get_prev_index(lablist, total_gap)
                prev_image, prev_label = self.get_image_label(
                    seqname, imagelist, lablist, prev_index
                )
                prev_objs = list(np.unique(prev_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, prev_index, curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index
                    )
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(prev_objs + curr_objs))

                start_index = prev_index
                end_index = max(curr_indices)
                # get ref frame
                _try_step = 0
                ref_index = self.get_ref_index_v2(seqname, lablist)
                while (
                    ref_index > start_index
                    and ref_index <= end_index
                    and _try_step < max_try
                ):
                    _try_step += 1
                    ref_index = self.get_ref_index_v2(seqname, lablist)
                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index
                )
                ref_objs = list(np.unique(ref_label))
            else:  # prev frame is next to ref frame
                # get ref frame
                ref_index = self.get_ref_index_v2(seqname, lablist)

                ref_image, ref_label = self.get_image_label(
                    seqname, imagelist, lablist, ref_index
                )
                ref_objs = list(np.unique(ref_label))

                # get curr frames
                curr_indices = self.get_curr_indices(lablist, ref_index, curr_gaps)
                curr_images, curr_labels, curr_objs = [], [], []
                for curr_index in curr_indices:
                    curr_image, curr_label = self.get_image_label(
                        seqname, imagelist, lablist, curr_index
                    )
                    c_objs = list(np.unique(curr_label))
                    curr_images.append(curr_image)
                    curr_labels.append(curr_label)
                    curr_objs.extend(c_objs)

                objs = list(np.unique(curr_objs))
                prev_image, prev_label = curr_images[0], curr_labels[0]
                curr_images, curr_labels = curr_images[1:], curr_labels[1:]

            is_consistent = True
            for obj in objs:
                if obj == 0:
                    continue
                if obj not in ref_objs:
                    is_consistent = False
                    break

        # get meta info
        obj_num = list(np.sort(ref_objs))[-1]

        sample = {
            "ref_img": ref_image,
            "prev_img": prev_image,
            "curr_img": curr_images,
            "ref_label": ref_label,
            "prev_label": prev_label,
            "curr_label": curr_labels,
        }
        sample["meta"] = {
            "seq_name": seqname,
            "frame_num": frame_num,
            "obj_num": obj_num,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        sample1 = self.sample_sequence(idx)

        if self.dynamic_merge and (
            sample1["meta"]["obj_num"] == 0 or random.random() < self.merge_prob
        ):
            rand_idx = np.random.randint(len(self.seqs))
            while rand_idx == (idx % len(self.seqs)):
                rand_idx = np.random.randint(len(self.seqs))

            sample2 = self.sample_sequence(rand_idx)

            sample = self.merge_sample(sample1, sample2)
        else:
            sample = sample1

        return sample

    def merge_sample(self, sample1, sample2, min_obj_pixels=100):
        return _merge_sample(sample1, sample2, min_obj_pixels, self.max_obj_n)


class MOSE_Train(VOSTrain):
    def __init__(
        self,
        root: str,
        meta_file: str,
        split: str = "train",
        transform: transforms.Compose | None = None,
        rgb: bool = True,
        rand_gap: int = 3,
        seq_len: int = 3,
        rand_reverse: bool = True,
        dynamic_merge: bool = True, # TODO: fix bugs when dynamic_merge is True
        enable_prev_frame: bool = False,
        max_obj_n: int = 10,
        merge_prob: float = 0.3,
    ):
        root = os.path.join(
            root,
        )
        image_root = os.path.join(root, split, "JPEGImages")
        label_root = os.path.join(root, split, "Annotations")
        self.seq_list_file = os.path.join(root, meta_file)
        self._check_preprocess()
        seq_names = list(self.ann_f.keys())

        imglistdic = {}
        for seq_name in seq_names:
            images = self.ann_f[seq_name]["frames"]
            labels = [x.split(".")[0] + ".png" for x in images]

            if len(images) < 2:
                print("Short video: " + seq_name)
                continue
            imglistdic[seq_name] = (images, labels)

        super(MOSE_Train, self).__init__(
            image_root,
            label_root,
            imglistdic,
            transform,
            rgb,
            1,
            rand_gap,
            seq_len,
            rand_reverse,
            dynamic_merge,
            enable_prev_frame,
            merge_prob=merge_prob,
            max_obj_n=max_obj_n,
        )

    def _check_preprocess(self):
        if not os.path.isfile(self.seq_list_file):
            print("No such file: {}".format(self.seq_list_file))
            return False
        else:
            self.ann_f = json.load(open(self.seq_list_file, "r"))["videos"]
            return True
