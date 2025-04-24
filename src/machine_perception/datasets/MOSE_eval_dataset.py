from __future__ import division
import os
import shutil
import json
import cv2
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOSEval(Dataset):
    def __init__(
        self,
        image_root,
        label_root,
        seq_name,
        images,
        labels,
        rgb=True,
        transform=None,
        single_obj=False,
        resolution=None,
    ):
        self.image_root = image_root
        self.label_root = label_root
        self.seq_name = seq_name
        self.images = images
        self.labels = labels
        self.obj_num = 1
        self.num_frame = len(self.images)
        self.transform = transform
        self.rgb = rgb
        self.single_obj = single_obj
        self.resolution = resolution

        self.obj_nums = []
        self.obj_indices = []

        curr_objs = [0]
        for img_name in self.images:
            self.obj_nums.append(len(curr_objs) - 1)
            current_label_name = img_name.split(".")[0] + ".png"
            if current_label_name in self.labels:
                current_label = self.read_label(current_label_name)
                curr_obj = list(np.unique(current_label))
                for obj_idx in curr_obj:
                    if obj_idx not in curr_objs:
                        curr_objs.append(obj_idx)
            self.obj_indices.append(curr_objs.copy())

        self.obj_nums[0] = self.obj_nums[1]

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_root, self.seq_name, img_name)
        img = cv2.imread(img_path)
        img = np.array(img, dtype=np.float32)
        if self.rgb:
            img = img[:, :, [2, 1, 0]]
        return img

    def read_label(self, label_name, squeeze_idx=None):
        label_path = os.path.join(self.label_root, self.seq_name, label_name)
        label = Image.open(label_path)
        label = np.array(label, dtype=np.uint8)
        if self.single_obj:
            label = (label > 0).astype(np.uint8)
        elif squeeze_idx is not None:
            squeezed_label = label * 0
            for idx in range(len(squeeze_idx)):
                obj_id = squeeze_idx[idx]
                if obj_id == 0:
                    continue
                mask = label == obj_id
                squeezed_label += (mask * idx).astype(np.uint8)
            label = squeezed_label
        return label

    def __getitem__(self, idx):
        img_name = self.images[idx]
        current_img = self.read_image(idx)
        height, width, channels = current_img.shape
        if self.resolution is not None:
            width = int(np.ceil(float(width) * self.resolution / float(height)))
            height = int(self.resolution)

        current_label_name = img_name.split(".")[0] + ".png"
        obj_num = self.obj_nums[idx]
        obj_idx = self.obj_indices[idx]

        if current_label_name in self.labels:
            current_label = self.read_label(current_label_name, obj_idx)
            sample = {"current_img": current_img, "current_label": current_label}
        else:
            sample = {"current_img": current_img}

        sample["meta"] = {
            "seq_name": self.seq_name,
            "frame_num": self.num_frame,
            "obj_num": obj_num,
            "current_name": img_name,
            "height": height,
            "width": width,
            "flip": False,
            "obj_idx": obj_idx,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class MOSE_Eval(object):
    def __init__(
        self,
        root: str,
        split: str = "val",
        transform: transforms.Compose | None = None,
        rgb: bool = True,
        full_resolution: bool = False,
        result_root: str | None = None,
    ):
        self.transform = transform
        self.rgb = rgb
        self.result_root = result_root
        self.single_obj = False
        if full_resolution:
            resolution = "Full-Resolution"
        else:
            resolution = "480p"
        self.image_root = os.path.join(root, split, "JPEGImages")
        self.label_root = os.path.join(root, split, "Annotations")
        self.seqs = list(np.unique(os.listdir(self.image_root)))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        images = list(np.sort(os.listdir(os.path.join(self.image_root, seq_name))))
        images = [img for img in images if img.endswith(".jpg")]
        labels = [images[0].replace("jpg", "png")]
        
       
        
        if not os.path.isfile(os.path.join(self.result_root, seq_name, labels[0])):
            seq_result_folder = os.path.join(self.result_root, seq_name)
            try:
                if not os.path.exists(seq_result_folder):
                    os.makedirs(seq_result_folder)
            except Exception as inst:
                print(inst)
                print(
                    "Failed to create a result folder for sequence {}.".format(seq_name)
                )
            source_label_path = os.path.join(self.label_root, seq_name, labels[0])
            result_label_path = os.path.join(self.result_root, seq_name, labels[0])
            if self.single_obj:
                label = Image.open(source_label_path)
                label = np.array(label, dtype=np.uint8)
                label = (label > 0).astype(np.uint8)
                label = Image.fromarray(label).convert("P")
                label.save(result_label_path)
            else:
                shutil.copy(source_label_path, result_label_path)

        seq_dataset = VOSEval(
            self.image_root,
            self.label_root,
            seq_name,
            images,
            labels,
            transform=self.transform,
            rgb=self.rgb,
            single_obj=self.single_obj,
            resolution=None,
        )
        return seq_dataset
