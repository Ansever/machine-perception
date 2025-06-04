import json
import os
from PIL import Image

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils import data


def resize_frames_and_masks(
    frames_t: torch.Tensor, masks_t: torch.Tensor, new_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        F.interpolate(frames_t, size=new_size, mode="bilinear"),
        F.interpolate(masks_t, size=new_size, mode="nearest"),
    )


class MoseStmDataset(data.Dataset):
    # for multi object, do shuffling

    def __init__(
        self,
        root: str,
        imset: str = "meta_train_split.json",
        single_object: bool = False,
        new_size: tuple[int, int] | None = None,
        n_frames_to_sample: int = 3,
    ):
        self.root = root
        self.mask_dir = os.path.join(root, "train", "Annotations")
        self.image_dir = os.path.join(root, "train", "JPEGImages")

        imset_fpath = os.path.join(root, imset)

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        with open(imset_fpath, "r") as f:
            meta = json.load(f)

        for _video, video_data in meta["videos"].items():
            self.videos.append(_video)
            self.num_frames[_video] = video_data["length"]
            self.num_objects[_video] = len(video_data["objects"])
            self.shape[_video] = (
                video_data["width"],
                video_data["height"],
            )  # Here the original values are swapped in the dataset meta files lol

        self.K = 11
        self.single_object = single_object
        self.new_size = new_size
        self.n_frames_to_sample = n_frames_to_sample

    def __len__(self):
        return len(self.videos)

    def to_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M

    def all_to_onehot(self, masks: np.ndarray):
        Ms = np.zeros(
            (self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8
        )
        for n in range(masks.shape[0]):
            Ms[:, n] = self.to_onehot(masks[n])
        return Ms

    def __getitem__(self, index: int):
        video = self.videos[index]
        info = {
            "name": video,
            "num_frames": self.num_frames[video],
        }

        final_num_frames = min(self.n_frames_to_sample, self.num_frames[video])
        frame_idxs = np.random.choice(
            self.num_frames[video], final_num_frames, replace=False
        )
        frame_idxs.sort()

        N_frames = np.empty(
            (final_num_frames,) + self.shape[video] + (3,), dtype=np.float32
        )
        N_masks = np.empty((final_num_frames,) + self.shape[video], dtype=np.uint8)
        for i, f in enumerate(frame_idxs):
            img_file = os.path.join(self.image_dir, video, "{:05d}.jpg".format(f))
            N_frames[i] = np.array(Image.open(img_file).convert("RGB")) / 255.0
            try:
                mask_file = os.path.join(self.mask_dir, video, "{:05d}.png".format(f))
                N_masks[i] = np.array(
                    Image.open(mask_file).convert("P"), dtype=np.uint8
                )
            except Exception:
                N_masks[i] = 255

        Fs = torch.from_numpy(
            np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()
        ).float()
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(
                np.uint8
            )
            Ms = torch.from_numpy(self.all_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            # return Fs, Ms, num_objects, info
        else:
            Ms = torch.from_numpy(self.all_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            # return Fs, Ms, num_objects, info

        # Fs, Ms = Fs[:, :, frame_idxs], Ms[:, :, frame_idxs]
        Fs, Ms = resize_frames_and_masks(Fs, Ms, new_size=(384, 384))
        Fs, Ms = Fs.to(Fs.device), Ms.to(Ms.device)
        return Fs, Ms, num_objects, info


if __name__ == "__main__":
    import tqdm

    dataset = MoseStmDataset(
        "data",
        imset="meta_train_split.json",
        single_object=False,
        new_size=(384, 384),
        n_frames_to_sample=3,
    )

    for frames, masks, num_objects, info in tqdm.tqdm(dataset):
        print(f"{frames.shape = }")
        print(f"{masks.shape = }")
        print(f"{num_objects = }")
        print(f"{info = }")
        break
        # print(f"{frames.shape = }")
        # print(f"{masks.shape = }")
        # print(f"{num_objects = }")
        # print(f"{info = }")
