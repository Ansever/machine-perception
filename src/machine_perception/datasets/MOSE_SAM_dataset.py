import os
import json
import cv2
import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class MOSE_SAM_Samples(Dataset):
    def __init__(self, images_path: str, labels_path: str):
        self.images_path = images_path
        self.labels_path = labels_path
        self.images = sorted(os.listdir(images_path))
        self.labels = sorted(os.listdir(labels_path))

    def read_image(self, idx: int) -> np.ndarray:
        img_name = self.images[idx]
        img_path = os.path.join(self.images_path, img_name)
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
        return img

    def read_label(self, idx: int) -> np.ndarray:
        label_name = self.labels[idx]
        label_path = os.path.join(self.labels_path, label_name)
        label = np.array(Image.open(label_path), dtype=np.uint8)
        return label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        img = self.read_image(idx)
        label = self.read_label(idx) if idx < len(self.labels) else None
        return {
            "image": img,
            "label": label,
            "image_name": self.images[idx],
            "label_name": self.labels[idx],
        }


class MOSE_SAM_Dataset(Dataset):
    def __init__(self, root: str, meta_file: str):
        self.image_root = os.path.join(root, "JPEGImages")
        self.label_root = os.path.join(root, "Annotations")
        self.meta_file = self.__read_meta_file(os.path.join(meta_file))
        self.seqs = list(self.meta_file.keys())

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq_name = self.seqs[idx]
        return {
            "seq_name": seq_name,
            "images_path": os.path.join(self.image_root, seq_name),
            "labels_path": os.path.join(self.label_root, seq_name),
            "video_frames": MOSE_SAM_Samples(
                os.path.join(self.image_root, seq_name),
                os.path.join(self.label_root, seq_name),
            ),
        }

    @staticmethod
    def __read_meta_file(meta_file: str) -> dict:
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Meta file {meta_file} does not exist.")
        with open(meta_file, "r") as f:
            meta_data = json.load(f).get("videos")
        if not meta_data:
            raise ValueError(f"Meta file {meta_file} is empty or invalid.")
        return meta_data
