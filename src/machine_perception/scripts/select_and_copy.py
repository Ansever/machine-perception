from pathlib import Path
import json
import shutil
from typing import Literal

import tqdm


def save_subset(
    data_path: Path,
    data_subset_path: Path,
    videos: dict,
    split: Literal["train", "test"],
):
    size = len(videos)

    video_ids = list(videos.keys())
    videos_subset = {
        "videos": {video_ids[i]: videos[video_ids[i]] for i in range(0, size, 4)}
    }

    with open(data_subset_path / f"meta_{split}_split.json", "w") as f:
        json.dump(videos_subset, f, indent=4)

    imgs_dir = data_path / "train" / "JPEGImages"
    anno_dir = data_path / "train" / "Annotations"

    imgs_subset_dir = data_subset_path / split / "JPEGImages"
    imgs_subset_dir.mkdir(parents=True, exist_ok=True)
    anno_subset_dir = data_subset_path / split / "Annotations"
    anno_subset_dir.mkdir(parents=True, exist_ok=True)
    for video_id in tqdm.tqdm(videos_subset["videos"].keys(), desc=split):
        shutil.copytree(imgs_dir / video_id, imgs_subset_dir / video_id)
        shutil.copytree(anno_dir / video_id, anno_subset_dir / video_id)


if __name__ == "__main__":
    data_path = Path().resolve() / "data"
    data_subset_path = Path().resolve() / "data_subset"
    data_subset_path.mkdir(exist_ok=True)

    with open(data_path / "meta_train_split.json") as f:
        train_split = json.load(f)
    with open(data_path / "meta_test_split.json") as f:
        test_split = json.load(f)

    train_videos = train_split["videos"]
    test_videos = test_split["videos"]

    # save_subset(data_path, data_subset_path, train_videos, "train")
    save_subset(data_path, data_subset_path, test_videos, "test")
