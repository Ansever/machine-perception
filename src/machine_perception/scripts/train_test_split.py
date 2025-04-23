
from pathlib import Path
import json
import random


BASE_DIR = Path(__file__).resolve().parents[3] / "data"
ANNOTATIONS_DIR = BASE_DIR / "train_original" / "Annotations"
FRAMES_DIR = BASE_DIR / "train_original" / "JPEGImages"
META_FILE = BASE_DIR / "meta_train.json"


def load_meta() -> dict:
    with open(META_FILE, "r") as f:
        return json.load(f)


def save_meta(filename: str, meta: dict) -> None:
    out_path = BASE_DIR / filename
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def split_dataset(
    meta: dict, train_ratio: float = 0.8, seed: int = 42
) -> tuple[dict, dict]:
    video_ids = list(meta["videos"].keys())
    random.seed(seed)
    random.shuffle(video_ids)

    split_index = int(len(video_ids) * train_ratio)
    train_ids = video_ids[:split_index]
    test_ids = video_ids[split_index:]

    train_meta = {
        "videos": {vid.split(".")[0]: meta["videos"][vid] for vid in train_ids}
    }
    test_meta = {"videos": {vid.split(".")[0]: meta["videos"][vid] for vid in test_ids}}

    return train_meta, test_meta


def main() -> None:
    meta = load_meta()
    train_meta, test_meta = split_dataset(meta)

    print(f"Train videos: {len(train_meta['videos'])}")
    print(f"Test videos: {len(test_meta['videos'])}")

    save_meta("meta_train_split.json", train_meta)
    save_meta("meta_test_split.json", test_meta)


if __name__ == "__main__":
    main()
