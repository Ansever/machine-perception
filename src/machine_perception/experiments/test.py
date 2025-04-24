import matplotlib.pyplot as plt

from machine_perception.datasets.MOSE_train_dataset import MOSE_Train
from machine_perception.datasets.MOSE_eval_dataset import MOSE_Eval
from torch.utils.data import DataLoader
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[3] / "data"
print(f"Data directory: {DATA_DIR}")


mose_train_dataset = MOSE_Train(
    root=DATA_DIR, split="train", meta_file="meta_train_split.json"
)

mose_eval_dataset = MOSE_Eval(
    root=DATA_DIR,
    split="val",
    result_root=DATA_DIR / "mose_eval_results",
)

mose_train_dataset_loader = DataLoader(
    dataset=mose_train_dataset,
    batch_size=1,
    shuffle=True,
)

mose_eval_dataset_loader = DataLoader(
    dataset=mose_eval_dataset,
    batch_size=1,
    shuffle=False,
)

try:
    for train_sample in mose_train_dataset_loader:
        print(f"Train sample: {train_sample['meta']['seq_name']} - {train_sample['meta']['frame_num']}")
except Exception as e:
    print(f"Error in train dataset: {e}")
    exit(1)