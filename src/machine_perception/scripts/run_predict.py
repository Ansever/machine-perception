from pathlib import Path

import torch
from ultralytics import YOLO
from machine_perception.experiments.predictor import (
    SamVideoPredictor,
    StmVideoPredictor,
    YoloVideoPredictor,
)
from machine_perception.models.sam2 import SAM2
from machine_perception.models.stm.model import STM, load_stm_state_dict
from machine_perception.utils import Instance


def run_sam(instance: Instance, device: str):
    base_dir = Path().resolve()

    # CHECKPOINT = base_dir / "resources" / "sam" / "sam2.1_hiera_large.pt"
    # CONFIG = base_dir / "resources" / "sam" / "sam2.1_hiera_l.yaml"
    CHECKPOINT = base_dir / "resources" / "sam" / "sam2.1_hiera_small.pt"
    CONFIG = base_dir / "resources" / "sam" / "sam2.1_hiera_s.yaml"

    sam = SAM2(config=str(CONFIG), checkpoint=CHECKPOINT, device=device)
    sam_predictor = SamVideoPredictor(sam)
    sam_predictor.predict_video(instance, f"temp_output_masks/sam/{instance['id_']}")


def run_stm(instance: Instance, device: str):
    WEIGHTS_PATH = Path().resolve() / "resources" / "stm" / "STM_weights.pth"

    stm = STM()
    stm.load_state_dict(load_stm_state_dict(WEIGHTS_PATH))
    stm.to(device)

    stm_predictor = StmVideoPredictor(
        stm, k=11, memory_every=5, memory_number=None, device=device
    )
    stm_predictor.predict_video(instance, f"temp_output_masks/stm/{instance['id_']}")


def run_yolo(instance: Instance, device: str):
    model_path = Path().resolve() / "resources" / "yolo" / "yolo11x-seg.pt"
    yolo = YOLO(model=model_path)
    yolo_predictor = YoloVideoPredictor(yolo)
    yolo_predictor.predict_video(instance, f"temp_output_masks/yolo/{instance['id_']}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    instance_id = "0a7a3629"
    instance: Instance = {
        "id_": instance_id,
        "frames_dir": f"data/train/JPEGImages/{instance_id}",
        "input_mask_path": f"data/train/Annotations/{instance_id}/00000.png",
        "n_frames": 15,
        "n_objects": 1,
        "frame_height": 720,
        "frame_width": 1076,
    }

    # run_sam(instance, DEVICE)
    # run_stm(instance, DEVICE)
    run_yolo(instance, DEVICE)
