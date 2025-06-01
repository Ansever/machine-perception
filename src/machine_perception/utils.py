from typing import TypedDict
import numpy as np
import cv2

from pathlib import Path


class Instance(TypedDict):
    id_: str
    frames_dir: str
    input_mask_path: str
    n_frames: int
    n_objects: int
    frame_height: int
    frame_width: int


def idx_to_file_stem(idx: int) -> str:
    idx_str = str(idx)
    return "0" * (5 - len(idx_str)) + idx_str


def frames_to_video(
    frames: list[np.ndarray] | np.ndarray, output_file: str | Path, frame_rate: int = 30
):
    if len(frames) == 0:
        raise ValueError("frames are empty")
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    for frame in frames:
        out.write(frame[:, :, [2, 1, 0]])
    out.release()


def create_video(
    frames_dir: str | Path, masks_dir: str | Path, mask_alpha: float = 0.8
):
    # TODO
    pass
