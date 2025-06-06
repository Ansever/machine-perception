from abc import ABC, abstractmethod
import glob
from pathlib import Path

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from ultralytics import YOLO
from sklearn.metrics import jaccard_score
from scipy.optimize import linear_sum_assignment

from machine_perception.models.sam2 import SAM2
from machine_perception.models.stm.model import STM
from machine_perception.utils import Instance, idx_to_file_stem


class BaseVideoPredictor(ABC):
    @abstractmethod
    def predict_video(self, instance: Instance, output_masks_dir: str | Path) -> None:
        pass


class SamVideoPredictor(BaseVideoPredictor):
    def __init__(self, sam: SAM2):
        self.sam = sam

    def predict_video(self, instance, output_masks_dir):
        output_masks_dir = Path(output_masks_dir)
        output_masks_dir.mkdir(exist_ok=True)

        input_mask_pil = Image.open(instance["input_mask_path"])

        input_mask = np.array(input_mask_pil, dtype=np.uint8)
        prediction = self.sam.predict(instance["frames_dir"], input_mask)
        for frame_id in range(0, instance["n_frames"]):
            instance_masks = prediction[frame_id]
            combined_mask = np.zeros_like(input_mask)
            for idx, (instance_id, mask) in enumerate(instance_masks.items()):
                single_mask = mask[0]  # (H, W)
                combined_mask[single_mask.astype(bool)] = instance_id

            combined_mask_pil = Image.fromarray(combined_mask, mode="P")
            combined_mask_pil.putpalette(input_mask_pil.palette)
            combined_mask_pil.save(
                output_masks_dir / f"{idx_to_file_stem(frame_id)}.png"
            )


class YoloVideoPredictor(BaseVideoPredictor):
    def __init__(self, yolo: YOLO):
        self.yolo = yolo

    @staticmethod
    def _match_yolo_to_orig(
        yolo_id_to_mask: dict[int, np.ndarray], orig_id_to_mask: dict[int, np.ndarray]
    ) -> dict[int, int]:
        scores = np.empty((len(yolo_id_to_mask), len(orig_id_to_mask)))
        yolo_ids, orig_ids = [], []
        for i, (yolo_id, yolo_mask) in enumerate(yolo_id_to_mask.items()):
            yolo_ids.append(yolo_id)
            for j, (orig_id, orig_mask) in enumerate(orig_id_to_mask.items()):
                orig_ids.append(orig_id)
                scores[i, j] = jaccard_score(yolo_mask, orig_mask, average="micro")

        yolo_idxs, orig_idxs = linear_sum_assignment(-scores)
        return {
            yolo_ids[yolo_idx]: orig_ids[orig_idx]
            for yolo_idx, orig_idx in zip(yolo_idxs, orig_idxs)
        }

    def predict_video(self, instance, output_masks_dir):
        output_masks_dir = Path(output_masks_dir)
        output_masks_dir.mkdir(exist_ok=True)

        frames = np.empty(
            (
                instance["n_frames"],
                instance["frame_height"],
                instance["frame_width"],
                3,
            ),
            dtype=np.uint8,
        )
        for i, fpath in enumerate(glob.glob(f"{instance['frames_dir']}/*")):
            frames[i] = np.array(Image.open(fpath).convert("RGB"))
        input_mask_pil = Image.open(instance["input_mask_path"])
        input_mask = np.array(input_mask_pil, dtype=np.uint8)

        yolo_to_orig_id: dict[int, int] = {}
        for frame_id, frame in enumerate(frames):
            result = self.yolo.track(np.ascontiguousarray(frame), retina_masks=True)[0]
            if (
                result.masks is not None
                and len(result.masks) > 0
                and result.boxes is not None
                and len(result.boxes) > 0
            ):
                masks = result.masks.data.cpu().numpy().astype(np.uint8)
                yolo_ids = result.boxes.id.cpu().numpy().astype(np.uint8)

                if len(yolo_to_orig_id) == 0:
                    yolo_to_orig_id = YoloVideoPredictor._match_yolo_to_orig(
                        {id_: mask for id_, mask in zip(yolo_ids, masks)},
                        {
                            id_: (input_mask == id_).astype(np.uint8)
                            for id_ in np.unique(input_mask)
                        },
                    )

                combined_mask = np.zeros_like(input_mask)

                for i, yolo_id in enumerate(yolo_ids):
                    if yolo_id not in yolo_to_orig_id:
                        continue
                    orig_id = yolo_to_orig_id[yolo_id]
                    combined_mask[masks[i].astype(bool)] = orig_id
            else:
                combined_mask = np.zeros_like(input_mask)

            combined_mask_pil = Image.fromarray(combined_mask, mode="P")
            combined_mask_pil.putpalette(input_mask_pil.palette)
            combined_mask_pil.save(
                output_masks_dir / f"{idx_to_file_stem(frame_id)}.png"
            )


class StmVideoPredictor(BaseVideoPredictor):
    def __init__(
        self,
        model: STM,
        k: int,
        memory_every: int | None = None,
        memory_number: int | None = None,
        device: str = "cpu",
    ):
        self.model = model
        self.k = k

        if memory_every is None and memory_number is None:
            raise ValueError(
                "At least one of 'memory_every' and 'memory_number' must be given."
            )
        self.memory_every = memory_every
        self.memory_number = memory_number
        self.device = device

    @staticmethod
    def to_onehot(mask: np.ndarray, k: int):
        M = np.zeros((k, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(k):
            M[k] = (mask == k).astype(np.uint8)
        return M

    @staticmethod
    def _read_to_tensors(
        instance: Instance, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = np.empty(
            (
                instance["n_frames"],
                instance["frame_height"],
                instance["frame_width"],
                3,
            ),
            dtype=np.float32,
        )
        # masks = np.empty(
        #     (instance["n_frames"], instance["frame_height"], instance["frame_width"]),
        #     dtype=np.uint8,
        # )
        for i, fpath in enumerate(glob.glob(f"{instance['frames_dir']}/*")):
            frames[i] = np.array(Image.open(fpath).convert("RGB")) / 255.0

        mask = np.array(
            Image.open(instance["input_mask_path"]).convert("P"), dtype=np.uint8
        )

        frames_t = torch.from_numpy(
            np.transpose(frames.copy(), (3, 0, 1, 2)).copy()
        ).float()
        mask_t = torch.from_numpy(StmVideoPredictor.to_onehot(mask, k).copy()).float()

        frames_t = frames_t.unsqueeze(dim=0)
        mask_t = mask_t.unsqueeze(dim=1).unsqueeze(dim=0)

        return frames_t, mask_t

    def _run_video(
        self,
        Fs: torch.Tensor,
        Ms: torch.Tensor,
        num_frames: int,
        num_objects: int,
    ) -> tuple[np.ndarray, torch.Tensor]:
        # initialize storage tensors
        if self.memory_every:
            to_memorize = [
                int(i) for i in np.arange(0, num_frames, step=self.memory_every)
            ]
        elif self.memory_number:
            to_memorize = [
                int(round(i))
                for i in np.linspace(0, num_frames, num=self.memory_number + 2)[:-1]
            ]
        else:
            raise ValueError(
                "At least one of 'memory_every' and 'memory_number' must be given."
            )

        Es = torch.zeros(
            (Ms.shape[0], Ms.shape[1], num_frames, Ms.shape[3], Ms.shape[4])
        )
        # Es = torch.zeros_like(Ms)
        Es[:, :, 0] = Ms[:, :, 0]

        Fs = Fs.to(self.device)
        Es = Es.to(self.device)
        num_objects_t = torch.tensor([num_objects], device=self.device)

        keys, values = None, None
        for t in tqdm.tqdm(range(1, num_frames)):
            # memorize
            with torch.no_grad():
                prev_key, prev_value = self.model(
                    Fs[:, :, t - 1], Es[:, :, t - 1], num_objects_t
                )

            if keys is None or values is None:
                this_keys, this_values = prev_key, prev_value  # only prev memory
            else:
                this_keys = torch.cat([keys, prev_key], dim=3)
                this_values = torch.cat([values, prev_value], dim=3)

            # segment
            with torch.no_grad():
                # this_keys = this_keys.to(device)
                # this_values = this_values.to(device)
                logit = self.model(Fs[:, :, t], this_keys, this_values, num_objects_t)
            Es[:, :, t] = F.softmax(logit, dim=1)

            # update
            if t - 1 in to_memorize:
                keys, values = this_keys, this_values

        probs = Es[0].cpu().numpy()
        max_prob_pred = np.argmax(probs, axis=0).astype(np.uint8)
        pred = np.where(
            np.take_along_axis(probs, max_prob_pred[None], axis=0)[0] > 0.7,
            max_prob_pred,
            0,
        )
        # pred = np.argmax(probs, axis=0).astype(np.uint8)

        return pred, Es

    def predict_video(self, instance, output_masks_dir: str | Path):
        output_masks_dir = Path(output_masks_dir)
        output_masks_dir.mkdir(exist_ok=True)
        frames_t, mask_t = StmVideoPredictor._read_to_tensors(instance, self.k)

        pred, Es = self._run_video(
            frames_t, mask_t, instance["n_frames"], instance["n_objects"]
        )

        palette = Image.open(instance["input_mask_path"]).palette
        for idx, mask in enumerate(pred):
            mask_pil = Image.fromarray(mask, mode="P")
            mask_pil.putpalette(palette)
            mask_pil.save(output_masks_dir / f"{idx_to_file_stem(idx)}.png")
