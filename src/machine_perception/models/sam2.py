import numpy as np
import cv2

from machine_perception.models.facebook.sam2.sam2.build_sam import (
    build_sam2_video_predictor,
)


class SAM2:
    def __init__(self, config: str, checkpoint: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint
        self.device = device
        self.predictor = build_sam2_video_predictor(config, checkpoint, device=device)

    def add_objects_input_annotations(
        self, video_path: str, input_mask: np.ndarray
    ) -> None:
        """
        Adds object annotations to the input mask for a given video frame.

        :param video_path: Path to the video file.
        :param input_mask: Input mask for the video frame.
        :return: Dictionary containing the updated input mask and its corresponding bounding boxes.
        """

        _, bboxes = self._get_bboxes_from_instance_mask(input_mask)
        labels, points = self._get_points_from_instance_mask(input_mask)
        points = np.array(points, dtype=np.float32)
        ann_frame_idx = 0

        for i in range(len(points)):
            ann_obj_id = i + 1
            ann_obj_labels = np.zeros_like(labels)
            ann_obj_labels[i] = 1
            _, object_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=ann_obj_labels,
                box=bboxes[i],
            )

    def _propagate_predictions_in_video(self) -> dict:
        """
        Propagates the predictions in the video using the initialized state.

        :return: None
        """
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        return video_segments

    def predict(self, video_path: str, input_mask: np.ndarray) -> dict:
        """
        Predicts the segmentation mask for a given video frame using SAM2.

        :param video_path: Path to the video file.
        :param input_mask: Input mask for the video frame.
        :return: Dictionary containing the predicted masks and their corresponding bounding boxes.
        """

        self.inference_state = self.predictor.init_state(video_path)
        self.add_objects_input_annotations(video_path, input_mask)
        video_segments = self._propagate_predictions_in_video()
        self.predictor.reset_state(self.inference_state)

        return video_segments

    @staticmethod
    def _get_bboxes_from_instance_mask(
        label_img: np.ndarray,
    ) -> tuple[list[int], list[list[int, int, int, int]]]:
        """
        Extract bounding boxes for all object IDs present in the label image.

        :param label_img: Label image with object instance IDs.
        :return: Tuple containing a list of instance IDs and a list of their bounding boxes (x1, y1, x2, y2).
        """
        instance_ids = np.unique(label_img)
        instance_ids = instance_ids[instance_ids != 0]

        bboxes = []
        for instance_id in instance_ids:
            mask = (label_img == instance_id).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            x, y, w, h = cv2.boundingRect(mask)
            bboxes.append([x, y, x + w, y + h])

        return instance_ids.tolist(), bboxes

    @staticmethod
    def _get_points_from_instance_mask(
        label_img: np.ndarray,
    ) -> tuple[list[int], list[list[int, int]]]:
        """
        Extract representative points (e.g., center of mass) for all object IDs in the label image.

        :param label_img: Label image with object instance IDs.
        :return: Tuple containing a list of instance IDs and a list of their points (x, y).
        """
        instance_ids = np.unique(label_img)
        instance_ids = instance_ids[instance_ids != 0]

        points = []
        for instance_id in instance_ids:
            mask = (label_img == instance_id).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            # Use moments to find the center of mass
            M = cv2.moments(mask)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points.append([cx, cy])

        return instance_ids.tolist(), points
