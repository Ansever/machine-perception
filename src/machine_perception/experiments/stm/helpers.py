from __future__ import division

import torch
import torch.nn.functional as F

import numpy as np

from scipy.ndimage import binary_dilation


def to_cuda(xs: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda()
    elif torch.backends.mps.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.to("mps") for x in xs]
        else:
            return xs.to("mps")
    else:
        return xs


def pad_divide_by(
    in_list: list[torch.Tensor], d: int, in_size: tuple[int, int]
) -> tuple[list[torch.Tensor], tuple[int, int, int, int]]:
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array


def overlay_davis(
    image: np.ndarray,
    mask: np.ndarray,
    colors: list[int] = [255, 0, 0],
    cscale: int = 2,
    alpha: float = 0.4,
):
    """Overlay segmentation on top of RGB image. from davis official"""
    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image * alpha + np.ones(image.shape) * (1 - alpha) * np.array(
            colors[object_id]
        )
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)
