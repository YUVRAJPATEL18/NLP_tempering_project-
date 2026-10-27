"""Grad-CAM tampering heatmaps aligned to the source image.

The CAM is computed on the tamper logit, upsampled bilinearly to the network
input resolution and, when needed, to the original image size — so overlays are
pixel-aligned with the source. CAMs are *explanations*, not ground truth.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .model import TamperingModel


class GradCAM:
    def __init__(self, model: TamperingModel, target_layer: str = "layer4"):
        self.model = model
        self.layer = getattr(model.backbone, target_layer)
        self._acts: Optional[torch.Tensor] = None
        self._handle = self.layer.register_forward_hook(self._save_acts)

    def _save_acts(self, module, inputs, output):  # noqa: ANN001
        self._acts = output

    def close(self) -> None:
        self._handle.remove()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized CAMs [B, H, W] at the input resolution (0..1)."""
        self._acts = None
        with torch.enable_grad():
            logits = self.model(x)
            if self._acts is None:
                raise RuntimeError("target layer produced no activations")
            acts = self._acts
            grads = torch.autograd.grad(logits.sum(), acts, retain_graph=False,
                                        create_graph=False)[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)          # GAP over spatial
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))  # [B,1,h,w]
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        flat_max = cam.flatten(1).max(dim=1).values.clamp_min(1e-8)
        cam = cam / flat_max.view(-1, 1, 1)
        return cam.detach()


def cam_to_regions(cam: np.ndarray, cam_threshold: float = 0.35,
                   min_area_frac: float = 0.005) -> list:
    """Connected components of a normalized CAM -> normalized boxes [x0,y0,x1,y1].

    Returns boxes sorted by peak activation, capped at 5 regions.
    """
    h, w = cam.shape
    mask = cam >= cam_threshold
    if not mask.any():
        return []
    try:
        from scipy import ndimage  # optional
        labeled, n = ndimage.label(mask)
        boxes = []
        for i in range(1, n + 1):
            ys, xs = np.where(labeled == i)
            if len(ys) < min_area_frac * h * w:
                continue
            boxes.append((float(cam[ys, xs].max()),
                          [round(xs.min() / w, 4), round(ys.min() / h, 4),
                           round((xs.max() + 1) / w, 4), round((ys.max() + 1) / h, 4)]))
        boxes.sort(reverse=True)
        return [b for _, b in boxes[:5]]
    except ImportError:
        # no scipy: bounding box of the whole thresholded mask (coarse)
        ys, xs = np.where(mask)
        if len(ys) < min_area_frac * h * w:
            return []
        return [[round(xs.min() / w, 4), round(ys.min() / h, 4),
                 round((xs.max() + 1) / w, 4), round((ys.max() + 1) / h, 4)]]


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45,
                size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Blend a CAM onto an RGB PIL image with a jet-style colormap."""
    import matplotlib.cm as cm

    img = image.convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    cam_img = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8))
    if cam_img.size != img.size:
        cam_img = cam_img.resize(img.size, Image.BILINEAR)
    heat = (cm.get_cmap("jet")(np.asarray(cam_img, dtype=np.float32) / 255.0)[:, :, :3] * 255)
    heat = heat.astype(np.uint8)
    base = np.asarray(img, dtype=np.float32)
    out = (1 - alpha) * base + alpha * heat.astype(np.float32)
    return Image.fromarray(out.astype(np.uint8))
