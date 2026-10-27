"""Production inference: model + quality + decision in one structured call.

The model instance is loaded once per process (eval mode, inference_mode);
preprocessing happens exactly once per image and is shared with Grad-CAM.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image, UnidentifiedImageError

from .decision_engine import DecisionResult, decide
from .forensic import quality_metrics
from .heatmap import GradCAM, cam_to_regions, overlay_cam
from .model import model_from_checkpoint, get_default_model

ROOT = Path(__file__).resolve().parents[2]

ImageLike = Union[Image.Image, np.ndarray, bytes]


class InvalidImageError(ValueError):
    """Raised for unsupported, corrupt or unreadable input images."""


@dataclass
class TamperingPrediction:
    """IDGuard tampering-inference result (stable API schema)."""

    tampering_probability: float
    tampering_detected: bool
    quality_score: float
    ocr_reliability: Optional[float]
    predicted_regions: Optional[List[List[float]]]
    decision: str
    decision_reasons: List[str]
    model_version: str
    # non-serialized extras
    heatmap: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    timings_ms: Dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self, include_heatmap: bool = False,
                heatmap_png_size: Optional[Tuple[int, int]] = None) -> Dict:
        out: Dict = {
            "tampering_probability": round(self.tampering_probability, 4),
            "tampering_detected": self.tampering_detected,
            "quality_score": round(self.quality_score, 4),
            "ocr_reliability": self.ocr_reliability,
            "predicted_regions": self.predicted_regions,
            "decision": self.decision,
            "decision_reasons": list(self.decision_reasons),
            "model_version": self.model_version,
        }
        if include_heatmap and self.heatmap is not None:
            import base64

            if self.timings_ms:
                out["timings_ms"] = {k: round(v, 1) for k, v in self.timings_ms.items()}
            buf = io.BytesIO()
            cam_img = Image.fromarray((np.clip(self.heatmap, 0, 1) * 255).astype(np.uint8), "L")
            cam_img.save(buf, format="PNG")
            out["heatmap_png_base64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            return out
        if self.timings_ms:
            out["timings_ms"] = {k: round(v, 1) for k, v in self.timings_ms.items()}
        return out


def _as_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim not in (2, 3) or image.dtype != np.uint8:
            raise InvalidImageError("numpy image must be a uint8 HxW or HxWx3 array")
        try:
            return Image.fromarray(image).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise InvalidImageError(f"cannot interpret array as image: {e}") from e
    if isinstance(image, (bytes, bytearray)):
        try:
            with Image.open(io.BytesIO(bytes(image))) as im:
                im.load()
                return im.convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImageError("unsupported or corrupt image data") from e
    raise InvalidImageError(f"unsupported image type: {type(image).__name__}")


class TamperingPredictor:
    """Loads the model once and serves structured tampering predictions."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None,
                 checkpoint: Optional[Union[str, Path]] = None, device: str = "auto"):
        t0 = time.perf_counter()
        cfg_path = Path(config_path) if config_path else ROOT / "config.yaml"
        with open(cfg_path) as f:
            self.cfg: Dict = yaml.safe_load(f)
        self.root = ROOT

        self.device = self._resolve_device(device or self.cfg["train"].get("device", "auto"))
        ckpt_path = Path(checkpoint) if checkpoint else None
        if ckpt_path is None and self.cfg["model"].get("checkpoint"):
            cand = self.root / self.cfg["model"]["checkpoint"]
            if cand.exists():
                ckpt_path = cand

        # A missing checkpoint is a deployment error, not something to paper over:
        # silently serving an untrained head from a fraud-detection API is worse
        # than failing at startup.
        if ckpt_path is not None and not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

        self._ckpt: Optional[Dict] = None
        if ckpt_path is not None:
            self.model, self._ckpt = model_from_checkpoint(ckpt_path, device=self.device)
            self.model_version = self._ckpt.get("model_version", f"ckpt:{ckpt_path.stem}")
            self.temperature = float(self._ckpt.get("temperature", 1.0))
            self.thresholds = dict(self._ckpt.get("thresholds_decision") or {})
        else:
            self.model = get_default_model(config=self.cfg, device=self.device)
            self.model_version = "resnet18-imagenet-untrained-head"
            self.temperature = 1.0
            self.thresholds = {}

        # fall back to config thresholds (safe defaults) when absent
        decision_cfg = self.cfg["decision"]
        thr = self.cfg["thresholds"]
        if thr.get("enforce_config_thresholds", True):
            self.thresholds["tamper_high"] = float(thr["tamper_high"])
            self.thresholds["tamper_low"] = float(thr["tamper_low"])
        else:
            self.thresholds.setdefault("tamper_high", float(thr["tamper_high"]))
            self.thresholds.setdefault("tamper_low", float(thr["tamper_low"]))
        self.thresholds.setdefault("rescan_below", float(self.cfg["quality"]["rescan_below"]))
        self.thresholds.setdefault("quality_degraded",
                                   float(self.cfg["quality"]["degraded_threshold"]))
        self.thresholds.setdefault("ocr_min", float(decision_cfg.get("ocr_min", 0.55)))
        self.gate_default = bool(decision_cfg.get("gate_enabled", True))
        # Patch (tiled) checkpoints score the document at native resolution; the
        # checkpoint records its own patch geometry so inference stays generic.
        self.patch_cfg = (self._ckpt or {}).get("patch")

        # The checkpoint owns its preprocessing: a 512 px model must not be fed
        # the 256 px default from config.yaml.
        self._data_cfg = dict(self.cfg["data"])
        ck = self._ckpt or {}
        if ck.get("image_size"):
            self._data_cfg["image_size"] = int(ck["image_size"])
        for key in ("normalize_mean", "normalize_std"):
            if ck.get(key):
                self._data_cfg[key] = ck[key]

        self._tf = None  # built lazily to avoid heavy imports in tests
        self._gradcam: Optional[GradCAM] = None
        self.load_seconds = time.perf_counter() - t0

        if self.cfg.get("inference", {}).get("warmup", True) and self.device != "cpu":
            try:
                s = int(self._data_cfg["image_size"])
                with torch.inference_mode():
                    self.model(torch.zeros(1, 3, s, s, device=self.device))
            except Exception:  # noqa: BLE001 - warmup is best-effort
                pass

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(spec: str) -> str:
        from .train import resolve_device

        return resolve_device(spec)

    def _transforms(self, augment: bool = False):
        if self._tf is None:
            from .dataset import build_transforms

            self._tf = build_transforms(self._data_cfg, augment=False)
        return self._tf

    def _prob(self, img_t: torch.Tensor, need_cam: bool) -> Tuple[float, Optional[np.ndarray]]:
        x = img_t.unsqueeze(0).to(self.device)
        if need_cam:
            if self._gradcam is None:
                layer = self.cfg.get("heatmap", {}).get("target_layer", "layer4")
                self._gradcam = GradCAM(self.model, target_layer=layer)
            cam = self._gradcam(x)[0].cpu().numpy()
            with torch.inference_mode():
                logit = float(self.model(x).reshape(-1)[0])
            return logit, cam
        with torch.inference_mode():
            logit = float(self.model(x).reshape(-1)[0])
        return logit, None

    # ------------------------------------------------------------------
    def predict(self, image: ImageLike,
                ocr_reliability: Optional[float] = None,
                include_heatmap: bool = False,
                gate_enabled: Optional[bool] = None) -> TamperingPrediction:
        """Analyze one document image.

        ocr_reliability is an external signal from the OCR pipeline (0..1) or
        None when OCR did not run — it never modifies the forensic probability.
        """
        t0 = time.perf_counter()
        pil = _as_pil(image)
        t_load = time.perf_counter()

        qmetrics = quality_metrics(pil)
        t_quality = time.perf_counter()

        want_map = (include_heatmap
                    or bool(self.cfg.get("inference", {}).get("return_regions", True)))
        want_regions = bool(self.cfg.get("inference", {}).get("return_regions", True))
        regions: Optional[List[List[float]]] = None
        cam: Optional[np.ndarray] = None

        if self.patch_cfg:
            from .patch import regions_from_tiles, score_document

            t_pre = time.perf_counter()
            scored = score_document(self.model, pil, self._transforms(), self.device,
                                    patch=int(self.patch_cfg["patch_size"]),
                                    stride=int(self.patch_cfg["stride"]),
                                    temperature=self.temperature, heatmap=want_map)
            prob = float(scored["probability"])
            cam = scored.get("heatmap")
            t_infer = time.perf_counter()
            if want_regions:
                regions = regions_from_tiles(scored["boxes"], scored["tile_probs"],
                                             pil.width, pil.height,
                                             threshold=max(float(self.thresholds["tamper_low"]), 0.30))
        else:
            img_t = self._transforms()(pil)
            t_pre = time.perf_counter()
            logit, cam = self._prob(img_t, need_cam=want_map)
            prob = float(1.0 / (1.0 + np.exp(-np.clip(logit / self.temperature, -60, 60))))
            t_infer = time.perf_counter()
            if want_regions and cam is not None:
                hc = self.cfg.get("heatmap", {})
                regions = cam_to_regions(
                    cam, cam_threshold=float(hc.get("cam_threshold", 0.35)),
                    min_area_frac=float(hc.get("region_min_area_frac", 0.005)))

        res: DecisionResult = decide(
            tampering_probability=prob,
            quality_score=qmetrics["quality_score"],
            ocr_reliability=ocr_reliability,
            thresholds=self.thresholds,
            gate_enabled=self.gate_default if gate_enabled is None else bool(gate_enabled),
        )
        t_end = time.perf_counter()

        return TamperingPrediction(
            tampering_probability=prob,
            tampering_detected=res.decision == "POTENTIAL_TAMPERING",
            quality_score=qmetrics["quality_score"],
            ocr_reliability=ocr_reliability,
            predicted_regions=regions,
            decision=res.decision,
            decision_reasons=res.reasons,
            model_version=self.model_version,
            heatmap=cam,
            timings_ms={
                "decode": (t_load - t0) * 1e3,
                "quality": (t_quality - t_load) * 1e3,
                "preprocess": (t_pre - t_quality) * 1e3,
                "inference": (t_infer - t_pre) * 1e3,
                "postprocess": (t_end - t_infer) * 1e3,
                "total": (t_end - t0) * 1e3,
            },
        )

    def predict_with_overlay(self, image: ImageLike, **kw) -> Tuple[TamperingPrediction, Image.Image]:
        """Convenience: prediction + source-aligned CAM overlay for reports."""
        kw["include_heatmap"] = True
        pil = _as_pil(image)
        pred = self.predict(pil, **kw)
        overlay = overlay_cam(pil, pred.heatmap) if pred.heatmap is not None else pil.copy()
        return pred, overlay
