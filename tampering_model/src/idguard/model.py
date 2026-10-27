"""TamperingModel — the shared model contract for IDGuard's tamper detection.

One architecture (ImageNet-pretrained ResNet-18 + binary head) is used for both
the baseline linear probe and the improved fine-tune; only the training recipe
differs, so baseline-vs-improved comparisons isolate the recipe.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models

MODEL_FORMAT_VERSION = 1


class TamperingModel(nn.Module):
    """Binary genuine/tampered classifier.

    forward(x) -> logits of shape [B] (raw logits; apply sigmoid for probability).
    """

    def __init__(self, arch: str = "resnet18", pretrained: bool = True,
                 dropout: float = 0.3, freeze_backbone: bool = False):
        super().__init__()
        if arch != "resnet18":
            raise ValueError(f"unsupported architecture: {arch}")
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.arch = arch
        self.pretrained = pretrained
        self.backbone = models.resnet18(weights=weights)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1)
        self.freeze_backbone_flag = False
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(self.dropout(feats)).squeeze(-1)

    # -- parameter groups ---------------------------------------------------
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze_backbone_flag = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.freeze_backbone_flag = False

    def parameter_groups(self, lr: float, backbone_lr: Optional[float] = None,
                         weight_decay: float = 1e-4) -> list:
        head = [p for p in self.head.parameters() if p.requires_grad]
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        groups = [{"params": head, "lr": lr, "weight_decay": weight_decay}]
        if backbone:
            groups.append({"params": backbone,
                           "lr": lr if backbone_lr is None else backbone_lr,
                           "weight_decay": weight_decay})
        return groups

    def model_config(self) -> Dict[str, Any]:
        return {
            "arch": self.arch,
            "pretrained": self.pretrained,
            "dropout": float(self.dropout.p),
            "freeze_backbone": self.freeze_backbone_flag,
        }


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model: TamperingModel, **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_state": model.state_dict(),
        "model_config": model.model_config(),
    }
    payload.update(meta)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"{path} is not an IDGuard tampering checkpoint")
    if ckpt.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format_version in {path}")
    return ckpt


def model_from_checkpoint(path: Path, device: str = "cpu",
                          freeze_backbone: Optional[bool] = None) -> tuple:
    """Rebuild the model from a checkpoint -> (model, checkpoint_dict)."""
    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = dict(ckpt["model_config"])
    if freeze_backbone is not None:
        cfg["freeze_backbone"] = freeze_backbone
    model = TamperingModel(arch=cfg["arch"], pretrained=False,
                           dropout=cfg["dropout"], freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    if freeze_backbone:
        model.freeze_backbone()
    model.to(device)
    model.eval()
    return model, ckpt


def get_default_model(config: Optional[Dict] = None, checkpoint: Optional[Path] = None,
                      device: str = "cpu") -> TamperingModel:
    """Build the default IDGuard tampering model.

    - checkpoint given: weights come from that checkpoint (nothing re-downloaded
      beyond the architecture definition; `pretrained` in the checkpoint governs
      config only — weights are always overridden by the state dict).
    - no checkpoint: ImageNet-pretrained backbone + freshly initialized head
      (the documented baseline starting point).
    """
    if checkpoint is not None:
        model, _ = model_from_checkpoint(Path(checkpoint), device=device)
        return model
    if config is None:
        model = TamperingModel()
    else:
        mc = config["model"]
        model = TamperingModel(arch=mc.get("architecture", "resnet18"),
                               pretrained=bool(mc.get("pretrained", True)),
                               dropout=float(mc.get("dropout", 0.3)))
    model.to(device)
    model.eval()
    return model
