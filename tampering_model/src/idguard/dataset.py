"""Dataset loading, validation and deterministic group-aware splits for SIDTD.

Leakage policy: the *group* key is the base document ID (filename with the
``_fake_*`` suffix removed). A genuine template and every tampered variant
derived from it always stay in the same split.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MANIFEST_FIELDS = [
    "image_path", "label", "label_name", "group_id", "nationality",
    "manipulation_type", "tamper_field", "src_image",
]


@dataclass
class Sample:
    image_path: str
    label: int
    label_name: str
    group_id: str
    nationality: str = ""
    manipulation_type: str = ""
    tamper_field: str = ""
    src_image: str = ""


def load_manifest(path: Path) -> List[Sample]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    missing = set(MANIFEST_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    out = []
    for r in rows:
        if r["label"] not in ("0", "1"):
            raise ValueError(f"bad label {r['label']!r} for {r['image_path']}")
        out.append(Sample(
            image_path=r["image_path"],
            label=int(r["label"]),
            label_name=r["label_name"],
            group_id=r["group_id"],
            nationality=r.get("nationality", ""),
            manipulation_type=r.get("manipulation_type", ""),
            tamper_field=r.get("tamper_field", ""),
            src_image=r.get("src_image", ""),
        ))
    return out


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
@dataclass
class ValidationReport:
    total: int = 0
    ok: int = 0
    missing: List[str] = field(default_factory=list)
    unreadable: List[str] = field(default_factory=list)
    wrong_mode: List[str] = field(default_factory=list)

    @property
    def invalid(self) -> List[str]:
        return self.missing + self.unreadable + self.wrong_mode

    def to_dict(self) -> Dict:
        return asdict(self)


def validate_images(root: Path, samples: Sequence[Sample],
                    check_pixels: bool = True) -> ValidationReport:
    """Detect missing, truncated and undecodable files.

    `check_pixels=True` forces a full decode (slower but catches truncated JPEGs
    that PIL's header-only verify() misses).
    """
    rep = ValidationReport(total=len(samples))
    for s in samples:
        p = root / s.image_path
        if not p.exists():
            rep.missing.append(s.image_path)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
            if check_pixels:
                with Image.open(p) as im:
                    im.load()
                    if im.mode not in ("RGB", "L", "CMYK", "P"):
                        rep.wrong_mode.append(s.image_path)
                        continue
                    if im.width < 32 or im.height < 32:
                        rep.wrong_mode.append(s.image_path + f" ({im.width}x{im.height})")
                        continue
        except (UnidentifiedImageError, OSError, ValueError):
            rep.unreadable.append(s.image_path)
            continue
        rep.ok += 1
    return rep


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------
def build_group_splits(samples: Sequence[Sample],
                       ratios: Sequence[float] = (0.70, 0.15, 0.15),
                       seed: int = 42) -> Dict[str, List[str]]:
    """Deterministic 70/15/15 split, group-disjoint by base document.

    Groups are shuffled with `seed`, then assigned greedily to the split with
    the largest remaining capacity (ties broken by fake-count deficit), which
    keeps image counts near target while balancing both classes.
    """
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must be three values summing to 1")
    names = ["train", "val", "test"]

    groups: Dict[str, List[Sample]] = {}
    for s in samples:
        groups.setdefault(s.group_id, []).append(s)

    total = len(samples)
    n_fake = sum(s.label for s in samples)
    targets = [r * total for r in ratios]
    fake_targets = [r * n_fake for r in ratios]

    gids = sorted(groups)
    random.Random(seed).shuffle(gids)

    counts = [0, 0, 0]
    fakes = [0, 0, 0]
    assign: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    assign_groups: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for gid in gids:
        g = groups[gid]
        g_fake = sum(s.label for s in g)
        # primary: biggest remaining image capacity; tie-break: fake deficit
        best = max(range(3), key=lambda i: (
            targets[i] - counts[i],
            fake_targets[i] - fakes[i],
            -counts[i],
        ))
        counts[best] += len(g)
        fakes[best] += g_fake
        assign[names[best]].extend(s.image_path for s in g)
        assign_groups[names[best]].append(gid)

    # assertions: group disjointness + full coverage
    seen: Dict[str, str] = {}
    for name in names:
        for gid in assign_groups[name]:
            if gid in seen:
                raise AssertionError(f"group {gid} in both {seen[gid]} and {name}")
            seen[gid] = name
    covered = sum(len(assign[n]) for n in names)
    if covered != total:
        raise AssertionError(f"coverage {covered} != {total}")

    assign["_groups"] = assign_groups  # type: ignore[assignment]
    return assign


def split_digest(splits: Dict[str, List[str]]) -> str:
    """Stable hash of a split assignment (recorded in checkpoints)."""
    h = hashlib.sha256()
    for name in ("train", "val", "test"):
        for p in sorted(splits[name]):
            h.update(f"{name}:{p}\n".encode())
    return h.hexdigest()[:16]


def save_splits(path: Path, splits: Dict[str, List[str]], meta: Optional[Dict] = None) -> None:
    payload = {k: v for k, v in splits.items() if not k.startswith("_")}
    payload["_meta"] = meta or {}
    path.write_text(json.dumps(payload, indent=1))


def load_splits(path: Path) -> Dict[str, List[str]]:
    data = json.loads(Path(path).read_text())
    data.pop("_meta", None)
    groups = data.pop("_groups", None)
    if groups:
        data["_groups"] = groups  # type: ignore[assignment]
    return data


# ---------------------------------------------------------------------------
# transforms / dataset
# ---------------------------------------------------------------------------
def build_transforms(cfg_data: Dict, augment: bool) -> transforms.Compose:
    """Preprocessing is deliberately conservative: square resize to the
    configured resolution, optional mild affine/photometric jitter, then the
    backbone's ImageNet normalization. Evaluation always uses the raw,
    unmodified dataset images (resize + normalize only)."""
    size = int(cfg_data["image_size"])
    ops = [transforms.Resize((size, size), interpolation=InterpolationMode.BILINEAR)]
    if augment:
        ops += [
            transforms.RandomAffine(degrees=3.0, translate=(0.02, 0.02),
                                    scale=(0.97, 1.03)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg_data["normalize_mean"], std=cfg_data["normalize_std"]),
    ]
    return transforms.Compose(ops)


class TamperDataset(Dataset):
    """Image-level genuine/tampered dataset.

    Returns dicts: image (CHW float tensor), label (0/1), index (row position).
    """

    def __init__(self, root: Path, samples: Sequence[Sample],
                 image_tf: transforms.Compose):
        self.root = Path(root)
        self.samples = list(samples)
        self.tf = image_tf

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict:
        s = self.samples[i]
        try:
            with Image.open(self.root / s.image_path) as im:
                img = im.convert("RGB")
                tensor = self.tf(img)
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"failed to load image {s.image_path}: {e}") from e
        return {"image": tensor, "label": s.label, "index": i}

    @property
    def labels(self) -> List[int]:
        return [s.label for s in self.samples]


def samples_for_split(samples: Sequence[Sample], splits: Dict[str, List[str]],
                      which: str) -> List[Sample]:
    wanted = set(splits[which])
    out = [s for s in samples if s.image_path in wanted]
    missing = wanted - {s.image_path for s in out}
    if missing:
        raise ValueError(f"split {which} references {len(missing)} paths absent "
                         f"from manifest (e.g. {next(iter(missing))})")
    return out


def make_loaders(root: Path, samples: Sequence[Sample], splits: Dict[str, List[str]],
                 cfg: Dict, augment: bool, batch_size: int) -> Dict[str, DataLoader]:
    tf_train = build_transforms(cfg["data"], augment=augment)
    tf_eval = build_transforms(cfg["data"], augment=False)
    nw = int(cfg["data"].get("num_workers", 0))
    out = {}
    for name, is_train in (("train", True), ("val", False), ("test", False)):
        ds = TamperDataset(root, samples_for_split(samples, splits, name),
                           tf_train if is_train else tf_eval)
        out[name] = DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                               num_workers=nw, drop_last=False)
    return out
