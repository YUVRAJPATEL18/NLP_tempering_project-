"""Dataset pipeline tests: validation of corrupt/missing files, transforms,
group-aware split integrity."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from conftest import make_document_bytes
from idguard.dataset import (Sample, TamperDataset, build_group_splits,
                             build_transforms, load_manifest, split_digest,
                             validate_images)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _mk_samples() -> list:
    out = []
    # group g1: 1 genuine + 2 fakes, group g2..g6: genuine (+1 fake for g2)
    specs = [
        ("reals/a_00.jpg", 0, "g_a00"), ("fakes/a_00_fake_6_1.jpg", 1, "g_a00"),
        ("fakes/a_00_fake_6_2.jpg", 1, "g_a00"),
        ("reals/b_00.jpg", 0, "g_b00"), ("fakes/b_00_fake_6_3.jpg", 1, "g_b00"),
        ("reals/c_00.jpg", 0, "g_c00"),
        ("reals/d_00.jpg", 0, "g_d00"),
        ("reals/e_00.jpg", 0, "g_e00"),
        ("reals/f_00.jpg", 0, "g_f00"),
    ]
    for path, label, group in specs:
        out.append(Sample(image_path=path, label=label,
                          label_name="reals" if label == 0 else "fakes",
                          group_id=group))
    return out


def test_validate_images_flags_corrupt_and_missing(tmp_path: Path):
    samples = _mk_samples()
    # valid, corrupt (garbage bytes), missing
    _write(tmp_path / "reals/a_00.jpg", make_document_bytes())
    _write(tmp_path / "fakes/a_00_fake_6_1.jpg", b"not a jpeg at all")
    # a_00_fake_6_2 intentionally not created
    report = validate_images(tmp_path, samples)
    assert report.total == len(samples)
    assert report.ok == 1, "only reals/a_00.jpg was written"
    assert "fakes/a_00_fake_6_1.jpg" in report.unreadable
    assert "fakes/a_00_fake_6_2.jpg" in report.missing
    assert len(report.missing) == len(samples) - 2
    assert len(report.invalid) == len(samples) - 1
    assert report.to_dict()["unreadable"]


def test_manifest_loader(tmp_path: Path):
    p = tmp_path / "manifest.csv"
    p.write_text(
        "image_path,label,label_name,group_id,nationality,manipulation_type,tamper_field,src_image\n"
        "reals/a_00.jpg,0,reals,g_a00,alb,,,\n"
        "fakes/a_00_fake_6_1.jpg,1,fakes,g_a00,alb,Inpaint_and_Rewrite,name,reals/a_00.jpg\n")
    samples = load_manifest(p)
    assert [s.label for s in samples] == [0, 1]
    assert samples[1].manipulation_type == "Inpaint_and_Rewrite"
    with pytest.raises(ValueError):
        (tmp_path / "bad.csv").write_text("image_path,label\nx,7\n")
        load_manifest(tmp_path / "bad.csv")


def test_group_split_disjoint_and_deterministic():
    samples = _mk_samples()
    s1 = build_group_splits(samples, ratios=(0.7, 0.15, 0.15), seed=42)
    s2 = build_group_splits(samples, ratios=(0.7, 0.15, 0.15), seed=42)
    assert s1["train"] == s2["train"] and s1["val"] == s2["val"]
    assert split_digest(s1) == split_digest(s2)

    g = s1["_groups"]
    tr, va, te = set(g["train"]), set(g["val"]), set(g["test"])
    assert not (tr & va) and not (tr & te) and not (va & te), "group leakage"
    total = len(s1["train"]) + len(s1["val"]) + len(s1["test"])
    assert total == len(samples)
    # the genuine image and both its fakes never get split apart
    g1 = {s.image_path for s in samples if s.group_id == "g_a00"}
    holders = [name for name in ("train", "val", "test") if g1 & set(s1[name])]
    assert len(holders) == 1


def test_split_digest_changes_with_seed():
    samples = _mk_samples()
    a = build_group_splits(samples, seed=1)
    b = build_group_splits(samples, seed=2)
    assert split_digest(a) != split_digest(b)


CFG_DATA = {
    "image_size": 64,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
}


def test_transforms_shapes_and_normalization():
    tf_eval = build_transforms(CFG_DATA, augment=False)
    img = Image.open(__import__("io").BytesIO(make_document_bytes())).convert("RGB")
    t = tf_eval(img)
    assert t.shape == (3, 64, 64) and t.dtype == torch.float32
    # eval path must be deterministic
    assert tf_eval(img).equal(t)
    # augmentation path still returns a valid tensor of the same shape
    tf_train = build_transforms(CFG_DATA, augment=True)
    t2 = tf_train(img)
    assert t2.shape == (3, 64, 64) and torch.isfinite(t2).all()


def test_dataset_loads_and_reports_corrupt(tmp_path: Path):
    _write(tmp_path / "reals/a_00.jpg", make_document_bytes())
    _write(tmp_path / "reals/b_00.jpg", b"broken")
    good = Sample(image_path="reals/a_00.jpg", label=0, label_name="reals", group_id="g")
    bad = Sample(image_path="reals/b_00.jpg", label=0, label_name="reals", group_id="g")
    ds = TamperDataset(tmp_path, [good, bad], build_transforms(CFG_DATA, augment=False))
    item = ds[0]
    assert item["image"].shape == (3, 64, 64) and item["label"] == 0
    with pytest.raises(RuntimeError, match="failed to load image"):
        _ = ds[1]
