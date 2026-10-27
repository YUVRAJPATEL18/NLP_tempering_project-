# Datasets

## SIDTD — Synthetic dataset of ID and Travel Documents

The tampering model is trained and evaluated on **SIDTD**, the primary dataset
required by the project brief. It is the only dataset acquired: one small,
well-labelled, identity-document-specific corpus instead of several large ones.

| Item | Value |
|---|---|
| Dataset | SIDTD (Boned, Talarmain et al., 2024) |
| TC11 page | https://tc11.cvc.uab.es/datasets/SIDTD_1 (slow/unreachable at acquisition time; figshare mirror used) |
| Acquired from | figshare article **27136242**, API record `https://api.figshare.com/v2/articles/27136242` |
| DOI / version | `10.6084/m9.figshare.27136242.v1` — version 1, published 2024-12-19 |
| Licence | **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/) — verified via the figshare API record |
| Paper | https://doi.org/10.1038/s41597-024-04160-9 |
| Citation | Talarmain, Maxime; Riera, Carlos Boned (2024). *Synthetic dataset of ID and Travel Documents*. figshare. https://doi.org/10.6084/m9.figshare.27136242.v1 |

### What was downloaded (and what was not)

The full SIDTD corpus is **79.4 GB** (54.5 GB videos, 23.5 GB clips). Only the
image-level task data were acquired:

| File | Size | MD5 (published) | Status |
|---|---|---|---|
| `templates.zip` (figshare file 49511154) | 1,273,966,468 B | `a790947e773686a2f1b2ab35be0ae38d` | downloaded + MD5 verified |
| `videos.zip` (54.5 GB), `clips.zip` (23.5 GB) | — | — | **deliberately skipped** |
| official split archives (< 1 MB) | — | — | fetched to `datasets/sidtd/official_splits/` for the record |

### Acquisition

```bash
.venv/bin/python scripts/download_sidtd.py          # resumable, 6-connection chunked download
.venv/bin/python scripts/prepare_dataset.py         # validate + group-aware split + quality table
```

The script resolves the file URL from the figshare API every run (no hard-coded
or guessed URLs), verifies the published MD5, extracts only
`templates/Images/**` and `templates/Annotations/**`, and regenerates
`manifest.csv` idempotently.

### Layout (after extraction)

```
datasets/sidtd/
├── manifest.csv                     # labels + provenance per image
├── splits.json                      # group-aware 70/15/15 split (seed 42, digest 697c850e9401825e)
├── quality.csv                      # forensic quality metrics per image
├── quality_meta.json                # degraded-quality threshold (pre-registered rule)
├── validation_report.json           # integrity check results (2222/2222 OK)
├── official_splits/                 # archived official CSVs (NOT used for training)
└── templates/
    ├── Images/reals/                # 1,000 genuine template images
    ├── Images/fakes/                # 1,222 tampered template images
    ├── Annotations/reals/*.json     # VIA rectangle annotations, 14 document fields × 100 images
    └── Annotations/fakes/*.json     # per-fake metadata: ctype, src, manipulated field
```

### Labels and annotations

- **Image-level labels**: `reals` = 0 (genuine), `fakes` = 1 (tampered) —
  official dataset labels, nothing invented.
- **Class counts**: 1,000 genuine / 1,222 tampered (2,222 total).
- **Manipulation types** (official `ctype`): `Inpaint_and_Rewrite` 1,077,
  `Crop_and_Replace` 145.
- **Tampered fields**: name, surname, expiry_date, gender, birth_date, MRZ
  lines, photo, etc. — the field that was altered on each fake.
- **Region masks**: SIDTD provides **no official pixel-level tampering masks**.
  For 1,190/1,222 fakes an *approximate* region mask is **derived** from the
  official annotations (`src` image + `tamper_field` → VIA field rectangle).
  All localization numbers are labelled as derived, never as official
  pixel ground truth.

### Splits (leakage prevention)

A genuine template and every tampered variant derived from it form one *group*
(base document ID). Splits are group-disjoint and deterministic:

| split | images | genuine | tampered | base documents |
|---|---|---|---|---|
| train | 1,556 | 698 | 858 | 698 |
| val | 333 | 153 | 180 | 153 |
| test | 333 | 149 | 184 | 149 |
| **total** | **2,222** | **1,000** | **1,222** | **1,000** |

- Seed 42, ratios 0.70/0.15/0.15, split digest `697c850e9401825e`.
- The **official** hold-out split was inspected and **not used**: it leaks base
  documents across splits (55 train∩val, 64 train∩test, 11 val∩test groups) and
  covers only 819 images. It remains archived under `official_splits/`.

### Quality / degraded strata

`quality_score = 0.6·sharpness + 0.4·contrast` (both in [0,1], definitions in
`src/idguard/forensic.py`). The degraded threshold **0.8251** is the 33rd
percentile of *train-split genuine* quality scores — fixed before any test
evaluation. Observed range over the dataset: 0.6372–0.9923.

### Restrictions and honesty notes

- All SIDTD documents are **synthetic** (AI-generated templates, printed
  counterfeits of those templates). They must never be described as real
  government-issued identity documents.
- Images are used under CC BY 4.0: attribution must accompany redistribution
  (see citation above); commercial and research use are permitted.
- Blur, compression or poor illumination alone never makes an image *tampered*
  — quality only feeds the separate `quality_score` signal.
- The test split is used exactly once per model for final reporting; thresholds
  are fitted on validation only.

### IMD2020 / MIDV-2020 (considered, not acquired)

- **IMD2020** (general manipulation detection) — not downloaded: SIDTD already
  provides document-specific labels, and the brief prefers one small
  well-labelled dataset. It remains an optional external-validation source.
- **MIDV-2020** — SIDTD's genuine templates are derived from MIDV-2020; the raw
  MIDV videos (video capture quality) were not downloaded (size). No genuine
  image is labelled tampered based on capture quality.
