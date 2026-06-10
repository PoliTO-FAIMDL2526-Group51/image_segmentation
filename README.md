# Comprehensive Road Scene Understanding for Autonomous Driving

Course project on **semantic / anomaly segmentation for road scenes**. We study the
**EoMT** mask-based segmentation model (DINOv2 ViT backbone) on the Cityscapes and
COCO domains, fine-tune the COCO checkpoint toward Cityscapes, and compare **mask-based
(EoMT)** vs **pixel-based (ERFNet)** post-hoc anomaly detection on standard road-anomaly
benchmarks.

## Project steps → code

| Step | What it does | File |
|------|--------------|------|
| **4** | Compare the two pretrained EoMT checkpoints (Cityscapes vs COCO) on the Cityscapes val set in a common label space; report mIoU + per-class IoU. | `eval.py` |
| **5** | Fine-tune the COCO-trained EoMT toward Cityscapes (head-only and last-block variants). | `finetune.py` |
| **7** | Pixel-based anomaly baselines with **ERFNet** (MSP / MaxLogit / MaxEntropy). | `erfnet.py` |
| **8** | Mask-based anomaly baselines with **EoMT** (MSP / MaxLogit / MaxEntropy / **RbA**) + temperature scaling, across all checkpoints. | `step8.py` |
| — | Shared post-hoc anomaly score functions (used by Steps 7 & 8). | `eomt.py` |

**Metrics** (Steps 7 & 8): AuPRC (↑) and FPR95 (↓), on SMIYC RA-21, SMIYC RO-21,
Fishyscapes Lost&Found, Fishyscapes Static, and Road Anomaly.

## Repository structure

```
.
├── eval.py / finetune.py / erfnet.py / eomt.py   # one script per project step
├── scores.py                                   # shared anomaly score functions
├── requirements.txt
├── results/                                     # per-(checkpoint,dataset,T) JSON metrics + figures
├── report/                                      # LaTeX report (main.tex -> main.pdf, fig/, ...)
└── README.md
```

> The `eomt/` model code is the course-provided EoMT fork. It is large; prefer using it
> as a git submodule or cloning it separately rather than committing it wholesale.

## Setup

```bash
git clone https://github.com/AlessandroMarinai/MaskArchitectureAnomaly_CourseProject.git
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# The EoMT model has its own deps; if anything is missing at runtime:
# pip install -r MaskArchitectureAnomaly_CourseProject/eomt/requirements.txt
```

## Data & checkpoints (not in the repo)

Datasets and model weights are **git-ignored** (large). Obtain them and point the scripts
at their locations via the CLI flags:

- **EoMT checkpoints** (`eomt_cityscapes.bin`, `eomt_coco.bin`, and the Step 5 fine-tuned
  `.bin` files) — professor-provided.
- **Cityscapes val** (`leftImg8bit_trainvaltest.zip`, `gtFine_trainvaltest.zip`) — Step 4.
- **Anomaly validation datasets** (`Validation_Dataset/` with the 5 benchmarks above) — Steps 7 & 8.

## Running

**Eval — checkpoint mIoU comparison:** `python step4.py` (expects the Cityscapes zips and the two `.bin` at the paths set in the script).

**Fine-tune — fine-tuning:** see `step5.py` for its arguments.

**ERFNET — pixel-based anomaly (ERFNet):** `python step7.py --input '<dir>/images/*.jpg' --loadDir <models>/ --method msp` (see script args).

**EoMT — mask-based anomaly (example):**
```bash
python step8.py \
  --kind cityscapes \                 # cityscapes | coco | ft_head | ft_last_block
  --ckpt    <path>/eomt_cityscapes.bin \
  --eomt-root <path>/eomt \
  --data-root <path>/Validation_Dataset \
  --datasets RoadAnomaly RoadAnomaly21 RoadObsticle21 fs_static FS_LostFound_full \
  --temps 0.25 0.5 0.75 1.0 1.5 2.0 2.5 \
  --out results
```
Writes one JSON per `(checkpoint, dataset, temperature)` to `--out` (resumable: existing
files are skipped).

## Results & report

- Numeric results: `results/*.json` (parse with the small loaders in the notebooks /
  `report/generate_assets.py`).
- Final report: `report/main.pdf`.

## Team

Xiangxi Li · Kutay Zilcioglu · Fabio Segalin · Xueyufei Zhang
