# PG-TIN: physics-guided tabular imaging for ball mill power

Code for the paper *How Many Samples Does a Mill Need? Sample-Efficient Power-Draw
Prediction and a Falsifiable Test for Tabular-to-Image Transfer* (Engineering Applications
of Artificial Intelligence, under review).

PG-TIN predicts ball mill power draw from a small number of labelled plant records.
Ridge regression makes the base prediction. Each record is also drawn as a small image
(an IGTD feature layout, Gaussian point-spread rendering and three channels), and a CNN
learns to correct the ridge residuals. The correction is scaled by a weight alpha that is
fitted on out-of-fold predictions only, so when the image branch does not help, alpha is
zero and the model is plain ridge.

## Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

A GPU is used when available; the code also runs on a CPU, only more slowly.

## Data

The plant data cannot be shared publicly because of a confidentiality agreement with the
operating plant. It is available from the corresponding author on reasonable request.
Place it next to the scripts as `Data.xlsx` (sheet `Sheet4`), or pass
`--data path/to/Data.xlsx` to any script.

## Reproducing the paper

Run the scripts in order:

| Script | What it does | Paper |
|---|---|---|
| `01_pgtin.py` | Trains and tests PG-TIN at K = 100, 300 and 500, seeds 123, 42 and 2026, on both feature sets | Tables 2 and 4 |
| `02_ablation_reference_full_model.py` | Ablation reference: the full model | Table 5 |
| `03_ablation_random_layout_instead_of_igtd.py` | Random feature layout instead of IGTD | Table 5 |
| `04_ablation_no_gaussian_splatting.py` | One pixel per feature instead of Gaussian splatting | Table 5 |
| `05_ablation_single_channel_image.py` | Value channel only instead of three channels | Table 5 |
| `06_ablation_no_ridge_anchor.py` | The CNN corrects the training mean instead of ridge | Table 5 |

```bash
python 01_pgtin.py
python 02_ablation_reference_full_model.py     # and likewise 03 to 06
```

Everything is written to `results/`. The ablations use the physics feature set and
10 seeds (123, 42, 2026 and 0-6) at K = 300 and 500, and are scored out of fold on the
training rows, so they never touch the test set.

Each script refuses to overwrite earlier results unless you add `--overwrite`. Run any
script with `--help` to see its options; for a quick check, try
`python 01_pgtin.py --seeds 123 --train-sizes 100 --epochs 20`.

## Reproducibility

Seeds are fixed and cuDNN runs in deterministic mode, so rerunning on the same machine
gives identical numbers. Results can differ slightly across GPUs, drivers and library
versions. Every run saves its settings, package versions and git commit in `config.json`.

## Citation

If you use this code, please cite the paper (details will be added on publication).
