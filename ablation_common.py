"""Shared code for the ablation scripts (02-06).

Each ablation changes one step of 01_pgtin.py and is scored out of fold on the
training rows only. The test rows are never touched.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent

# The main script's name starts with a digit, so it cannot be imported the usual way.
_spec = importlib.util.spec_from_file_location("pgtin", PROJECT_ROOT / "01_pgtin.py")
pgtin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgtin)
torch = pgtin.torch

FEATURE_SET = "raw_plus_physics"
OUTPUT_FILES = {"config.json", "oof_metrics.csv"}


def rmse(y, pred):
    return float(np.mean((y - pred) ** 2) ** 0.5)


def oof_scores(X, y, seed, n_folds, epochs, layout, bank_builder, image_builder, use_ridge_anchor):
    """Same steps as fit_predict_pgtin, but every number comes from held-out folds."""
    folds = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=seed)
    ridge_oof = cross_val_predict(RidgeCV(alphas=pgtin.RIDGE_ALPHAS), X, y, cv=folds)
    anchor_oof = ridge_oof if use_ridge_anchor else np.full(len(y), float(np.mean(y)))
    resid = y - anchor_oof
    scale = float(resid.std()) + 1e-9

    side = int(math.ceil(math.sqrt(X.shape[1])))
    bank = bank_builder(layout(X, side, seed=seed), side)

    cnn_oof = np.zeros(len(y))
    for f, (fit, held) in enumerate(folds.split(X)):
        median, saliency = pgtin.channel_stats(X[fit], resid[fit])
        model = pgtin.train_network(image_builder(X[fit], bank, median, saliency), X[fit],
                                    resid[fit] / scale, seed=seed + 100 * f, epochs=epochs)
        cnn_oof[held] = pgtin.predict(model, image_builder(X[held], bank, median, saliency),
                                      X[held], seed=seed + f) * scale
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    denom = float(np.dot(cnn_oof, cnn_oof))
    if denom > 1e-12:
        alpha = float(np.clip(np.dot(resid, cnn_oof) / (denom + 1e-12), 0.0, 1.5))
    else:
        alpha = 0.0
    corr = float(np.corrcoef(resid, cnn_oof)[0, 1]) if cnn_oof.std() > 1e-12 else 0.0

    return {"alpha": alpha,
            "cnn_oof_corr": corr,
            "ridge_oof_rmse": rmse(y, ridge_oof),
            "final_oof_rmse": rmse(y, anchor_oof + alpha * cnn_oof)}


def prepare_output_dir(out_dir, overwrite):
    if out_dir.exists():
        present = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()}
        unknown = sorted(present - OUTPUT_FILES)
        if unknown:
            raise SystemExit(f"{out_dir} contains files this script does not write:\n  "
                             + "\n  ".join(unknown) + "\nMove them away and run again.")
        if present and not overwrite:
            raise SystemExit(f"{out_dir} already holds results. Use --overwrite to replace them.")
    out_dir.mkdir(parents=True, exist_ok=True)


def run(name, description, layout=None, bank_builder=None, image_builder=None,
        use_ridge_anchor=True):
    """Run one ablation. Pass only the step you want to replace; the rest is unchanged."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "Data.xlsx")
    ap.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "ablations")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 42, 2026, 0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[300, 500])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = args.output_root / name
    prepare_output_dir(out_dir, args.overwrite)
    pgtin.use_deterministic_torch()

    config = {"ablation": name, "description": description, "feature_set": FEATURE_SET,
              "data": str(args.data), "seeds": args.seeds, "train_sizes": args.train_sizes,
              "folds": args.folds, "epochs": args.epochs, "device": str(pgtin.DEVICE),
              "git_commit": pgtin.git_commit(), "versions": pgtin.package_versions()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    data = pgtin.load_data(args.data)
    cols = pgtin.FEATURE_SETS[FEATURE_SET]
    rows = []
    for k in args.train_sizes:
        for seed in args.seeds:
            train_idx, _ = pgtin.make_split(len(data), k, seed)   # same split as 01_pgtin.py
            y = data.loc[train_idx, pgtin.TARGET].to_numpy()
            X = StandardScaler().fit_transform(data.loc[train_idx, cols]).astype(np.float32)
            scores = oof_scores(X, y, seed, args.folds, args.epochs,
                                layout or pgtin.igtd_layout,
                                bank_builder or pgtin.gaussian_bank,
                                image_builder or pgtin.to_images,
                                use_ridge_anchor)
            rows.append({"ablation": name, "train_size": k, "seed": seed, **scores})
            # Save after every run, so an interrupted job keeps what it has done.
            pd.DataFrame(rows).to_csv(out_dir / "oof_metrics.csv", index=False)
            print(f"{name}  K={k:<4d} seed={seed:<5d} corr={scores['cnn_oof_corr']:+.3f}  "
                  f"alpha={scores['alpha']:.3f}  oof_rmse={scores['final_oof_rmse']:.5f}  "
                  f"(ridge {scores['ridge_oof_rmse']:.5f})", flush=True)
