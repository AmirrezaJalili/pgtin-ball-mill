"""Train and test PG-TIN, the physics-guided tabular imaging network.

Ridge regression makes the base prediction. Each record is also drawn as a small
image (IGTD layout, Gaussian point-spread rendering, three channels), and a CNN
learns to correct the ridge residuals. The correction is scaled by alpha, which is
fitted on out-of-fold predictions only, so alpha = 0 gives back plain ridge.

Usage:
    python 01_pgtin.py
    python 01_pgtin.py --seeds 123 --train-sizes 100 --epochs 20    # quick check
"""
import argparse
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

# Needed for deterministic cuBLAS; must be set before torch is imported.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RIDGE_ALPHAS = np.logspace(-3, 4, 40)   # penalty grid shared with the ridge baseline
UPSCALE = 4                             # each grid cell becomes a 4 x 4 pixel block
SIGMA = 1.4                             # width of the Gaussian blob, in pixels
N_TTA = 8                               # prediction passes averaged at test time
EPS = 1e-9

# Column names in the spreadsheet -> short names used in the code.
COLUMNS = {
    "Work Index (sampling)": "work_index",
    "Feed Rate (from belt weigher log)": "feed_rate",
    "Ball Charge(from operation data)": "ball_charge",
    "Power (from logsheet)": "power",
    "Feed Size (sampling)": "feed_size",
    "Size of Product (sample from CY)": "product_size",
}
TARGET = "power"
RAW_FEATURES = ["work_index", "feed_rate", "ball_charge", "feed_size", "product_size"]
PHYSICS_FEATURES = ["size_ratio", "log_size_ratio", "bond_term", "bond_energy", "bond_power_mw",
                    "feed_per_ball", "ball_per_feed", "feed_x_ball", "feed_x_bond_energy",
                    "work_x_log_ratio"]
FEATURE_SETS = {"raw": RAW_FEATURES, "raw_plus_physics": RAW_FEATURES + PHYSICS_FEATURES}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(path):
    raw = pd.read_excel(path, sheet_name="Sheet4")
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {missing}")
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return add_physics_features(df)


def add_physics_features(df):
    """Add the ten features derived from Bond's law (sizes in um, Wi in kWh/t, Q in t/h)."""
    df = df.copy()
    F = df["feed_size"].clip(lower=EPS)
    P = df["product_size"].clip(lower=EPS)
    Q = df["feed_rate"]
    B = df["ball_charge"].clip(lower=EPS)
    Wi = df["work_index"]
    df["size_ratio"] = F / P
    df["log_size_ratio"] = np.log(F / P)
    df["bond_term"] = 1 / np.sqrt(P) - 1 / np.sqrt(F)
    df["bond_energy"] = 10 * Wi * df["bond_term"]          # kWh/t
    df["bond_power_mw"] = Q * df["bond_energy"] / 1000.0   # MW
    df["feed_per_ball"] = Q / B
    df["ball_per_feed"] = B / Q.clip(lower=EPS)
    df["feed_x_ball"] = Q * B
    df["feed_x_bond_energy"] = Q * df["bond_energy"]
    df["work_x_log_ratio"] = Wi * df["log_size_ratio"]
    return df


def make_split(n, k, seed):
    """Draw k training rows at random; every other row is a test row."""
    rng = np.random.default_rng(seed)
    train_idx = np.sort(rng.choice(n, size=k, replace=False))
    is_test = np.ones(n, dtype=bool)
    is_test[train_idx] = False
    return train_idx, np.flatnonzero(is_test)


def evaluate(y_true, y_pred):
    return {"RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
            "MAE": mean_absolute_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Turning a record into an image
# ---------------------------------------------------------------------------

def _rank_matrix(D):
    upper = np.triu_indices(D.shape[0], 1)
    R = np.zeros_like(D, dtype=float)
    R[upper] = np.argsort(np.argsort(D[upper]))
    return R + R.T


def igtd_layout(X, side, seed, max_sweeps=60):
    """IGTD (Zhu et al., 2021): put correlated features in neighbouring grid cells.

    Feature distance is 1 - |corr|. We swap pairs of cells until the ranking of feature
    distances matches the ranking of cell distances as closely as possible.
    Returns one (row, col) cell per feature.
    """
    n = X.shape[1]
    feat_dist = 1.0 - np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    np.fill_diagonal(feat_dist, 0.0)
    R = _rank_matrix(feat_dist)

    cells = np.array([(r, c) for r in range(side) for c in range(side)], dtype=float)
    Q = _rank_matrix(np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1)))

    perm = np.arange(len(cells))
    np.random.default_rng(seed).shuffle(perm)

    def error(p):
        idx = p[:n]
        return float(np.abs(R - Q[np.ix_(idx, idx)]).sum())

    best = error(perm)
    for _ in range(max_sweeps):
        improved = False
        for a in range(n):
            for b in range(a + 1, len(cells)):
                cand = perm.copy()
                cand[a], cand[b] = cand[b], cand[a]
                e = error(cand)
                if e < best - 1e-9:
                    perm, best, improved = cand, e, True
        if not improved:
            break
    return [(int(r), int(c)) for r, c in cells[perm[:n]].astype(int)]


def gaussian_bank(positions, side):
    """One Gaussian blob per feature, centred on its cell of the upscaled canvas."""
    size = side * UPSCALE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    bank = np.zeros((len(positions), size, size), dtype=np.float32)
    for j, (r, c) in enumerate(positions):
        cy, cx = (r + 0.5) * UPSCALE, (c + 0.5) * UPSCALE
        bank[j] = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * SIGMA ** 2))
    return bank


def _splat(X, bank):
    # Each image is the sum of the feature blobs, each weighted by its value.
    return np.tensordot(X.astype(np.float32), bank, axes=([1], [0]))


def to_images(X, bank, median, saliency):
    """Three channels: the values, |value - median| and values weighted by saliency."""
    channels = [_splat(X, bank),
                _splat(np.abs(X - median[None, :]), bank),
                _splat(X * saliency[None, :], bank)]
    return np.stack(channels, axis=1).astype(np.float32)


def channel_stats(X, resid):
    """Median of each feature, and its |corr| with the ridge residual scaled to max 1.

    Always computed on the rows the network is trained on, never on held-out rows.
    """
    median = np.median(X, axis=0)
    saliency = np.zeros(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        if X[:, j].std() > 1e-12 and resid.std() > 1e-12:
            saliency[j] = abs(np.corrcoef(X[:, j], resid)[0, 1])
    saliency = np.nan_to_num(saliency)
    return median, saliency / (saliency.max() + 1e-9)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class CoordConv(nn.Module):
    """Append y and x coordinate channels so the convolutions know where they are."""

    def forward(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, ys, xs], dim=1)


class PGTIN(nn.Module):
    """CNN branch on the image plus a small MLP on the feature vector, fused in one head."""

    def __init__(self, n_features, in_channels=3, width=32, dropout=0.25):
        super().__init__()
        self.coord = CoordConv()
        self.vision = nn.Sequential(
            nn.Conv2d(in_channels + 2, width, 3, padding=1), nn.BatchNorm2d(width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.GELU(),
            nn.Conv2d(width * 2, width * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(width * 2), nn.GELU(),
        )
        self.tabular = nn.Sequential(
            nn.Linear(n_features, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 32), nn.GELU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4 + 32, 64), nn.GELU(),
                                  nn.Linear(64, 1))
        self.gate = nn.Parameter(torch.zeros(1))
        # The last layer starts at zero, so training starts from "no correction" (plain ridge).
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, img, tab):
        f = self.vision(self.coord(img))
        # Average and max pooling, side by side, plus the MLP features.
        z = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3)), self.tabular(tab)], dim=1)
        return self.head(z).squeeze(1) * torch.exp(self.gate)


class EMA:
    """Exponential moving average of the weights; the average is what we keep."""

    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def copy_to(self, model):
        target = model.state_dict()
        model.load_state_dict({k: self.shadow[k].to(dtype=target[k].dtype) for k in target})


def train_network(images, tab, target, seed, epochs, lr=3e-3, weight_decay=1e-2,
                  mixup_alpha=0.2, noise=0.05):
    """Train for a fixed number of epochs (no early stopping, so no rows are held back)."""
    seed_everything(seed)
    model = PGTIN(tab.shape[1], in_channels=images.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(tab)
    batch_size = int(min(32, max(8, n)))
    # Drop a last batch of size 1, since BatchNorm cannot train on a single row.
    loader = DataLoader(TensorDataset(torch.tensor(images),
                                      torch.tensor(tab, dtype=torch.float32),
                                      torch.tensor(target, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True,
                        drop_last=(n % batch_size == 1) and (n > batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(1, epochs * max(1, len(loader))), pct_start=0.2)
    loss_fn = nn.HuberLoss(delta=1.0)
    ema = EMA(model)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        model.train()
        for xb, tb, yb in loader:
            xb, tb, yb = xb.to(DEVICE), tb.to(DEVICE), yb.to(DEVICE)
            if xb.shape[0] < 2:
                continue
            # Small input noise, then mixup on images, features and targets together.
            xb = xb + noise * torch.randn_like(xb)
            tb = tb + noise * torch.randn_like(tb)
            if xb.shape[0] > 2:
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                p = torch.randperm(xb.shape[0], device=xb.device)
                xb = lam * xb + (1 - lam) * xb[p]
                tb = lam * tb + (1 - lam) * tb[p]
                yb = lam * yb + (1 - lam) * yb[p]
            optimizer.zero_grad()
            loss = loss_fn(model(xb, tb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ema.update(model)

    ema.copy_to(model)
    model.eval()
    return model


@torch.no_grad()
def predict(model, images, tab, seed, jitter=0.03):
    """Average N_TTA passes: the clean input plus slightly jittered copies."""
    xi = torch.tensor(images, dtype=torch.float32, device=DEVICE)
    ti = torch.tensor(tab, dtype=torch.float32, device=DEVICE)
    preds = [model(xi, ti).cpu().numpy()]
    g = torch.Generator().manual_seed(int(seed))
    for _ in range(N_TTA - 1):
        dx = (torch.randn(xi.shape, generator=g) * jitter).to(DEVICE)
        dt = (torch.randn(ti.shape, generator=g) * jitter).to(DEVICE)
        preds.append(model(xi + dx, ti + dt).cpu().numpy())
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------------------
# PG-TIN
# ---------------------------------------------------------------------------

def fit_predict_pgtin(X_train, y_train, X_test, seed, n_folds, epochs):
    folds = KFold(n_splits=min(n_folds, len(y_train)), shuffle=True, random_state=seed)

    # Anchor: ridge fitted on all training rows. The CNN learns its out-of-fold
    # residuals, because in-sample residuals would be too small.
    ridge_oof = cross_val_predict(RidgeCV(alphas=RIDGE_ALPHAS), X_train, y_train, cv=folds)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(X_train, y_train)
    ridge_test = ridge.predict(X_test)
    resid = y_train - ridge_oof
    scale = float(resid.std()) + 1e-9   # the network trains on residuals in units of their std

    # The layout is fitted once on the training rows and then kept fixed.
    side = int(math.ceil(math.sqrt(X_train.shape[1])))
    bank = gaussian_bank(igtd_layout(X_train, side, seed=seed), side)

    # Five fold models: each predicts its held-out fold (for alpha) and the test rows.
    cnn_oof = np.zeros(len(y_train))
    cnn_test = []
    for f, (fit, held) in enumerate(folds.split(X_train)):
        median, saliency = channel_stats(X_train[fit], resid[fit])
        model = train_network(to_images(X_train[fit], bank, median, saliency), X_train[fit],
                              resid[fit] / scale, seed=seed + 100 * f, epochs=epochs)
        cnn_oof[held] = predict(model, to_images(X_train[held], bank, median, saliency),
                                X_train[held], seed=seed + f) * scale
        cnn_test.append(predict(model, to_images(X_test, bank, median, saliency),
                                X_test, seed=seed + f) * scale)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # alpha is the least-squares weight of the CNN on the residual, clipped to [0, 1.5].
    # It only ever sees out-of-fold predictions; alpha = 0 means plain ridge.
    denom = float(np.dot(cnn_oof, cnn_oof))
    if denom > 1e-12:
        alpha = float(np.clip(np.dot(resid, cnn_oof) / (denom + 1e-12), 0.0, 1.5))
    else:
        alpha = 0.0
    oof_corr = float(np.corrcoef(resid, cnn_oof)[0, 1]) if cnn_oof.std() > 0 else 0.0

    pred = ridge_test + alpha * np.mean(cnn_test, axis=0)
    return pred, {"alpha": alpha, "oof_corr": oof_corr, "ridge_penalty": float(ridge.alpha_),
                  "ridge_test_pred": ridge_test}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_FILES = ["config.json", "run_metrics.csv", "summary.csv", "diagnostics.csv",
                "predictions.csv"]


def check_output_dir(out_dir, overwrite):
    """Refuse to mix results with files from somewhere else, or to overwrite silently."""
    if not out_dir.exists():
        return
    present = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()}
    unknown = sorted(present - set(OUTPUT_FILES))
    if unknown:
        raise SystemExit(f"{out_dir} contains files this script does not write:\n  "
                         + "\n  ".join(unknown) + "\nMove them away and run again.")
    if present and not overwrite:
        raise SystemExit(f"{out_dir} already holds results ({', '.join(sorted(present))}). "
                         "Use --overwrite to replace them.")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "not a git repository"


def package_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}


def use_deterministic_torch():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=PROJECT_ROOT / "Data.xlsx")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "pgtin")
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 42, 2026])
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--feature-sets", nargs="+", choices=list(FEATURE_SETS),
                    default=list(FEATURE_SETS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    check_output_dir(args.output_dir, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_deterministic_torch()

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "ridge_alphas": RIDGE_ALPHAS.tolist(), "upscale": UPSCALE, "sigma": SIGMA,
              "n_tta": N_TTA, "device": str(DEVICE), "git_commit": git_commit(),
              "versions": package_versions()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    data = load_data(args.data)
    print(f"{len(data)} rows | device {DEVICE}")

    runs, diagnostics, predictions = [], [], []
    for k in args.train_sizes:
        for seed in args.seeds:
            train_idx, test_idx = make_split(len(data), k, seed)
            y_train = data.loc[train_idx, TARGET].to_numpy()
            y_test = data.loc[test_idx, TARGET].to_numpy()

            for fs in args.feature_sets:
                cols = FEATURE_SETS[fs]
                scaler = StandardScaler().fit(data.loc[train_idx, cols])
                X_train = scaler.transform(data.loc[train_idx, cols]).astype(np.float32)
                X_test = scaler.transform(data.loc[test_idx, cols]).astype(np.float32)

                pred, info = fit_predict_pgtin(X_train, y_train, X_test, seed,
                                               args.folds, args.epochs)
                scores = evaluate(y_test, pred)
                ridge_rmse = evaluate(y_test, info["ridge_test_pred"])["RMSE"]

                tag = {"train_size": k, "seed": seed, "feature_set": fs}
                runs.append({**tag, **scores})
                diagnostics.append({**tag, "alpha": info["alpha"], "oof_corr": info["oof_corr"],
                                    "ridge_penalty": info["ridge_penalty"],
                                    "ridge_only_RMSE": ridge_rmse})
                predictions.append(pd.DataFrame({**tag, "row_index": test_idx,
                                                 "y_true": y_test, "y_pred": pred}))
                print(f"K={k:<4d} seed={seed:<5d} {fs:<17s} RMSE={scores['RMSE']:.5f}  "
                      f"(ridge alone {ridge_rmse:.5f}, alpha={info['alpha']:.3f})")

    runs = pd.DataFrame(runs)
    summary = (runs.groupby(["train_size", "feature_set"])
               .agg(runs=("RMSE", "size"), RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
                    MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
                    R2_mean=("R2", "mean"), R2_std=("R2", "std"))
               .reset_index())

    runs.to_csv(args.output_dir / "run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output_dir / "diagnostics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "predictions.csv",
                                                     index=False)

    print("\nPG-TIN test RMSE (mean ± std over seeds)")
    for _, r in summary.iterrows():
        print(f"  K={r.train_size:<4d} {r.feature_set:<17s} {r.RMSE_mean:.5f} ± {r.RMSE_std:.5f}"
              f"   R2 {r.R2_mean:.4f}")
    print(f"\nResults written to {args.output_dir}")


if __name__ == "__main__":
    main()
