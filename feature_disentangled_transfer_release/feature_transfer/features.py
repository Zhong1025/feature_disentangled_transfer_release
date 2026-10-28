"""CNT input contract and raw/shared feature construction.

The frozen encoder input scaler is fitted on source A only.
Target raw-feature scalers are fitted only on real training-fold rows.
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler


BASE_DIR = str(Path(__file__).resolve().parent)
RELEASE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = RELEASE_ROOT / 'data/cnt'
SHARED_ENCODER_PATH = RELEASE_ROOT / 'weights/source_encoder.pth'

FEATURES = [
    "Process Temp(oC)",
    "Total flow (sccm)",
    "H2 flow (sccm)",
    "C-flow (sccm)",
    "H/C",
    "Carbon_Gibbs",
    "Carbon_Enthalpy",
    "Cat_MeltingPoint",
    "Predicted Formation Energy",
    "Cat_LatticeConstant",
    "Etch_level(ppm)",
    "Thermal Conductivity",
    "Support_latticeConstant",
    "Support_AnnealingTemp",
    "Support_AnnealingTime",
]

FIELD_ALIASES = {
    "Process Temp": "Process Temp(oC)",
    "Total flow": "Total flow (sccm)",
    "H2 flow": "H2 flow (sccm)",
    "C-flow": "C-flow (sccm)",
    "Etch_level": "Etch_level(ppm)",
}


@dataclass
class RunConfig:
    name: str
    feature_mode: str
    scaler_mode: str
    augmentation: str
    classifier: str
    seed: int
    params: Dict


def load_shared_encoder(device: str):
    from .networks import SharedSpecificRepresentation

    model = SharedSpecificRepresentation(d_in=15, hidden_dims=[64, 32])
    state = torch.load(SHARED_ENCODER_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def encode_shared_features(model, X_scaled: np.ndarray, device: str) -> np.ndarray:
    if len(X_scaled) == 0:
        return np.empty((0, 15), dtype=float)
    with torch.no_grad():
        x = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        return model.domain_shared(x).detach().cpu().numpy()


def load_real_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(os.path.join(DATA_DIR, "CNT chirality_0903.csv"))
    X = df[FEATURES].astype(float).values
    y = df["Chirality Classification"].astype(int).values - 1
    return df, X, y


def get_value(d: Dict, field: str):
    if field in d:
        return d[field]
    for alias, canonical in FIELD_ALIASES.items():
        if canonical == field and alias in d:
            return d[alias]
    return None


def repair_to_train_distribution(X: np.ndarray, train_X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return X.copy()
    out = X.copy().astype(float)
    lo = np.quantile(train_X, 0.02, axis=0)
    hi = np.quantile(train_X, 0.98, axis=0)
    out = np.minimum(np.maximum(out, lo), hi)
    out[:, 2] = np.maximum(out[:, 2], 0.0)
    out[:, 3] = np.maximum(out[:, 3], 1e-6)
    out[:, 1] = np.maximum(out[:, 1], out[:, 2] + out[:, 3])
    out[:, 4] = out[:, 2] / out[:, 3]
    return out


def nearest_llm_subset(
    X_llm: np.ndarray,
    y_llm: np.ndarray,
    train_X: np.ndarray,
    train_y: np.ndarray,
    seed: int,
    per_class_cap: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(X_llm) == 0:
        return X_llm, y_llm
    X_rep = repair_to_train_distribution(X_llm, train_X)
    scaler = StandardScaler().fit(train_X)
    Xz = scaler.transform(X_rep)
    train_z = scaler.transform(train_X)
    keep = []
    rng = np.random.default_rng(seed)
    for cls in (0, 1, 2):
        idx = np.where(y_llm == cls)[0]
        ref = train_z[train_y == cls]
        if len(idx) == 0 or len(ref) == 0:
            continue
        dists = []
        for j in idx:
            d = np.linalg.norm(ref - Xz[j], axis=1).min()
            dists.append((d, rng.random() * 1e-6, j))
        dists.sort()
        keep.extend([j for _, _, j in dists[:per_class_cap]])
    keep = np.asarray(sorted(keep), dtype=int)
    return X_rep[keep], y_llm[keep]


def random_oversample(X: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=3)
    target = counts.max()
    xs, ys = [X], [y]
    for cls, count in enumerate(counts):
        if count == 0 or count >= target:
            continue
        idx = np.where(y == cls)[0]
        add = rng.choice(idx, size=target - count, replace=True)
        xs.append(X[add])
        ys.append(y[add])
    return np.vstack(xs), np.concatenate(ys)


def get_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    return StandardScaler()


def fit_source_scaler() -> StandardScaler:
    df = pd.read_csv(os.path.join(DATA_DIR, "CNT forest_0903.csv"))
    return StandardScaler().fit(df[FEATURES].astype(float).values)


def featurize_fold(
    model,
    cfg: RunConfig,
    X_train_real: np.ndarray,
    X_train_aug: np.ndarray,
    X_test: np.ndarray,
    ab_scaler: StandardScaler,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    fold_scaler = get_scaler(cfg.scaler_mode).fit(X_train_real)
    raw_train = fold_scaler.transform(X_train_aug)
    raw_test = fold_scaler.transform(X_test)
    if cfg.feature_mode == "raw":
        return raw_train, raw_test

    if cfg.feature_mode in ("shared_ab", "combo_ab", "wide_ab"):
        shared_train = encode_shared_features(model, ab_scaler.transform(X_train_aug), device)
        shared_test = encode_shared_features(model, ab_scaler.transform(X_test), device)
    else:
        shared_train = encode_shared_features(model, raw_train, device)
        shared_test = encode_shared_features(model, raw_test, device)

    if cfg.feature_mode in ("shared", "shared_ab"):
        return shared_train, shared_test
    if cfg.feature_mode in ("combo", "combo_ab"):
        return np.hstack([raw_train, shared_train]), np.hstack([raw_test, shared_test])
    if cfg.feature_mode in ("wide", "wide_ab"):
        return (
            np.hstack([raw_train, shared_train, raw_train * shared_train]),
            np.hstack([raw_test, shared_test, raw_test * shared_test]),
        )
    raise ValueError(f"Unknown feature mode: {cfg.feature_mode}")


def build_classifier(cfg: RunConfig, input_dim: int, device: str):
    """Fixed historical ExtraTrees gate, preserving the archived parameters."""
    if cfg.classifier != 'et':
        raise ValueError('Use the archived ExtraTrees gate or probability SVC head')
    p = cfg.params
    return ExtraTreesClassifier(n_estimators=int(p.get('n_estimators', 160)),
        max_depth=p.get('max_depth', 3), min_samples_leaf=int(p.get('min_samples_leaf', 1)),
        class_weight='balanced', random_state=cfg.seed, n_jobs=1)
