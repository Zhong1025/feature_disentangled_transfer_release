"""Archived candidates used as hard-labelled synthetic training observations.

Encoding does not turn these observations into a soft constraint.
No API calls are made here.
"""
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.svm import SVC

from .features import (
    DATA_DIR,
    FEATURES,
    RunConfig,
    build_classifier,
    get_value,
    nearest_llm_subset,
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_LLM_PATHS = {
    "new1": str(DATA_DIR / "new1.json"),
    "new2": str(DATA_DIR / "new2.json"),
}





def load_user_candidates() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows, labels, meta = [], [], []
    for source, path in USER_LLM_PATHS.items():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            d = item.get("data", item)
            row = []
            ok = True
            for field in FEATURES:
                try:
                    value = get_value(d, field)
                    row.append(float(value))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue
            label = d.get("Chirality Classification")
            if label is None:
                continue
            label = int(label) - 1
            if label not in (0, 1, 2):
                continue
            rows.append(row)
            labels.append(label)
            meta.append(
                {
                    "source": source,
                    "row": i,
                    "label_0based": label,
                    "chiral_index": d.get("Chiral Index", ""),
                    "chiral_angle": d.get("Chiral angle", ""),
                }
            )
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), pd.DataFrame(meta)


def select_user_aug(
    mode: str,
    X_user: np.ndarray,
    y_user: np.ndarray,
    meta: pd.DataFrame,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if mode == "none":
        return np.empty((0, X_train.shape[1])), np.empty((0,), dtype=int), meta.iloc[[]].copy()

    repaired = mode.endswith("_nearest_repaired")
    source_part = mode.replace("_raw", "").replace("_nearest_repaired", "")
    if source_part == "new1_new2":
        source_mask = meta["source"].isin(["new1", "new2"]).values
    else:
        source_mask = (meta["source"] == source_part).values

    Xs = X_user[source_mask]
    ys = y_user[source_mask]
    ms = meta[source_mask].copy()
    if repaired and len(Xs):
        Xr, yr = nearest_llm_subset(Xs, ys, X_train, y_train, seed=seed, per_class_cap=8)
        # nearest_llm_subset does not return row ids; recover selected rows by nearest repaired values.
        # For reporting counts, label/source totals are enough, so preserve a compact synthetic meta table.
        ms = pd.DataFrame({"source": [source_part] * len(yr), "label_0based": yr})
        return Xr, yr, ms
    return Xs, ys, ms


def build_eval_classifier(cfg: RunConfig, input_dim: int, device: str):
    if cfg.classifier == "svc_prob":
        return SVC(
            C=float(cfg.params.get("C", 1.0)),
            gamma=cfg.params.get("gamma", "scale"),
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=cfg.seed,
        )
    return build_classifier(cfg, input_dim=input_dim, device=device)



