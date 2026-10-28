"""ExtraTrees gate followed by a three-class probability-SVC head."""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score
from .features import RunConfig,featurize_fold,random_oversample
from .priors import select_user_aug,build_eval_classifier

CLASS_NAMES = ["near-Zigzag", "Chiral", "near-Armchair"]

def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    recalls = []
    precisions = []
    for idx in range(3):
        row_total = cm[idx].sum()
        col_total = cm[:, idx].sum()
        recalls.append(float(cm[idx, idx] / row_total) if row_total else 0.0)
        precisions.append(float(cm[idx, idx] / col_total) if col_total else 0.0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "correct": int((y_true == y_pred).sum()),
        "n": int(len(y_true)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "zigzag_recall": recalls[0],
        "chiral_recall": recalls[1],
        "armchair_recall": recalls[2],
        "zigzag_precision": precisions[0],
        "chiral_precision": precisions[1],
        "armchair_precision": precisions[2],
        "confusion_matrix": cm.tolist(),
    }


def fit_fold_model(
    cfg: RunConfig,
    mode: str,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    X_user: np.ndarray,
    y_user: np.ndarray,
    user_meta: pd.DataFrame,
    model,
    ab_scaler,
    device: str,
    fold: int,
):
    X_train_real = X[tr_idx]
    y_train_real = y[tr_idx]
    X_add, y_add, _ = select_user_aug(
        mode,
        X_user,
        y_user,
        user_meta,
        X_train_real,
        y_train_real,
        seed=cfg.seed + fold * 997,
    )
    X_aug = np.vstack([X_train_real, X_add]) if len(X_add) else X_train_real
    y_aug = np.concatenate([y_train_real, y_add]) if len(y_add) else y_train_real

    X_feat_train, X_feat_test = featurize_fold(
        model, cfg, X_train_real, X_aug, X[te_idx], ab_scaler, device
    )
    if cfg.params.get("oversample", False):
        X_feat_train, y_aug = random_oversample(
            X_feat_train, y_aug, seed=cfg.seed + fold
        )

    clf = build_eval_classifier(cfg, input_dim=X_feat_train.shape[1], device=device)
    clf.fit(X_feat_train, y_aug)
    return clf, X_feat_test, int(len(y_add)), int((y_add == 0).sum())


def predict_ac_from_probabilities(clf, X_feat_test: np.ndarray) -> int:
    if not hasattr(clf, "predict_proba"):
        pred = int(clf.predict(X_feat_test)[0])
        return pred if pred in (1, 2) else 1
    proba = clf.predict_proba(X_feat_test)[0]
    classes = list(getattr(clf, "classes_", []))
    candidates = [cls for cls in (1, 2) if cls in classes]
    if not candidates:
        pred = int(clf.predict(X_feat_test)[0])
        return pred if pred in (1, 2) else 1
    best = max(candidates, key=lambda cls: proba[classes.index(cls)])
    return int(best)

