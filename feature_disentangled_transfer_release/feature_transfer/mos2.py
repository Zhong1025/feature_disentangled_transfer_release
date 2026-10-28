"""MoS2 descriptor mapping and transferred-feature construction."""
from __future__ import annotations
import math, re
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from .features import encode_shared_features

CLASS_NAMES = ["monolayer", "bilayer", "multilayer"]


CLASS_LABELS = [0, 1, 2]


CNT_FEATURES = [
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


MOS_PROCESS_NUMERIC = [
    "growth_temperature_C",
    "mo_source_temperature_C",
    "s_source_temperature_C",
    "substrate_temperature_C",
    "total_flow_sccm",
    "ar_flow_sccm",
    "n2_flow_sccm",
    "h2_flow_sccm",
    "h2s_flow_sccm",
    "o2_flow_sccm",
    "growth_pressure_Torr",
    "holding_time_min",
    "ramp_rate_C_min",
    "mo_precursor_mass_mg",
    "mo_precursor_thickness_nm",
    "s_precursor_mass_mg",
    "precursor_ratio",
]


MOS_ADV_NUMERIC = ["promoter_used", "oxygen_used", "nacl_mass_mg_num"]


MOS_CATEGORICAL = [
    "growth_method",
    "mo_precursor",
    "s_precursor",
    "substrate_class",
    "pressure_type",
    "promoter_group",
]


MOS_DIRECT_FEATURES = MOS_PROCESS_NUMERIC + MOS_ADV_NUMERIC + MOS_CATEGORICAL


MOS_ADV_FEATURES = MOS_ADV_NUMERIC + ["substrate_class", "pressure_type", "promoter_group"]


FORBIDDEN_MOS_FEATURE_TERMS = ("area", "domain", "edge", "coverage", "film", "size")


@dataclass(frozen=True)
class FeatureSet:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    feature_names: list[str]
    group_names: list[str]


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def as_dense(x: object) -> np.ndarray:
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def parse_first_float(value: object) -> float:
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(nums[0]) if nums else np.nan


def promoter_group(value: object) -> str:
    text = str(value or "").lower()
    if not text or text == "nan" or "none" in text:
        return "none"
    if any(token in text for token in ["nacl", "na ", "sodium", "alkali", "kcl", "koh"]):
        return "salt/alkali"
    if "oxygen" in text or "o2" in text:
        return "oxygen"
    if "ptas" in text or "seed" in text:
        return "seed"
    if "glass" in text or "molten" in text:
        return "glass"
    if any(token in text for token in ["barrier", "foil", "confined", "face-to-face"]):
        return "barrier/geometry"
    return "other"


def additive_presence(value: object) -> float:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return np.nan
    if any(marker in text for marker in ["none", "without", "no additive", "not reported"]):
        return 0.0
    return 1.0


def precursor_property(value: object, prop: str) -> float:
    """Small no-area chemistry lookup used only for mapping to CNT descriptors."""
    text = str(value or "").lower().replace(" ", "")
    table = {
        "moo3": {
            "delta_g": -667.94,
            "delta_h": -745.10,
            "boiling": 1155.0,
            "chemical_potential": -667.94,
        },
        "mocl5": {
            "delta_g": -540.0,
            "delta_h": -600.0,
            "boiling": 268.0,
            "chemical_potential": -540.0,
        },
        "mo(co)6": {
            "delta_g": -940.0,
            "delta_h": -996.0,
            "boiling": 156.0,
            "chemical_potential": -940.0,
        },
        "moo2": {
            "delta_g": -533.0,
            "delta_h": -588.0,
            "boiling": np.nan,
            "chemical_potential": -533.0,
        },
    }
    for key, vals in table.items():
        if key in text:
            return vals[prop]
    return np.nan


def substrate_thermal_conductivity(value: object) -> float:
    text = str(value or "").lower()
    if "silicon nitride" in text or re.search(r"(?<![a-z0-9])si3n4(?![a-z0-9])", text):
        return np.nan
    if "glass" in text and "sapphire" in text:
        return np.nan
    lookup = [
        ("diamond", 2000.0),
        ("graphene", 500.0),
        ("hbn", 400.0),
        ("bn", 400.0),
        ("silicon carbide", 120.0),
        ("sic", 120.0),
        ("sapphire", 35.0),
        ("al2o3", 35.0),
        ("alumina", 35.0),
        ("quartz", 1.4),
        ("silicon dioxide", 1.4),
        ("silicon oxide", 1.4),
        ("sio2", 1.4),
        ("aluminosilicate glass", 1.0),
        ("soda-lime-silica glass", 1.0),
        ("silica glass", 1.4),
        ("glass", 1.0),
        ("silica", 1.4),
        ("oxide", 1.4),
        ("mica", 0.7),
    ]
    for key, val in lookup:
        if key in text:
            return val
    if re.search(r"(?<![a-z0-9])silicon(?![a-z0-9])", text) or re.search(r"(?<![a-z0-9])si(?![a-z0-9])", text):
        return 149.0
    return np.nan


def substrate_lattice_constant(value: object) -> float:
    text = str(value or "").lower()
    if "silicon nitride" in text or re.search(r"(?<![a-z0-9])si3n4(?![a-z0-9])", text):
        return np.nan
    if "sapphire" in text or "al2o3" in text:
        return 4.758
    if "sio2" in text or "silicon dioxide" in text or "silicon oxide" in text:
        return 4.91
    if "quartz" in text:
        return 4.91
    if "glass" in text:
        return np.nan
    lookup = [
        ("sic", 3.08),
        ("silicon carbide", 3.08),
        ("hbn", 2.50),
        ("bn", 2.50),
        ("mica", 5.20),
        ("silica", 4.91),
    ]
    for key, val in lookup:
        if key in text:
            return val
    if re.search(r"(?<![a-z0-9])silicon(?![a-z0-9])", text) or re.search(r"(?<![a-z0-9])si(?![a-z0-9])", text):
        return 5.431
    return np.nan


def assert_no_mos_area_features(feature_names: Iterable[str]) -> None:
    bad = [
        name
        for name in feature_names
        if any(term in name.lower() for term in FORBIDDEN_MOS_FEATURE_TERMS)
    ]
    if bad:
        raise ValueError(f"MoS2 area/domain-related features are forbidden here: {bad}")


def normalize_mos_layer_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "paper_use_status" in df.columns:
        df = df[df["paper_use_status"].astype(str).ne("reference_support")].copy()
    if "nacl_mass_mg" not in df.columns:
        df["nacl_mass_mg"] = np.nan
    df["nacl_mass_mg_num"] = df["nacl_mass_mg"].map(parse_first_float)
    for col in MOS_PROCESS_NUMERIC + MOS_ADV_NUMERIC + ["layer_class_3"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in MOS_CATEGORICAL:
        if col not in df.columns:
            df[col] = ""
    if "promoter_type" not in df.columns:
        df["promoter_type"] = ""
    df["promoter_group"] = df["promoter_type"].map(promoter_group)
    df["substrate_class"] = df["substrate_class"].fillna("other").replace("", "other")
    df["pressure_type"] = df["pressure_type"].fillna("APCVD").replace("", "APCVD")
    df["growth_method"] = df["growth_method"].fillna("CVD").replace("", "CVD")
    df["doi"] = df.get("doi", pd.Series(["unknown"] * len(df))).fillna("unknown")
    df["row_id"] = df.get("row_id", pd.Series([f"row_{i}" for i in range(len(df))])).fillna("")
    df = df.dropna(subset=["layer_class_3"]).copy()
    df["layer_class_3"] = df["layer_class_3"].astype(int)
    df = df[df["layer_class_3"].isin(CLASS_LABELS)].copy()
    assert_no_mos_area_features(MOS_DIRECT_FEATURES + CNT_FEATURES)
    return df.reset_index(drop=True)


def total_flow(df: pd.DataFrame) -> pd.Series:
    total = pd.to_numeric(df["total_flow_sccm"], errors="coerce")
    flow_cols = ["ar_flow_sccm", "n2_flow_sccm", "h2_flow_sccm", "h2s_flow_sccm", "o2_flow_sccm"]
    fallback = df[flow_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    return total.where(total.notna(), fallback)


def map_mos_to_cnt15(df: pd.DataFrame) -> pd.DataFrame:
    substrate_text = df.get("substrate", pd.Series([""] * len(df), index=df.index))
    mo_precursor = df.get("mo_precursor", pd.Series([""] * len(df), index=df.index))
    s_source = pd.to_numeric(df.get("s_source_temperature_C", np.nan), errors="coerce")
    h2s = pd.to_numeric(df.get("h2s_flow_sccm", np.nan), errors="coerce")
    # Gas H2S is the closest direct analogue for CNT carbon feed rate; when it
    # is absent, source temperature gives a weak vapor-supply proxy.
    reactive_feed = h2s.where(h2s.notna(), s_source / 100.0)
    out = pd.DataFrame(
        {
            "Process Temp(oC)": df["growth_temperature_C"],
            "Total flow (sccm)": total_flow(df),
            "H2 flow (sccm)": df["h2_flow_sccm"],
            "C-flow (sccm)": reactive_feed,
            "H/C": df["precursor_ratio"],
            "Carbon_Gibbs": mo_precursor.map(lambda x: precursor_property(x, "delta_g")),
            "Carbon_Enthalpy": mo_precursor.map(lambda x: precursor_property(x, "delta_h")),
            "Cat_MeltingPoint": mo_precursor.map(lambda x: precursor_property(x, "boiling")),
            "Predicted Formation Energy": mo_precursor.map(lambda x: precursor_property(x, "chemical_potential")),
            "Cat_LatticeConstant": df["precursor_ratio"],
            "Etch_level(ppm)": df["promoter_type"].map(additive_presence).where(
                df["promoter_type"].notna(), df["promoter_used"]
            ),
            "Thermal Conductivity": substrate_text.map(substrate_thermal_conductivity),
            "Support_latticeConstant": substrate_text.map(substrate_lattice_constant),
            "Support_AnnealingTemp": df["substrate_temperature_C"],
            "Support_AnnealingTime": df["holding_time_min"] / 60.0,
        },
        index=df.index,
    )
    return out[CNT_FEATURES].apply(pd.to_numeric, errors="coerce")


def fill_with_medians(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    med = train.median(numeric_only=True).reindex(train.columns).fillna(0.0)
    return train.fillna(med).to_numpy(dtype=float), test.fillna(med).to_numpy(dtype=float)


def fill_with_reference(df: pd.DataFrame, reference: pd.Series) -> np.ndarray:
    fill = reference.reindex(df.columns).fillna(0.0)
    return df.fillna(fill).to_numpy(dtype=float)


def table_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    cat_cols: list[str],
    numeric_group: str,
    cat_group: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )
    X_train = as_dense(pre.fit_transform(train_df[numeric_cols + cat_cols]))
    X_test = as_dense(pre.transform(test_df[numeric_cols + cat_cols]))
    names: list[str] = []
    groups: list[str] = []
    if numeric_cols:
        names.extend(numeric_cols)
        groups.extend([numeric_group] * len(numeric_cols))
    if cat_cols:
        enc = pre.named_transformers_["cat"].named_steps["onehot"]
        try:
            cat_names = enc.get_feature_names_out(cat_cols).tolist()
        except Exception:
            cat_names = [f"cat_{i}" for i in range(X_train.shape[1] - len(numeric_cols))]
        names.extend(cat_names)
        groups.extend([cat_group] * len(cat_names))
    return X_train, X_test, names, groups


def proba_with_all_classes(clf, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    out = np.zeros((len(X), len(CLASS_LABELS)), dtype=float)
    for col_idx, cls in enumerate(clf.classes_):
        out[:, int(cls)] = proba[:, col_idx]
    return out


def make_random_splits(df: pd.DataFrame, n_splits: int = 30, test_size: float = 0.25) -> list[tuple[np.ndarray, np.ndarray]]:
    y = df["layer_class_3"].to_numpy(dtype=int)
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=13)
    return list(splitter.split(df, y))


def build_transfer_features(train, test, model, source_scaler, source_medians, device):
    mapped_tr = map_mos_to_cnt15(train)
    mapped_te = map_mos_to_cnt15(test)
    raw_tr, raw_te = fill_with_medians(mapped_tr, mapped_te)
    scaler = StandardScaler().fit(raw_tr)
    mx, mz = scaler.transform(raw_tr), scaler.transform(raw_te)
    sx = encode_shared_features(model, source_scaler.transform(fill_with_reference(mapped_tr, source_medians)), device)
    sz = encode_shared_features(model, source_scaler.transform(fill_with_reference(mapped_te, source_medians)), device)
    ax, az, _, _ = table_transform(train, test, MOS_ADV_NUMERIC,
        ['substrate_class','pressure_type','promoter_group'], 'MoS advanced','MoS advanced')
    return {
        'mapped_only_classifier': (mx, mz),
        'mapped_advanced': (np.hstack([mx, ax]), np.hstack([mz, az])),
        'mapped_shared_advanced': (np.hstack([mx, sx, ax]), np.hstack([mz, sz, az]))}
