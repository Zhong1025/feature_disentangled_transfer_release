"""Generate or check independent synthetic input-format examples (standard library).

Every number and label is artificial and for demonstration only. The literal
ranges below are illustrative choices, not measurements, fitted statistics,
validated synthesis recipes, or predictions. Nothing is read from private data,
checkpoints, existing model code, or published representative examples.

The seed is fixed at 42. Labels cycle through the documented class codes and
have no experimental meaning. This command does not train or evaluate a model,
and does not make the unchanged historical model scripts use these examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
ORIGIN = "synthetic / demonstration only"
CNT_LABEL = "Chirality Classification"
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
CNT_FIELDS = ["sample_id", "data_origin", *CNT_FEATURES]
MOS_NUMERIC = [
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
    "promoter_used",
    "oxygen_used",
    "nacl_mass_mg",
]
MOS_CATEGORICAL = [
    "growth_method",
    "mo_precursor",
    "s_precursor",
    "substrate",
    "substrate_class",
    "pressure_type",
    "promoter_type",
]
MOS_FIELDS = ["row_id", "data_origin", *MOS_NUMERIC, *MOS_CATEGORICAL, "layer_class_3"]
MOS_FLOW_FIELDS = ["ar_flow_sccm", "n2_flow_sccm", "h2_flow_sccm", "h2s_flow_sccm", "o2_flow_sccm"]
# Zero-based row positions. Empty CSV cells mean unavailable, never zero.
MOS_MISSING = {
    (0, "h2s_flow_sccm"),
    (1, "total_flow_sccm"),
    (4, "nacl_mass_mg"),
    (5, "total_flow_sccm"),
    (6, "h2s_flow_sccm"),
    (8, "mo_source_temperature_C"),
}


def draw(rng: random.Random, low: float, high: float, digits: int = 3) -> float:
    """Use only the explicit illustrative range and the seeded PRNG."""
    return round(low + (high - low) * rng.random(), digits)


def cnt_values(rng: random.Random) -> dict:
    hydrogen = draw(rng, 10, 50)
    carbon = draw(rng, 2, 8)
    return {
        "Process Temp(oC)": draw(rng, 600, 900),
        "Total flow (sccm)": round(hydrogen + carbon + draw(rng, 20, 80), 3),
        "H2 flow (sccm)": hydrogen,
        "C-flow (sccm)": carbon,
        "H/C": round(hydrogen / carbon, 9),
        "Carbon_Gibbs": draw(rng, -120, 80),
        "Carbon_Enthalpy": draw(rng, -150, 100),
        "Cat_MeltingPoint": draw(rng, 1000, 1600),
        "Predicted Formation Energy": draw(rng, -5, -1),
        "Cat_LatticeConstant": draw(rng, 2, 5),
        "Etch_level(ppm)": draw(rng, 0, 50),
        "Thermal Conductivity": draw(rng, 1, 100),
        "Support_latticeConstant": draw(rng, 2, 6),
        "Support_AnnealingTemp": draw(rng, 400, 700),
        "Support_AnnealingTime": draw(rng, 0.5, 2),
    }


def csv_bytes(fields: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def expected_files() -> dict[str, bytes]:
    """Build the six payloads entirely in memory without reading any files."""
    rng = random.Random(SEED)
    payloads = {}
    for name, count in (("source_a", 12), ("source_b", 12), ("target_c", 9)):
        rows = []
        for i in range(count):
            row = {"sample_id": f"synthetic_cnt_{name}_{i + 1:02d}", "data_origin": ORIGIN}
            row.update(cnt_values(rng))
            if name == "target_c":
                row[CNT_LABEL] = i % 3 + 1
            rows.append(row)
        fields = CNT_FIELDS + ([CNT_LABEL] if name == "target_c" else [])
        payloads[f"cnt/synthetic/{name}.csv"] = csv_bytes(fields, rows)
    for name in ("new1", "new2"):
        rows = []
        for label in (1, 2, 3):
            data = cnt_values(rng)
            data[CNT_LABEL] = label
            rows.append({
                "sample_id": f"synthetic_cnt_{name}_{label:02d}",
                "data_origin": ORIGIN,
                "data": data,
            })
        payloads[f"cnt/synthetic/{name}.json"] = (
            json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")

    mos_rows = []
    for i in range(12):
        ar = draw(rng, 30, 100)
        n2 = draw(rng, 0, 20)
        hydrogen = draw(rng, 1, 10)
        h2s = 0.0 if i % 2 else draw(rng, 0.1, 1)
        oxygen = 0.1 if i % 4 == 2 else 0.0
        if (i, "h2s_flow_sccm") in MOS_MISSING:
            h2s = ""
        substrate, substrate_class = (
            ("quartz", "oxide"), ("sapphire", "oxide"), ("mica", "mica")
        )[i % 3]
        promoter = ("none", "NaCl", "oxygen", "PTAS")[i % 4]
        row = {
            "row_id": f"synthetic_mos2_{i + 1:02d}",
            "data_origin": ORIGIN,
            "growth_temperature_C": draw(rng, 600, 850),
            "mo_source_temperature_C": draw(rng, 300, 550),
            "s_source_temperature_C": draw(rng, 100, 200),
            "substrate_temperature_C": draw(rng, 550, 800),
            "total_flow_sccm": round(ar + n2 + hydrogen + (h2s or 0) + oxygen, 3),
            "ar_flow_sccm": ar,
            "n2_flow_sccm": n2,
            "h2_flow_sccm": hydrogen,
            "h2s_flow_sccm": h2s,
            "o2_flow_sccm": oxygen,
            "growth_pressure_Torr": draw(rng, 10, 100) if i % 2 else 760.0,
            "holding_time_min": draw(rng, 5, 25),
            "ramp_rate_C_min": draw(rng, 5, 20),
            "mo_precursor_mass_mg": draw(rng, 1, 10),
            "mo_precursor_thickness_nm": draw(rng, 1, 5),
            "s_precursor_mass_mg": draw(rng, 20, 80),
            "precursor_ratio": draw(rng, 2, 10),
            "promoter_used": int(promoter != "none"),
            "oxygen_used": int(oxygen > 0),
            "nacl_mass_mg": 0.1 if promoter == "NaCl" else 0.0,
            "growth_method": "CVD",
            "mo_precursor": ("MoO3", "MoCl5", "Mo(CO)6")[i % 3],
            "s_precursor": "sulfur" if h2s in ("", 0.0) else "H2S",
            "substrate": substrate,
            "substrate_class": substrate_class,
            "pressure_type": "LPCVD" if i % 2 else "APCVD",
            "promoter_type": promoter,
            "layer_class_3": i % 3,
        }
        for row_index, field in MOS_MISSING:
            if row_index == i:
                row[field] = ""
        mos_rows.append(row)
    payloads["mos2/synthetic/raw_inputs.csv"] = csv_bytes(MOS_FIELDS, mos_rows)
    return payloads


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: object, field: str) -> float:
    require(not isinstance(value, bool), f"{field}: boolean is not a numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field}: expected a finite number") from error
    require(math.isfinite(result), f"{field}: expected a finite number")
    return result


def read_csv(payload: bytes, fields: list[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    require(reader.fieldnames == fields, "CSV fields or field order differ from the declared schema")
    rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows), "Malformed CSV row")
    return rows


def check_metadata(rows: list[dict], id_field: str) -> None:
    ids = [row[id_field] for row in rows]
    require(len(ids) == len(set(ids)), "Sample identifiers must be unique within a file")
    require(all(value.startswith("synthetic_") for value in ids), "Identifiers must declare synthetic origin")
    require(all(row.get("data_origin") == ORIGIN for row in rows), "Missing synthetic / demonstration-only marker")


def check_cnt_row(row: dict) -> None:
    values = {field: finite(row[field], field) for field in CNT_FEATURES}
    hydrogen, carbon = values["H2 flow (sccm)"], values["C-flow (sccm)"]
    require(hydrogen >= 0 and carbon > 0, "CNT flows must be nonnegative, with positive carbon flow")
    require(values["Total flow (sccm)"] >= hydrogen + carbon, "CNT total flow is below H2 + carbon flow")
    require(math.isclose(values["H/C"], hydrogen / carbon, rel_tol=0, abs_tol=1e-9), "CNT H/C must equal H2 flow / carbon flow")
    for field in CNT_FEATURES:
        if field not in ("Carbon_Gibbs", "Carbon_Enthalpy", "Predicted Formation Energy"):
            require(values[field] >= 0, f"CNT {field} must be nonnegative")


def validate_file(name: str, payload: bytes) -> None:
    if name.endswith(".json"):
        rows = json.loads(payload)
        require(isinstance(rows, list) and len(rows) == 3, "Each candidate file must contain three records")
        require(all(list(row) == ["sample_id", "data_origin", "data"] for row in rows), "Candidate metadata schema differs")
        check_metadata(rows, "sample_id")
        labels = []
        for row in rows:
            data = row["data"]
            require(list(data) == [*CNT_FEATURES, CNT_LABEL], "Candidate data schema differs")
            check_cnt_row(data)
            require(type(data[CNT_LABEL]) is int, "Candidate labels must be integers")
            labels.append(data[CNT_LABEL])
        require(Counter(labels) == {1: 1, 2: 1, 3: 1}, "Each candidate file requires one artificial label per class")
    elif name.startswith("cnt/"):
        target = name.endswith("target_c.csv")
        rows = read_csv(payload, CNT_FIELDS + ([CNT_LABEL] if target else []))
        require(len(rows) == (9 if target else 12), "Unexpected CNT row count")
        check_metadata(rows, "sample_id")
        for row in rows:
            check_cnt_row(row)
        if target:
            require(Counter(row[CNT_LABEL] for row in rows) == {"1": 3, "2": 3, "3": 3}, "CNT target requires three artificial labels per class")
    else:
        rows = read_csv(payload, MOS_FIELDS)
        require(len(rows) == 12, "MoS2 requires 12 records")
        check_metadata(rows, "row_id")
        require(Counter(row["layer_class_3"] for row in rows) == {"0": 4, "1": 4, "2": 4}, "MoS2 requires four artificial labels per class")
        observed_missing = set()
        for i, row in enumerate(rows):
            values = {}
            for field in MOS_NUMERIC:
                if row[field] == "":
                    observed_missing.add((i, field))
                    continue
                values[field] = finite(row[field], field)
                require(values[field] >= 0, f"MoS2 {field} must be nonnegative")
            require(all(row[field] for field in MOS_CATEGORICAL), "MoS2 category cells must be populated")
            for field in ("promoter_used", "oxygen_used"):
                require(values[field] in (0, 1), f"MoS2 {field} must be 0 or 1")
            component_total = sum(values.get(field, 0) for field in MOS_FLOW_FIELDS)
            require(component_total > 0, "MoS2 component flows must support total-flow fallback")
            if "total_flow_sccm" in values:
                require(math.isclose(values["total_flow_sccm"], component_total, rel_tol=0, abs_tol=1e-6), "MoS2 total flow differs from the sum of provided component flows")
            if "h2s_flow_sccm" not in values:
                require(values["s_source_temperature_C"] > 0, "Missing H2S needs the explicit source-temperature fallback input")
        require(observed_missing == MOS_MISSING, "MoS2 empty cells differ from the declared missing-value cases")
        for field in ("mo_precursor", "substrate", "pressure_type", "promoter_type"):
            require(len({row[field] for row in rows}) >= 2, f"MoS2 {field} must cover multiple categories")


def check_files(output: Path, expected: dict[str, bytes]) -> None:
    errors = []
    for name, expected_payload in expected.items():
        try:
            actual = (output / name).read_bytes()
            validate_file(name, actual)
            require(actual == expected_payload, "Bytes differ from fixed-seed output (including encoding and line endings)")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{name}: {error}")
    if errors:
        raise ValueError("\n".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only schema, consistency and byte-for-byte determinism check; never create or repair files")
    parser.add_argument("--output", type=Path, default=ROOT / "examples", help="Examples root; default: %(default)s")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    expected = expected_files()
    try:
        for name, payload in expected.items():
            validate_file(name, payload)
        if not args.check:
            for name, payload in expected.items():
                destination = output / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
        check_files(output, expected)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Synthetic example check failed:\n{error}", file=sys.stderr)
        return 1
    action = "Checked (read-only)" if args.check else "Generated and checked"
    print(f"{action} {len(expected)} synthetic example files at {output} (seed {SEED}).")
    print("Demonstration only: no private inputs, model training, or paper-result reproduction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
