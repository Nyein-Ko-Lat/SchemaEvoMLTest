# evolve_autopipeline_patched.py
# Improvements:
# - Drop blank/Unnamed index columns early (prevents bogus join keys / target picks)
# - Per-task deterministic seed derived from global seed + task_id
# - Deterministic label detection with configurable priority (default Score first)
# - Protect label column from evolution ops (rename/drop/type/meaning/address)
# - Ignore blank/unnamed columns in join-key detection
# - Fix argparse typo (help=)
#
# Usage:
# python evolve_autopipeline_patched.py --root ./data/original --out ./data/output/evolved_tasks --limit 0 --seed 7
# Switch label priority:
# python evolve_autopipeline_patched.py --label_priority risk_category Score

import argparse
import hashlib
import json
import traceback
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import re


# -------------------------
# IO helpers
# -------------------------
READ_KW = dict(engine="python", on_bad_lines="skip")  # keep consistent with evaluation

def read_csv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:

    if nrows is None:
        df = pd.read_csv(path, **READ_KW)
    else:
        df = pd.read_csv(path, nrows=nrows, **READ_KW)
    return drop_blank_index_cols(df)

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


# -------------------------
# Column name hygiene
# -------------------------
def is_bad_colname(c: Any) -> bool:
    n = str(c).strip().lower()
    return n == "" or n.startswith("unnamed")

def drop_blank_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    bad = [c for c in df.columns if is_bad_colname(c)]
    if bad:
        df = df.drop(columns=bad)
    return df


# -------------------------
# Schema snapshots
# -------------------------
def schema_snapshot(df: pd.DataFrame, sample_unique: int = 3) -> Dict[str, Any]:
    cols = []
    for c in df.columns:
        s = df[c]
        cols.append({
            "name": str(c),
            "dtype": str(s.dtype),
            "null_ratio": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "sample": s.dropna().astype(str).unique()[:sample_unique].tolist(),
        })
    return {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "columns": cols}

def schema_diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    b = {x["name"]: x for x in before["columns"]}
    a = {x["name"]: x for x in after["columns"]}

    removed = sorted(list(set(b) - set(a)))
    added = sorted(list(set(a) - set(b)))
    common = sorted(list(set(a) & set(b)))

    dtype_changes = []
    for c in common:
        if b[c]["dtype"] != a[c]["dtype"]:
            dtype_changes.append({"column": c, "before": b[c]["dtype"], "after": a[c]["dtype"]})

    return {"added": added, "removed": removed, "dtype_changes": dtype_changes}


# -------------------------
# Helpers: schema signals
# -------------------------
def common_join_keys(dfs: List[pd.DataFrame], min_tables: int = 2) -> List[str]:
    """Join keys are columns present in at least `min_tables` tables. Ignores blank/unnamed cols."""
    if len(dfs) < 2:
        return []
    cnt = Counter()
    for d in dfs:
        cols = [str(c) for c in d.columns if not is_bad_colname(c)]
        cnt.update(cols)
    return sorted([c for c, k in cnt.items() if k >= min_tables])

def is_address_col(name: str) -> bool:
    n = name.lower()
    return "address" in n or "addr" in n

def looks_like_address_series(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if len(sample) == 0:
        return False
    comma_ratio = sample.str.contains(",", regex=False).mean()
    tokens_ratio = sample.str.split().map(len).gt(2).mean()
    return (comma_ratio > 0.3) or (tokens_ratio > 0.7)


# -------------------------
# Datetime parsing (consistent & reproducible)
# -------------------------
COMMON_DT_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M",
]

def detect_datetime_format(sample: pd.Series, min_ok: float = 0.80) -> Optional[str]:
    vals = sample.dropna().astype(str).head(300)
    if len(vals) == 0:
        return None
    best_fmt, best_ok = None, 0.0
    for fmt in COMMON_DT_FORMATS:
        parsed = pd.to_datetime(vals, format=fmt, errors="coerce")
        ok = float(parsed.notna().mean())
        if ok > best_ok:
            best_ok, best_fmt = ok, fmt
            if best_ok >= 0.99:
                break
    return best_fmt if best_ok >= min_ok else None

def safe_to_datetime(s: pd.Series, fmt: Optional[str]) -> Tuple[pd.Series, Dict[str, Any]]:
    ss = s.astype("string")
    if fmt:
        dt = pd.to_datetime(ss, format=fmt, errors="coerce")
        used = "explicit_format"
    else:
        dt = pd.to_datetime(ss, errors="coerce")
        used = "fallback_infer"
    info = {
        "parse_used": used,
        "valid_ratio": float(dt.notna().mean()),
    }
    return dt, info


# -------------------------
# Deterministic per-task seed
# -------------------------
def per_task_seed(global_seed: int, task_id: str) -> int:
    h = hashlib.sha1(f"{global_seed}:{task_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit


# -------------------------
# Label detection (Score-first by default, switchable)
# -------------------------
def detect_label_col_from_tables(
    tables: List[Tuple[str, pd.DataFrame]],
    label_priority: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find label column by priority across any training table.
    Returns (label_col, source_filename).
    """
    # exact match (case-insensitive) on priority names
    for wanted in label_priority:
        w = wanted.strip().lower()
        for fname, df in tables:
            lower_map = {str(c).lower(): str(c) for c in df.columns}
            if w in lower_map:
                return lower_map[w], fname
    return None, None

# -------------------------
# Check for already evolved tasks
# -------------------------
def is_task_already_evolved(out_folder: Path, variant: str) -> bool:
    """
    Return True if this task output folder looks complete for this variant.
    For baseline: target.csv + at least one training/test csv expected.
    For non-baseline: manifest.json + evo_log.json exist (created by process_task_folder).
    """
    if not out_folder.exists() or not out_folder.is_dir():
        return False

    if variant == "baseline":
        # baseline writes manifest.json and evo_log.json too,
        # but baseline copies files byte-for-byte, so accept either condition
        if (out_folder / "manifest.json").exists() and (out_folder / "evo_log.json").exists():
            return True
        # fallback: at least target.csv exists
        return (out_folder / "target.csv").exists()

    # non-baseline: must have manifest + evo_log
    return (out_folder / "manifest.json").exists() and (out_folder / "evo_log.json").exists()

# -------------------------
# Evolution plan (per task)
# -------------------------
def build_plan_for_task(
    training_tables_sample: List[pd.DataFrame],
    seed: int,
    mode_key_sensitive: str = "non_key",  # "key" or "non_key"
    protected_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    protected_cols = set(protected_cols or [])

    plan: Dict[str, Any] = {"seed": seed, "built_at": datetime.now().isoformat()}

    # Union schema for planning (stable)
    all_cols: List[str] = []
    for t in training_tables_sample:
        all_cols.extend(map(str, t.columns))
    all_cols = [c for c in dict.fromkeys(all_cols) if not is_bad_colname(c)]

    join_keys = common_join_keys(training_tables_sample, min_tables=2)
    plan["join_keys"] = join_keys

    non_special = [c for c in all_cols if not str(c).startswith("__")]
    non_key_cols = [c for c in non_special if c not in join_keys and c not in protected_cols]
    key_cols = [c for c in non_special if c in join_keys and c not in protected_cols]
    any_cols = [c for c in non_special if c not in protected_cols]

    def choose_from(cols: List[str]) -> Optional[str]:
        if not cols:
            return None
        return cols[int(rng.integers(0, len(cols)))]

    # Rename target
    if mode_key_sensitive == "key" and key_cols:
        plan["rename_col"] = choose_from(key_cols)
    else:
        plan["rename_col"] = choose_from(non_key_cols) or choose_from(any_cols)

    # Drop: prefer non-key, avoid protected
    plan["drop_col"] = choose_from(non_key_cols) or choose_from(any_cols)

    # Address split candidate
    addr_col = None
    for c in any_cols:
        if is_address_col(c):
            addr_col = c
            break
    plan["address_col"] = addr_col

    # Datetime candidate by name signals
    dt_candidates = [c for c in any_cols if ("date" in c.lower() or "time" in c.lower() or c.lower().endswith("_dt"))]
    plan["datetime_col"] = choose_from(dt_candidates) if dt_candidates else None

    # Store datetime format detected on training sample
    plan["datetime_format"] = None
    if plan["datetime_col"]:
        for t in training_tables_sample:
            if plan["datetime_col"] in map(str, t.columns):
                fmt = detect_datetime_format(t[plan["datetime_col"]])
                plan["datetime_format"] = fmt
                break

    # Meaning change candidates (skip protected)
    money_tokens = ["amount", "price", "sales", "revenue", "income", "tax", "cost", "msrp", "score"]
    plan["meaning_num_col"] = None
    for c in any_cols:
        lc = c.lower()
        if any(t in lc for t in money_tokens):
            plan["meaning_num_col"] = c
            break

    cat_tokens = ["state", "country", "city", "product", "category", "type", "status", "region", "risk"]
    plan["meaning_cat_col"] = None
    for c in any_cols:
        lc = c.lower()
        if any(t in lc for t in cat_tokens):
            plan["meaning_cat_col"] = c
            break

    # Fallbacks
    if plan["meaning_num_col"] is None:
        plan["meaning_num_col"] = choose_from(non_key_cols) or choose_from(any_cols)
    if plan["meaning_cat_col"] is None:
        plan["meaning_cat_col"] = choose_from(non_key_cols) or choose_from(any_cols)

    # Type evolution rules:
    plan["datetime_rule"] = rng.choice(["flip_date_time", "split_date_parts"]).item()

    # Severity knobs
    plan["category_keep_topk"] = int(rng.choice([2, 5, 10]).item())
    plan["unit_scale_divisor"] = float(rng.choice([10.0, 1000.0]).item())

    return plan


# -------------------------
# Operators (respect protected cols)
# -------------------------
def is_protected(col: Optional[str], plan: Dict[str, Any]) -> bool:
    prot = set(plan.get("protected_cols", []) or [])
    return (col is None) or (col in prot)

def op_add(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(plan["seed"])
    out = df.copy()
    out["__new_constant"] = "unknown"
    mask = rng.random(len(out)) < 0.2
    out["__new_sparse"] = np.where(mask, 1.0, np.nan)
    out["__row_id"] = np.arange(len(out))
    return out, {"op": "add", "added": ["__new_constant", "__new_sparse", "__row_id"], "sparse_ratio": float(mask.mean())}

def op_rename(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    c = plan.get("rename_col")
    if is_protected(c, plan):
        return out, {"op": "rename", "applied": None, "note": "rename_col protected or missing"}
    if not c or c not in out.columns:
        return out, {"op": "rename", "applied": None, "note": "rename_col not found in this file"}
    new = f"{c}__renamed"
    out = out.rename(columns={c: new})
    return out, {"op": "rename", "applied": {c: new}}

def op_drop(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    c = plan.get("drop_col")
    if is_protected(c, plan):
        return out, {"op": "drop", "applied": None, "note": "drop_col protected or missing"}
    if not c or c not in out.columns:
        return out, {"op": "drop", "applied": None, "note": "drop_col not found in this file"}
    if out.shape[1] <= 2:
        return out, {"op": "drop", "applied": None, "note": "too few columns"}
    out = out.drop(columns=[c])
    return out, {"op": "drop", "applied": c}

def op_type(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    info = {"op": "type", "changes": []}

    dt_col = plan.get("datetime_col")
    rule = plan.get("datetime_rule")
    fmt = plan.get("datetime_format")

    # skip if protected
    if dt_col and not is_protected(dt_col, plan) and dt_col in out.columns:
        before_dtype = str(out[dt_col].dtype)
        dt, parse_info = safe_to_datetime(out[dt_col], fmt=fmt)
        change_rec = {"column": dt_col, "before": before_dtype, **parse_info}
        valid = parse_info["valid_ratio"]

        if valid < 0.5:
            change_rec["note"] = "datetime parse poor"
            info["changes"].append(change_rec)
        else:
            if rule == "flip_date_time":
                has_time_ratio = float((dt.dt.hour.ne(0) | dt.dt.minute.ne(0) | dt.dt.second.ne(0)).mean())
                if has_time_ratio > 0.2:
                    out[dt_col] = dt.dt.strftime("%Y-%m-%d").where(dt.notna(), other=np.nan)
                    applied = "datetime_to_date_str"
                else:
                    out[dt_col] = dt.dt.strftime("%Y-%m-%d 00:00:00").where(dt.notna(), other=np.nan)
                    applied = "date_to_datetime_str"
                change_rec["rule"] = applied
                change_rec["after"] = str(out[dt_col].dtype)
                change_rec["has_time_ratio"] = has_time_ratio
                info["changes"].append(change_rec)

            elif rule == "split_date_parts":
                out[f"{dt_col}_year"] = dt.dt.year
                out[f"{dt_col}_month"] = dt.dt.month
                out[f"{dt_col}_day"] = dt.dt.day
                info["changes"].append({
                    "column": dt_col,
                    "rule": "add_date_parts",
                    **parse_info,
                    "added": [f"{dt_col}_year", f"{dt_col}_month", f"{dt_col}_day"],
                })

    # numeric-like string -> numeric (skip protected candidate)
    obj_cols = [c for c in out.columns if (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]))]
    numeric_candidate = None
    for c in obj_cols:
        if str(c).startswith("__") or is_protected(c, plan):
            continue
        sample = out[c].dropna().astype(str).head(300)
        if len(sample) == 0:
            continue
        numericish = sample.str.match(r"^\s*-?\d+(\.\d+)?\s*$").mean()
        if float(numericish) > 0.7:
            numeric_candidate = c
            break

    if numeric_candidate:
        c = numeric_candidate
        before = str(out[c].dtype)
        before_na = float(out[c].isna().mean())
        out[c] = pd.to_numeric(out[c], errors="coerce")
        after_na = float(out[c].isna().mean())
        info["changes"].append({
            "column": c,
            "before": before,
            "after": str(out[c].dtype),
            "rule": "to_numeric(errors=coerce)",
            "na_ratio_before": before_na,
            "na_ratio_after": after_na,
        })

    return out, info

def op_meaning(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    info = {"op": "meaning", "changes": []}

    cnum = plan.get("meaning_num_col")
    divisor = float(plan.get("unit_scale_divisor", 1000.0))
    if cnum and (not is_protected(cnum, plan)) and cnum in out.columns and pd.api.types.is_numeric_dtype(out[cnum]):
        before_stats = {"mean": float(pd.to_numeric(out[cnum], errors="coerce").mean())}
        out[cnum] = out[cnum] / divisor
        after_stats = {"mean": float(pd.to_numeric(out[cnum], errors="coerce").mean())}
        info["changes"].append({
            "type": "unit_scale",
            "column": cnum,
            "rule": f"divide_by_{divisor:g}",
            "before_stats": before_stats,
            "after_stats": after_stats,
        })

    ccat = plan.get("meaning_cat_col")
    topk = int(plan.get("category_keep_topk", 5))
    if ccat and (not is_protected(ccat, plan)) and ccat in out.columns:
        vc = out[ccat].value_counts(dropna=True)
        orig_k = int(len(vc))
        if orig_k > 0:
            keep = set(vc.head(min(topk, orig_k)).index.tolist())
            before_other_ratio = float((~out[ccat].isin(keep) & out[ccat].notna()).mean())
            out[ccat] = out[ccat].where(out[ccat].isin(keep), other="Other")
            new_k = int(out[ccat].nunique(dropna=True))
            info["changes"].append({
                "type": "category_collapse",
                "column": ccat,
                "rule": f"keep_top{topk}_else_Other",
                "cardinality_before": orig_k,
                "cardinality_after": new_k,
                "mapped_to_other_ratio": before_other_ratio,
            })

    return out, info

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().strip())

def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        k = _norm(cand)
        if k in norm_map:
            return norm_map[k]
    return None

def _find_address_parts(df: pd.DataFrame, base_col: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Find common address component columns. Uses:
    - prefix strategy: provider_street_address + provider_city/state/zip_code
    - generic strategy: addressline1/city/state/zip/postal/country...
    """
    cols_norm = {_norm(c): c for c in df.columns}

    parts = {"street": None, "city": None, "state": None, "region": None, "zip": None, "country": None}

    # 1) Prefix strategy: "provider_street_address" -> prefix "provider_"
    prefix = None
    if base_col:
        n = str(base_col)
        if "_" in n:
            prefix = n.split("_")[0] + "_"  # e.g., "provider_"

    if prefix:
        parts["city"]    = _first_present(df, [f"{prefix}city"])
        parts["state"]   = _first_present(df, [f"{prefix}state"])
        parts["zip"]     = _first_present(df, [f"{prefix}zip", f"{prefix}zipcode", f"{prefix}zip_code", f"{prefix}postal", f"{prefix}postalcode", f"{prefix}postal_code"])
        parts["country"] = _first_present(df, [f"{prefix}country"])
        # street might be the base col itself
        if base_col in df.columns:
            parts["street"] = base_col

    # 2) Generic strategy (fallback)
    if parts["street"] is None:
        parts["street"] = _first_present(df, ["address", "street_address", "street", "addressline1", "address1", "addr1"])
    if parts["city"] is None:
        parts["city"] = _first_present(df, ["city", "town"])
    if parts["state"] is None and parts["region"] is None:
        parts["state"]  = _first_present(df, ["state", "province"])
        parts["region"] = _first_present(df, ["region"])
    if parts["zip"] is None:
        parts["zip"] = _first_present(df, ["zip", "zipcode", "zip_code", "postal", "postalcode", "postal_code"])
    if parts["country"] is None:
        parts["country"] = _first_present(df, ["country"])

    # 3) Prefix group detection (xxx_city/state/zip etc.)
    pref = _detect_prefix_address_group(df)
    for k in parts:
        if parts[k] is None and pref.get(k):
            parts[k] = pref[k]

    return parts

def _combine_address(df: pd.DataFrame, parts: Dict[str, Optional[str]], out_col: str) -> pd.Series:
    def clean_col(cname: Optional[str]) -> pd.Series:
        if cname is None or cname not in df.columns:
            return pd.Series([""] * len(df), index=df.index, dtype="string")
        s = df[cname].astype("string").fillna("").str.strip()
        return s

    street = clean_col(parts.get("street"))
    city   = clean_col(parts.get("city"))
    state  = clean_col(parts.get("state")) if parts.get("state") else clean_col(parts.get("region"))
    zipp   = clean_col(parts.get("zip"))
    ctry   = clean_col(parts.get("country"))

    # build "street, city, state zip, country" (skip empty parts)
    line1 = street
    line2 = (city.where(city != "", other="") +
             (", " + state).where((state != "") & (city != ""), other=state))
    line3 = (zipp.where(zipp != "", other="") +
             (", " + ctry).where((ctry != "") & (zipp != ""), other=ctry))

    # stitch with separators but avoid ", ,"
    full = line1
    full = full + (", " + line2).where(line2 != "", other="")
    full = full + (", " + line3).where(line3 != "", other="")
    full = full.str.replace(r"^\s*,\s*", "", regex=True).str.replace(r"\s*,\s*$", "", regex=True)
    return full

def _split_freetext_address(s: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split common patterns:
    - "street, city, region" (comma-separated)
    - "street | city | region" (pipe-separated)
    If no separators, treat everything as street.
    """
    ss = s.astype("string").fillna("").str.strip()

    # Prefer comma, else pipe
    if ss.str.contains(",", regex=False).mean() > 0.2:
        parts = ss.str.split(",", n=2, expand=True)
    elif ss.str.contains("|", regex=False).mean() > 0.2:
        parts = ss.str.split("|", n=2, expand=True)
    else:
        # no split signal
        street = ss
        city = pd.Series([""] * len(ss), index=ss.index, dtype="string")
        region = pd.Series([""] * len(ss), index=ss.index, dtype="string")
        return street, city, region

    street = parts[0].astype("string").str.strip()
    city   = parts[1].astype("string").str.strip() if parts.shape[1] > 1 else pd.Series([""] * len(ss), index=ss.index, dtype="string")
    region = parts[2].astype("string").str.strip() if parts.shape[1] > 2 else pd.Series([""] * len(ss), index=ss.index, dtype="string")
    return street, city, region

def _detect_prefix_address_group(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Detect groups like xxx_city/state/zip/etc..
    Returns best-matching group (most parts found).
    """
    cols = list(df.columns)
    norm = {_norm(c): c for c in cols}

    # Find all "*_city" columns
    city_cols = [c for c in cols if _norm(c).endswith("city") and "_" in str(c)]
    best = {"street": None, "city": None, "state": None, "region": None, "zip": None, "country": None}
    best_score = 0

    for city_col in city_cols:
        prefix = str(city_col).rsplit("_", 1)[0]  # contributor, provider, etc.

        cand = {
            "street": _first_present(df, [f"{prefix}_street_address", f"{prefix}_street", f"{prefix}_address", f"{prefix}_addressline1", f"{prefix}_addr1"]),
            "city": city_col,
            "state": _first_present(df, [f"{prefix}_state", f"{prefix}_province"]),
            "zip": _first_present(df, [f"{prefix}_zip", f"{prefix}_zipcode", f"{prefix}_zip_code", f"{prefix}_postal", f"{prefix}_postalcode", f"{prefix}_postal_code"]),
            "country": _first_present(df, [f"{prefix}_country"]),
            "region": _first_present(df, [f"{prefix}_region"]),
        }

        score = sum(1 for k in ["city", "state", "zip", "country", "street"] if cand.get(k))
        if score > best_score:
            best_score = score
            best.update(cand)

    return best if best_score >= 2 else {"street": None, "city": None, "state": None, "region": None, "zip": None, "country": None}

def op_address_split(df: pd.DataFrame, plan: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evolution operator:
    - If dataset already has split parts (street + >=1 of city/state/zip/country): COMBINE into one column and DROP parts
    - Else if dataset has a free-text address column with separators: SPLIT into parts and DROP original
    - Else: no-op
    """
    out = df.copy()
    c = plan.get("address_col")  # may be None

    if c is not None and is_protected(c, plan):
        return out, {"op": "address_split", "applied": None, "note": "address_col protected"}

    # Detect address parts (works even if c is None)
    parts = _find_address_parts(out, base_col=c if (c in out.columns) else None)

    # Count how many location parts exist (besides street)
    loc_cols = [parts.get("city"), parts.get("state"), parts.get("region"), parts.get("zip"), parts.get("country")]
    loc_present = [x for x in loc_cols if x and x in out.columns]
    street_present = bool(parts.get("street")) and (parts["street"] in out.columns)

    # ---------
    # COMBINE branch (already split)
    # ---------
    if (street_present and len(loc_present) >= 1) or (len(loc_present) >= 2):
        # if no street column exists, use a stable base name
        base_name = parts.get("street") or "address"
        full_col = f"{base_name}_full_address"

        out[full_col] = _combine_address(out, parts, full_col).replace({"": np.nan})

        drop_cols = set()
        for k, col in parts.items():
            if col and col in out.columns and col != full_col:
                drop_cols.add(col)

        out = out.drop(columns=sorted(drop_cols), errors="ignore")

        return out, {
            "op": "address_split",
            "mode": "combine_then_drop_parts",
            "created": full_col,
            "dropped": sorted(list(drop_cols))[:30],
            "n_dropped": int(len(drop_cols)),
            "used_parts": {k: v for k, v in parts.items() if v},
        }

    # ---------
    # SPLIT branch (free-text)
    # ---------
    # pick a free-text address column if plan.address_col missing
    addr_col = c if (c and c in out.columns) else parts.get("street")
    if addr_col and addr_col in out.columns:
        s = out[addr_col].astype("string")
        # only split if there are separators (comma/pipe)
        sep_score = max(s.str.contains(",", regex=False).mean(), s.str.contains("|", regex=False).mean())
        if sep_score > 0.2:
            street, city, region = _split_freetext_address(s)
            out[f"{addr_col}_street"] = street.replace({"": np.nan})
            out[f"{addr_col}_city"] = city.replace({"": np.nan})
            out[f"{addr_col}_region"] = region.replace({"": np.nan})

            # DROP original to make it a real schema evolution
            out = out.drop(columns=[addr_col], errors="ignore")

            return out, {
                "op": "address_split",
                "mode": "split_then_drop_original",
                "applied": addr_col,
                "created": [f"{addr_col}_street", f"{addr_col}_city", f"{addr_col}_region"],
                "dropped": [addr_col],
                "separator_score": float(sep_score),
            }

    return out, {
        "op": "address_split",
        "applied": None,
        "note": "no split parts to combine, and no free-text address to split"
    }

OPS = {
    "add": op_add,
    "rename": op_rename,
    "type": op_type,
    "drop": op_drop,
    "meaning": op_meaning,
    "address_split": op_address_split,
}


# -------------------------
# Batch processing
# -------------------------
@dataclass
class TaskSummary:
    folder: str
    status: str
    n_training_files: int
    n_test_files: int
    target_present: bool
    out_folder: Optional[str]
    applied_ops: List[str]
    error: Optional[str]
    label_col: Optional[str]
    label_source_file: Optional[str]


def evolve_file_with_plan(path: Path, out_path: Path, ops: List[str], plan: Dict[str, Any]) -> Dict[str, Any]:
    df = read_csv(path)
    before = schema_snapshot(df)
    op_logs = []

    cur = df
    for op in ops:
        fn = OPS[op]
        cur, info = fn(cur, plan)
        op_logs.append(info)

    after = schema_snapshot(cur)
    diff = schema_diff(before, after)

    write_csv(cur, out_path)

    return {
        "input": str(path),
        "output": str(out_path),
        "ops": op_logs,
        "rows_before": before["n_rows"],
        "rows_after": after["n_rows"],
        "schema_diff": diff,
    }


def process_task_folder(
    folder: Path,
    out_root: Path,
    ops: List[str],
    global_seed: int,
    mode_key_sensitive: str,
    plan_sample_rows: int,
    label_priority: List[str],
) -> TaskSummary:
    try:
        target = folder / "target.csv"
        training_files = sorted(folder.glob("training_*.csv"))
        test_files = sorted(folder.glob("test_*.csv"))

        if not target.exists():
            return TaskSummary(str(folder), "skipped_no_target", len(training_files), len(test_files),
                               False, None, ops, None, None, None)
        if len(training_files) == 0 and len(test_files) == 0:
            return TaskSummary(str(folder), "skipped_no_train_test", 0, 0, True, None, ops, None, None, None)

        out_folder = out_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        # Copy target unchanged (do NOT evolve target)
        copy_file(target, out_folder / "target.csv")

        # Sample training tables for plan building
        training_tables_sample: List[pd.DataFrame] = []
        named_samples: List[Tuple[str, pd.DataFrame]] = []
        for p in training_files:
            df_s = read_csv(p, nrows=plan_sample_rows)
            training_tables_sample.append(df_s)
            named_samples.append((p.name, df_s))

        # Detect label column by priority across tables (Score-first by default)
        label_col, label_source_file = detect_label_col_from_tables(named_samples, label_priority)

        # Build per-task seed and plan
        task_seed = per_task_seed(global_seed, folder.name)
        protected_cols = []
        if label_col:
            protected_cols.append(label_col)

        plan = build_plan_for_task(
            training_tables_sample,
            seed=task_seed,
            mode_key_sensitive=mode_key_sensitive,
            protected_cols=protected_cols,
        )
        plan["task_id"] = folder.name
        plan["ops"] = ops
        plan["plan_sample_rows"] = plan_sample_rows
        plan["label_col"] = label_col
        plan["label_source_file"] = label_source_file
        plan["protected_cols"] = protected_cols
        plan["global_seed"] = global_seed
        plan["task_seed"] = task_seed
        plan["label_priority"] = label_priority

        (out_folder / "manifest.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        per_file_logs = {
            "task_folder": str(folder),
            "created_at": datetime.now().isoformat(),
            "ops": ops,
            "mode_key_sensitive": mode_key_sensitive,
            "plan": plan,
            "files": [],
            "note": "target.csv copied unchanged; training/test evolved using the same per-task plan; blank/unnamed index cols dropped",
        }

        for f in training_files:
            log = evolve_file_with_plan(f, out_folder / f.name, ops, plan)
            per_file_logs["files"].append(log)

        for f in test_files:
            log = evolve_file_with_plan(f, out_folder / f.name, ops, plan)
            per_file_logs["files"].append(log)

        (out_folder / "evo_log.json").write_text(json.dumps(per_file_logs, indent=2), encoding="utf-8")

        return TaskSummary(str(folder), "ok", len(training_files), len(test_files), True, str(out_folder),
                           ops, None, label_col, label_source_file)

    except Exception:
        return TaskSummary(str(folder), "failed", 0, 0, (folder / "target.csv").exists(),
                           None, ops, traceback.format_exc()[:4000], None, None)


def write_baseline_task(folder: Path, out_root: Path) -> TaskSummary:
    try:
        target = folder / "target.csv"
        training_files = sorted(folder.glob("training_*.csv"))
        test_files = sorted(folder.glob("test_*.csv"))

        if not target.exists():
            return TaskSummary(str(folder), "skipped_no_target", len(training_files), len(test_files),
                               False, None, ["baseline"], None, None, None)
        if len(training_files) == 0 and len(test_files) == 0:
            return TaskSummary(str(folder), "skipped_no_train_test", 0, 0, True, None, ["baseline"], None, None, None)

        out_folder = out_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        copy_file(target, out_folder / "target.csv")
        for f in training_files:
            # keep baseline byte-for-byte (do not drop blank col here)
            copy_file(f, out_folder / f.name)
        for f in test_files:
            copy_file(f, out_folder / f.name)

        (out_folder / "manifest.json").write_text(json.dumps({
            "task_id": folder.name,
            "created_at": datetime.now().isoformat(),
            "ops": ["baseline"],
            "note": "No evolution applied. Files copied byte-for-byte."
        }, indent=2), encoding="utf-8")

        (out_folder / "evo_log.json").write_text(json.dumps({
            "task_folder": str(folder),
            "created_at": datetime.now().isoformat(),
            "ops": ["baseline"],
            "note": "No evolution applied. Files copied as-is."
        }, indent=2), encoding="utf-8")

        return TaskSummary(str(folder), "ok", len(training_files), len(test_files), True,
                           str(out_folder), ["baseline"], None, None, None)

    except Exception:
        return TaskSummary(str(folder), "failed", 0, 0, (folder / "target.csv").exists(),
                           None, ["baseline"], traceback.format_exc()[:4000], None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="C:/ThesisData/original",
                    help="Root directory containing task folders")
    ap.add_argument("--out", default="C:/ThesisData/output/evolved_tasks",
                    help="Output root for evolved datasets")
    ap.add_argument("--limit", type=int, default=50,
                    help="Limit number of folders (0=all)")
    ap.add_argument("--offset", type=int, default=0,
                    help="Start from this task index (0-based) after sorting task folders.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip tasks already evolved (output folder exists and has manifest/evo_log).")
    ap.add_argument("--seed", type=int, default=7, help="Global random seed")
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "add", "drop", "rename", "type", "meaning", "address_split"],
                    help="Which evo variants to generate")
    ap.add_argument("--key_mode", choices=["key", "non_key"], default="key",
                    help="When a join key exists, apply rename/drop to key or non-key columns.")
    ap.add_argument("--plan_sample_rows", type=int, default=500,
                    help="Rows to sample from training files to build evolution plan (speed).")
    ap.add_argument("--label_priority", nargs="+",
                    default=["Score", "label", "target", "y", "class", "outcome", "risk_category"],
                    help="Label column name priority (case-insensitive). Put Score first if desired.")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    task_folders = sorted([p for p in root.iterdir() if p.is_dir()])

    if args.offset and args.offset > 0:
        task_folders = task_folders[args.offset:]

    if args.limit and args.limit > 0:
        task_folders = task_folders[:args.limit]

    all_summaries = []

    for variant in args.variants:
        variant_root = out_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)

        summaries = []
        for i, folder in enumerate(task_folders, 1):
            out_folder = variant_root / folder.name  # <-- YES, here

            if args.resume and is_task_already_evolved(out_folder, variant):
                d = {
                    "folder": str(folder),
                    "status": "skipped_resume_exists",
                    "n_training_files": len(list(folder.glob("training_*.csv"))),
                    "n_test_files": len(list(folder.glob("test_*.csv"))),
                    "target_present": (folder / "target.csv").exists(),
                    "out_folder": str(out_folder),
                    "applied_ops": [variant],
                    "error": None,
                    "label_col": None,
                    "label_source_file": None,
                    "variant": variant,
                }
                summaries.append(d)
                all_summaries.append(d)
                continue

            # otherwise do normal processing
            if variant == "baseline":
                s = write_baseline_task(folder, variant_root)
                d = asdict(s)
                d["variant"] = variant
            else:
                s = process_task_folder(
                    folder=folder,
                    out_root=variant_root,
                    ops=[variant],
                    global_seed=args.seed,
                    mode_key_sensitive=args.key_mode,
                    plan_sample_rows=args.plan_sample_rows,
                    label_priority=args.label_priority,
                )
                d = asdict(s)
                d["variant"] = variant

            summaries.append(d)
            all_summaries.append(d)

            if i % 25 == 0:
                print(f"[{variant}] Processed {i}/{len(task_folders)}")

        pd.DataFrame(summaries).to_csv(variant_root / "batch_summary.csv", index=False)
        (variant_root / "batch_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    pd.DataFrame(all_summaries).to_csv(out_root / "batch_summary_ALL.csv", index=False)
    (out_root / "batch_summary_ALL.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print("Done.")
    print("Global CSV :", out_root / "batch_summary_ALL.csv")


if __name__ == "__main__":
    main()