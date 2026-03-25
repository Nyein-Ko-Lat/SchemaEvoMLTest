import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge


# ---------------------------
# Settings / constants
# ---------------------------
READ_KW_FAST = dict(engine="c")
READ_KW_FALLBACK = dict(engine="python", on_bad_lines="skip")

COMMON_DT_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]

RANDOM_SEED = 42


# ---------------------------
# Failure taxonomy
# ---------------------------
class FailType(str, Enum):
    IO_ERROR = "IO_ERROR"
    JOIN_ERROR = "JOIN_ERROR"
    TARGET_ERROR = "TARGET_ERROR"
    PREPROCESS_ERROR = "PREPROCESS_ERROR"
    MODEL_FIT_ERROR = "MODEL_FIT_ERROR"
    PREDICT_ERROR = "PREDICT_ERROR"
    EVAL_ERROR = "EVAL_ERROR"
    RESOURCE_ERROR = "RESOURCE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


# ---------------------------
# Utilities
# ---------------------------
def sha1_file(path: Path, max_bytes: int = 5_000_000) -> str:
    """Fast-ish fingerprint: sha1 over first max_bytes bytes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        chunk = f.read(max_bytes)
    h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "sha1_head": sha1_file(path),
    }


def env_info() -> Dict[str, Any]:
    import sklearn

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "seed": RANDOM_SEED,
    }


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_json(obj: Any) -> Any:
    """Make nested objects JSON serializable."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


# ---------------------------
# IO
# ---------------------------

def read_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV reader that repairs rows where the entire CSV row
    is embedded into one quoted cell (Excel corruption pattern).
    """
    try:
        # First try normal read
        df = pd.read_csv(path, engine="python")
        return df
    except Exception:
        pass

        # 2) Slower normal fallback
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            pass

        # 3) Repair only as last resort
        raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if not raw_lines:
            return pd.DataFrame()

        header = raw_lines[0]
        n_cols = len(next(csv.reader([header])))

        out_rows = [next(csv.reader([header]))]

        for line in raw_lines[1:]:
            row = next(csv.reader([line]))

            if len(row) == n_cols:
                out_rows.append(row)
                continue

            # embedded-row corruption pattern
            if (
                    len(row) >= 2
                    and row[0].strip().isdigit()
                    and row[1]
                    and "," in row[1]
                    and all(x == "" for x in row[2:])
            ):
                idx = row[0]
                embedded = row[1]
                embedded_row = next(csv.reader([embedded]))
                repaired = [idx] + embedded_row

                if len(repaired) < n_cols:
                    repaired += [""] * (n_cols - len(repaired))
                else:
                    repaired = repaired[:n_cols]

                out_rows.append(repaired)
                continue

            # fallback pad/truncate
            if len(row) < n_cols:
                row += [""] * (n_cols - len(row))
            else:
                row = row[:n_cols]

            out_rows.append(row)

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerows(out_rows)
        buf.seek(0)

        return pd.read_csv(buf, engine="c")


# ---------------------------
# Join logic
# ---------------------------
def _is_id_like(colname: str) -> bool:
    n = str(colname).lower()
    return n == "id" or n.endswith("_id") or n in ("key", "uid") or ("id" in n and len(n) <= 12)

def pick_join_key(a: pd.DataFrame, b: pd.DataFrame) -> str:
    common = [c for c in a.columns if c in b.columns]
    if not common:
        raise RuntimeError("No common columns between tables.")

    id_like = [c for c in common if _is_id_like(c)]
    candidates = id_like or common

    best = None
    best_score = -1.0
    for c in candidates:
        a_u = a[c].nunique(dropna=True) / max(len(a), 1)
        b_u = b[c].nunique(dropna=True) / max(len(b), 1)
        score = min(a_u, b_u)
        if score > best_score:
            best_score, best = score, c
    return best

def join_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    if len(tables) == 1:
        return tables[0].copy()

    df = tables[0].copy()
    for t in tables[1:]:
        t = t.copy()
        key = pick_join_key(df, t)
        #check for join table row count.
        exp_mult = estimate_merge_expansion(df, t, key)
        if exp_mult > 5.0:  # tune threshold (3–10)
            raise RuntimeError(f"Join expansion too large on key={key}: estimated x{exp_mult:.2f}")

        left_dup = float(df[key].duplicated().mean()) if len(df) else 0.0
        right_dup = float(t[key].duplicated().mean()) if len(t) else 0.0
        if left_dup > 0.01 and right_dup > 0.01:
            raise RuntimeError(f"Many-to-many risk on key={key} (dupL={left_dup:.3f}, dupR={right_dup:.3f})")

        df = df.merge(t, on=key, how="left")
    return df


def prepare_y_and_tasktype(y_raw: pd.Series) -> tuple[np.ndarray, str, dict, np.ndarray]:
    """
    Returns:
      y_final (np.ndarray),
      task_type ("classification"/"regression"),
      y_meta (dict),
      keep_mask (np.ndarray bool, same length as original y_raw)
    """
    y_meta: Dict[str, Any] = {}

    # Normalize index
    y = y_raw.reset_index(drop=True)

    # Boolean => classification
    uniq = set(pd.Series(y.dropna().unique()).tolist())
    if pd.api.types.is_bool_dtype(y) or uniq.issubset({True, False}):
        y_meta["y_kind"] = "boolean"
        y_filled = y.fillna(False).astype(int).to_numpy()
        keep_mask = np.ones(len(y_filled), dtype=bool)
        return y_filled, "classification", y_meta, keep_mask

    # Try numeric conversion
    y_num = pd.to_numeric(y, errors="coerce")
    numeric_ratio = float(y_num.notna().mean())
    y_meta["numeric_ratio"] = numeric_ratio

    # Mostly numeric => regression OR small-int classification
    if numeric_ratio >= 0.85:
        keep_mask = y_num.notna().to_numpy()
        y_clean = y_num[keep_mask].to_numpy()

        uniq_n = int(pd.Series(y_clean).nunique())
        is_int_like = bool(np.allclose(y_clean, np.round(y_clean)))

        if is_int_like and uniq_n <= 20 and (uniq_n / max(len(y_clean), 1)) < 0.2:
            y_meta["y_kind"] = "numeric_int_class"
            return y_clean, "classification", y_meta, keep_mask

        y_meta["y_kind"] = "numeric_reg"
        return y_clean, "regression", y_meta, keep_mask

    # Otherwise => categorical classification
    y_meta["y_kind"] = "categorical"
    y_str = y.astype("string").fillna("NA").astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    y_meta["n_classes"] = int(len(le.classes_))
    y_meta["classes_head"] = list(map(str, le.classes_[:30]))

    keep_mask = np.ones(len(y_enc), dtype=bool)
    return y_enc, "classification", y_meta, keep_mask


# ---------------------------
# Target inference
# ---------------------------
def _is_date_like(colname: str) -> bool:
    n = str(colname).lower()
    return ("date" in n) or ("time" in n) or n.endswith("_dt")


def infer_target_from_training_table(df: pd.DataFrame) -> str:
    preferred = [
        "target", "label", "y", "class", "outcome",
        "score", "sales", "price", "amount", "revenue", "income", "budget",
        "grade", "risk_category"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p in lower_map:
            return lower_map[p]

    # helper filters
    def _is_date_like(colname: str) -> bool:
        n = str(colname).lower()
        return ("date" in n) or ("time" in n) or n.endswith("_dt")

    def _is_id_like(colname: str) -> bool:
        n = str(colname).lower()
        return n == "id" or n.endswith("_id") or n in ("key", "uid") or ("id" in n and len(n) <= 12)

    # ---------- B) numeric substitute logic ----------
    n = len(df)
    best = None
    best_score = -1.0

    for c in df.columns:
        if _is_date_like(c):
            continue
        if _is_id_like(c):
            continue

        s_num = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = float(s_num.notna().mean())
        if valid_ratio < 0.85:
            continue

        nunq = int(s_num.dropna().nunique())
        if n > 0 and (nunq / n) > 0.95:
            continue

        score = valid_ratio - abs((nunq / max(n, 1)) - 0.2)
        if score > best_score:
            best_score = score
            best = c

    if best is not None:
        return best

    # ---------- C) Semantic fallback (only runs if A+B failed) ----------
    semantic_priority = [
        # common “score” / “metric” targets
        "score", "happiness_score", "rating", "metric",
        # revenue
        "sales", "revenue", "amount", "price", "fare", "cost",
        # ranks can be targets too (ordinal/regression)
        "rank", "ranking",
        # counts
        "count", "qty", "quantity",
    ]

    # Find candidate cols by name contains
    candidates = []
    for c in df.columns:
        lc = str(c).lower()
        if _is_date_like(c) or _is_id_like(c):
            continue
        for token in semantic_priority:
            if token in lc:
                candidates.append(c)
                break

    # Quality checks for semantic candidates
    def ok_target(col: str) -> bool:
        s = df[col]
        s_num = pd.to_numeric(s, errors="coerce")
        valid_ratio = float(s_num.notna().mean())
        if valid_ratio >= 0.30:
            # must not be almost-constant
            nunq = int(s_num.dropna().nunique())
            if nunq <= 1:
                return False
            return True

        # otherwise allow categorical targets if enough non-null and not too unique
        s_str = s.astype("string")
        nonnull = float(s_str.notna().mean())
        if nonnull < 0.30:
            return False
        nunq = int(s_str.dropna().nunique())
        if n > 0 and (nunq / n) > 0.95:
            return False
        if nunq <= 1:
            return False
        return True

    for c in candidates:
        if ok_target(c):
            return c

    raise RuntimeError("No suitable target column found.")


def decide_task_type(y_train: pd.Series) -> str:
    y = y_train.dropna()
    if len(y) == 0:
        return "regression"
    uniq = int(y.nunique())
    is_int_like = bool(np.allclose(y.values, np.round(y.values)))
    if is_int_like and uniq <= 20 and (uniq / max(len(y), 1)) < 0.2:
        return "classification"
    return "regression"


# ---------------------------
# Schema / profiling helpers
# ---------------------------
def schema_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(map(str, df.columns.tolist())),
        "dtypes": dtypes,
    }


def basic_na_profile(df: pd.DataFrame) -> Dict[str, Any]:
    if df.shape[1] == 0:
        return {"mean_row_na_frac": None, "p95_row_na_frac": None}
    row_na = df.isna().mean(axis=1).values
    return {
        "mean_row_na_frac": round(float(np.mean(row_na)), 2),
        "p95_row_na_frac": round(float(np.quantile(row_na, 0.95)), 2),
    }


def align_test_to_train_strict(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    train_cols = list(X_train.columns)
    missing = [c for c in train_cols if c not in X_test.columns]
    if missing:
        raise RuntimeError(f"STRICT_SCHEMA_FAIL: missing_in_test={missing[:20]} (total={len(missing)})")
    return X_test[train_cols]

def align_test_to_train(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    train_cols = list(X_train.columns)
    train_set = set(train_cols)
    test_set = set(X_test.columns)

    missing_in_test = sorted(list(train_set - test_set))
    extra_in_test = sorted(list(test_set - train_set))

    X_test2 = X_test.copy()
    for c in missing_in_test:
        X_test2[c] = np.nan
    if extra_in_test:
        X_test2 = X_test2.drop(columns=extra_in_test)

    X_test2 = X_test2[train_cols]
    return X_test2, {"missing_in_test": missing_in_test, "extra_in_test": extra_in_test}


# ---------------------------
# Datetime parsing -> features
# ---------------------------
def detect_dt_format(s: pd.Series, min_ok: float = 0.8) -> Optional[str]:
    vals = s.dropna().astype(str).head(300)
    if len(vals) == 0:
        return None
    best_fmt, best_ok = None, 0.0
    for fmt in COMMON_DT_FORMATS:
        ok = pd.to_datetime(vals, format=fmt, errors="coerce").notna().mean()
        ok = float(ok)
        if ok > best_ok:
            best_ok, best_fmt = ok, fmt
            if best_ok >= 0.99:
                break
    return best_fmt if best_ok >= min_ok else None


def parse_datetime_columns(X_train: pd.DataFrame, X_other: pd.DataFrame, threshold: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    X_train = X_train.copy()
    X_other = X_other.copy()
    dt_info: Dict[str, Any] = {}

    for c in X_train.columns:
        if pd.api.types.is_object_dtype(X_train[c]) or pd.api.types.is_string_dtype(X_train[c]):
            fmt = detect_dt_format(X_train[c], min_ok=threshold)
            if fmt:
                tr = pd.to_datetime(X_train[c].astype("string"), format=fmt, errors="coerce")
                X_train[c] = tr

                if c in X_other.columns:
                    ot = pd.to_datetime(X_other[c].astype("string"), format=fmt, errors="coerce")
                    X_other[c] = ot
                    other_ok = float(ot.notna().mean())
                else:
                    other_ok = None

                dt_info[c] = {
                    "format": fmt,
                    "train_valid_ratio": float(tr.notna().mean()),
                    "other_valid_ratio": other_ok,
                }
    return X_train, X_other, dt_info


def datetime_cols_to_features(X: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
    X = X.copy()
    for c in datetime_cols:
        y, m, d = c + "_year", c + "_month", c + "_day"
        if c in X.columns and pd.api.types.is_datetime64_any_dtype(X[c]):
            X[y] = X[c].dt.year
            X[m] = X[c].dt.month
            X[d] = X[c].dt.day
        else:
            X[y] = np.nan
            X[m] = np.nan
            X[d] = np.nan
        if c in X.columns:
            X = X.drop(columns=[c])
    return X


def normalize_string_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("object")
    return df


# ---------------------------
# Model pipeline
# ---------------------------
def make_ohe():
    # ALWAYS sparse to avoid OOM with high-cardinality categoricals
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_pipeline(X_train: pd.DataFrame, task_type: str) -> Pipeline:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=1.0,  # keep sparse output when possible
    )

    if task_type == "classification":
        model = LogisticRegression(
            solver = "saga",
            max_iter = 2000,
            n_jobs = 1,
            penalty = "l2",
        )
    else:
        model = Ridge()

    return Pipeline([("pre", pre), ("model", model)])


# ---------------------------
# Silent failure metrics
# ---------------------------
def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 0.00) -> float:
    """Jensen–Shannon divergence for distributions along last axis, averaged over rows."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return round(float(np.mean(0.5 * (kl_pm + kl_qm))), 2)


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Simple KS statistic without scipy (approx via sorted CDFs)."""
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a_s = np.sort(a)
    b_s = np.sort(b)
    x = np.sort(np.unique(np.concatenate([a_s, b_s])))
    a_cdf = np.searchsorted(a_s, x, side="right") / len(a_s)
    b_cdf = np.searchsorted(b_s, x, side="right") / len(b_s)
    return round(float(np.max(np.abs(a_cdf - b_cdf))), 2)


# ---------------------------
# Estimate row count after join
# ---------------------------
def estimate_merge_expansion(left: pd.DataFrame, right: pd.DataFrame, key: str, sample: int = 200000) -> float:
    """
    Estimate how much a left merge can expand row count due to multiple matches in right.
    Returns estimated multiplier (>=1).
    """
    if len(left) == 0:
        return 1.0
    # sample left keys to reduce cost
    lk = left[key]
    if len(lk) > sample:
        lk = lk.sample(sample, random_state=0)

    # count matches in right for those keys
    rc = right[key].value_counts(dropna=True)
    matches = lk.map(rc).fillna(0).astype(int)
    # expected rows after merge per sampled row = max(1, matches)
    mult = round(float(np.mean(np.maximum(1, matches.values))), 2)
    return mult
# ---------------------------
# Load & join
# ---------------------------
def load_joined_tables(task_folder: Path, pattern: str) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    paths = sorted(task_folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"Missing files: {pattern}")
    fps = [file_fingerprint(p) for p in paths]

    tables = [read_csv(p) for p in paths]
    df = join_tables(tables)

    # Drop unnamed columns
    df = df.loc[:, [c for c in df.columns if str(c).strip() != "" and not str(c).lower().startswith("unnamed")]]
    return df, fps



#----------------------------
#Load already done folder/task from output folder
#----------------------------
def load_done_ids(jsonl_path: Path) -> set[str]:
    done = set()
    if not jsonl_path.exists():
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "task_id" in obj:
                    done.add(str(obj["task_id"]))
            except Exception:
                pass
    return done

# ---------------------------
# Relational-safe multi-table feature builder
# Base table = table that contains y (label).
# Other tables are attached via:
#  - 1:1 merge (safe)
#  - 1:many aggregation (count/nunique/mean/std/min/max) then merge (safe)
#  - many:many => skip
# ---------------------------

def drop_bad_cols(df: pd.DataFrame) -> pd.DataFrame:
    # drop blank/unnamed index columns early
    keep = [c for c in df.columns if str(c).strip() != "" and not str(c).lower().startswith("unnamed")]
    return df.loc[:, keep]

def find_label_table(training_tables: List[Tuple[Path, pd.DataFrame]]) -> Tuple[int, str]:
    """
    Pick base table index + label column name.
    Strategy:
      1) If any table contains preferred label names, pick that (first match).
      2) Else: pick table where substitute logic finds best target.
    """
    preferred = ["target", "label", "y", "class", "outcome", "score", "risk_category", "grade"]
    # pass 1: preferred names
    for i, (p, df) in enumerate(training_tables):
        lower_map = {str(c).lower(): c for c in df.columns}
        for name in preferred:
            if name in lower_map:
                return i, str(lower_map[name])

    # pass 2: substitute logic
    best_i, best_y = None, None
    for i, (p, df) in enumerate(training_tables):
        try:
            y = infer_target_from_training_table(df)
            # sanity: y must exist and have some non-null values
            if y in df.columns and df[y].notna().mean() > 0.05:
                best_i, best_y = i, str(y)
                break
        except Exception:
            continue

    if best_i is None or best_y is None:
        raise RuntimeError("TARGET_ERROR: could not find a label column in any training table.")
    return best_i, best_y

def best_common_key(base: pd.DataFrame, other: pd.DataFrame) -> Optional[str]:
    """
    Choose a join key among common columns.
    Prefer id-like, then highest min-uniqueness between tables.
    """
    common = [c for c in base.columns if c in other.columns]
    common = [c for c in common if str(c).strip() != "" and not str(c).lower().startswith("unnamed")]
    if not common:
        return None

    id_like = [c for c in common if _is_id_like(c)]
    candidates = id_like or common

    best, best_score = None, -1.0
    for c in candidates:
        a_u = base[c].nunique(dropna=True) / max(len(base), 1)
        b_u = other[c].nunique(dropna=True) / max(len(other), 1)
        score = min(a_u, b_u)
        if score > best_score:
            best_score, best = score, c
    return best

def relationship_type(base: pd.DataFrame, other: pd.DataFrame, key: str) -> str:
    """
    Determine relationship based on duplicate rate of key.
      base unique + other dup => one_to_many (other is child)
      base unique + other unique => one_to_one
      base dup + other unique => many_to_one (other is parent)
      base dup + other dup => many_to_many
    """
    if len(base) == 0 or len(other) == 0:
        return "unknown"

    base_dup = round(float(base[key].duplicated().mean()), 2) if key in base.columns else 1.0
    other_dup = round(float(other[key].duplicated().mean()), 2) if key in other.columns else 1.0

    base_unique = base_dup < 0.01
    other_unique = other_dup < 0.01

    if base_unique and not other_unique:
        return "one_to_many"
    if base_unique and other_unique:
        return "one_to_one"
    if (not base_unique) and other_unique:
        return "many_to_one"
    return "many_to_many"

def aggregate_child(child: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Aggregate child table into fixed-width features per key.
    Keeps it cheap + robust.
    """
    out = child.copy()
    out = out.drop(columns=[c for c in out.columns if c == key], errors="ignore")
    g = child.groupby(key, dropna=False)

    feats = {}
    # count rows per key
    feats[f"__agg__{key}__row_count"] = g.size()

    # numeric aggregations
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for c in num_cols:
        s = g[c]
        feats[f"__agg__{c}__mean"] = s.mean()
        feats[f"__agg__{c}__std"] = s.std()
        feats[f"__agg__{c}__min"] = s.min()
        feats[f"__agg__{c}__max"] = s.max()

    # categorical-ish aggregations
    cat_cols = [c for c in out.columns if (pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c]) or pd.api.types.is_categorical_dtype(out[c]))]
    for c in cat_cols:
        s = g[c]
        feats[f"__agg__{c}__nunique"] = s.nunique(dropna=True)

    agg_df = pd.DataFrame(feats)
    agg_df.index.name = key
    agg_df = agg_df.reset_index()
    return agg_df

def build_relational_features(
    train_tables: List[Tuple[Path, pd.DataFrame]],
    test_tables: List[Tuple[Path, pd.DataFrame]],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, Any]]:
    """
    Returns:
      X_train (pd.DataFrame),
      y_raw (pd.Series)   # raw label column (NOT converted here!)
      X_test  (pd.DataFrame),
      join_log (dict)
    """
    join_log: Dict[str, Any] = {"base_table": None, "y_col": None, "steps": []}

    base_idx, y_col = find_label_table(train_tables)
    base_path, base_train = train_tables[base_idx]

    join_log["base_table"] = base_path.name
    join_log["y_col"] = y_col

    if y_col not in base_train.columns:
        raise RuntimeError(f"TARGET_ERROR: label column '{y_col}' not found in base table after selection.")

    # Base train split
    y_raw = base_train[y_col].copy()
    X_train = base_train.drop(columns=[y_col], errors="ignore").copy()

    # Pick base test table by max column overlap
    base_test = None
    base_test_path = None
    base_cols = set(base_train.columns)

    best_overlap = -1
    for p, df in test_tables:
        overlap = len(set(df.columns) & base_cols)
        if overlap > best_overlap:
            best_overlap = overlap
            base_test = df.copy()
            base_test_path = p.name

    if base_test is None:
        raise RuntimeError("IO_ERROR: no test tables found to match base table.")

    X_test = base_test.drop(columns=[y_col], errors="ignore").copy()
    join_log["base_test_table"] = base_test_path

    # Attach other tables
    for i, (p_tr, df_tr) in enumerate(train_tables):
        if i == base_idx:
            continue

        # Find matching test table for this training table
        df_te = None
        p_te_name = None

        guess = p_tr.name.replace("training_", "test_")
        for p2, d2 in test_tables:
            if p2.name == guess:
                df_te = d2.copy()
                p_te_name = p2.name
                break

        if df_te is None:
            tr_cols = set(df_tr.columns)
            best_o = -1
            for p2, d2 in test_tables:
                o = len(set(d2.columns) & tr_cols)
                if o > best_o:
                    best_o = o
                    df_te = d2.copy()
                    p_te_name = p2.name

        key = best_common_key(base_train, df_tr)
        if key is None:
            join_log["steps"].append({
                "train_table": p_tr.name,
                "test_table": p_te_name,
                "action": "skip",
                "reason": "no_common_key_with_base",
            })
            continue

        rel = relationship_type(base_train, df_tr, key)

        if rel == "one_to_many":
            agg_tr = aggregate_child(df_tr, key=key)
            agg_te = aggregate_child(df_te, key=key) if (df_te is not None and key in df_te.columns) else None

            X_train = X_train.merge(agg_tr, on=key, how="left")

            if agg_te is not None:
                X_test = X_test.merge(agg_te, on=key, how="left")
            else:
                for c in agg_tr.columns:
                    if c != key and c not in X_test.columns:
                        X_test[c] = np.nan

            join_log["steps"].append({
                "train_table": p_tr.name,
                "test_table": p_te_name,
                "key": key,
                "relationship": rel,
                "action": "aggregate_then_merge",
                "n_added_cols": int(agg_tr.shape[1] - 1),
            })
            continue

        if rel in ("one_to_one", "many_to_one"):
            add_cols = [c for c in df_tr.columns if c != key and c not in X_train.columns]
            merge_tr = df_tr[[key] + add_cols].copy()
            X_train = X_train.merge(merge_tr, on=key, how="left")

            if df_te is not None:
                add_cols_te = [c for c in df_te.columns if c != key and c in merge_tr.columns]
                merge_te = df_te[[key] + add_cols_te].copy()
                X_test = X_test.merge(merge_te, on=key, how="left")

            join_log["steps"].append({
                "train_table": p_tr.name,
                "test_table": p_te_name,
                "key": key,
                "relationship": rel,
                "action": "safe_left_merge",
                "n_added_cols": int(len(add_cols)),
            })
            continue

        join_log["steps"].append({
            "train_table": p_tr.name,
            "test_table": p_te_name,
            "key": key,
            "relationship": rel,
            "action": "skip",
            "reason": "many_to_many_unsafe",
        })

    # Reset indexes to keep everything consistent
    X_train = X_train.reset_index(drop=True)
    y_raw = y_raw.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, y_raw, X_test, join_log

def load_relational_task(task_folder: Path, kind: str) -> List[Tuple[Path, pd.DataFrame]]:
    """
    kind: 'training' or 'test'
    """
    pat = f"{kind}_*.csv"
    paths = sorted(task_folder.glob(pat))
    if not paths:
        raise FileNotFoundError(f"Missing files: {pat}")
    out = []
    for p in paths:
        df = read_csv(p)
        df = drop_bad_cols(df)
        out.append((p, df))
    return out

# ---------------------------
# Core evaluation
# ---------------------------
def evaluate_one_task(
    baseline_task: Path,
    evolved_task: Path,
    variant: str,
    align_mode: str,   # "stability" or "strict"
) -> Dict[str, Any]:
    t0 = now_ms()
    rec: Dict[str, Any] = {
        "task_id": baseline_task.name,
        "variant": variant,
        "align_mode": align_mode,
        "status": "unknown",
    }

    try:
        # 1) Load baseline relational tables
        train_tables = load_relational_task(baseline_task, "training")
        test_tables  = load_relational_task(baseline_task, "test")
        rec["baseline_files"] = [file_fingerprint(p) for (p, _) in (train_tables + test_tables)]

        # 2) Build baseline X/y and baseline test features
        X_train_raw, y_raw, X_base_test, join_log = build_relational_features(train_tables, test_tables)
        rec["join_log"] = join_log
        rec["y_col"] = join_log.get("y_col")
        rec["base_table"] = join_log.get("base_table")
        rec["base_test_table"] = join_log.get("base_test_table")

        # 3) Prepare y (classification/regression) + safe row filter
        y_final, task_type, y_meta, keep_mask = prepare_y_and_tasktype(y_raw)
        rec["task_type"] = task_type
        rec["y_meta"] = y_meta

        # Apply keep_mask to X_train (NUMPY MASK => never index mismatch)
        X_train = X_train_raw.iloc[keep_mask].reset_index(drop=True)
        y_train = y_final  # already filtered if regression path

        if len(y_train) < 10:
            raise RuntimeError("TARGET_ERROR: too few usable training rows after y cleaning.")

        if task_type == "classification":
            n_classes = int(pd.Series(y_train).nunique())
            if n_classes < 2:
                raise RuntimeError(f"MODEL_FIT_ERROR: only one class in y (n_classes={n_classes}).")

        rec["n_train"] = int(len(X_train))
        rec["n_base_test"] = int(len(X_base_test))

        # Schemas (preprocessing input)
        rec["baseline_train_schema"] = schema_fingerprint(X_train)
        rec["baseline_test_schema"] = schema_fingerprint(X_base_test)

        # 4) Datetime parse (learn on train, apply to base test)
        X_train, X_base_test, dt_info = parse_datetime_columns(X_train, X_base_test, threshold=0.8)
        dt_cols = list(dt_info.keys())
        rec["dt_info"] = dt_info
        rec["dt_cols"] = dt_cols

        X_train = datetime_cols_to_features(X_train, dt_cols)
        X_base_test = datetime_cols_to_features(X_base_test, dt_cols)

        X_train = normalize_string_dtypes(X_train)
        X_base_test = normalize_string_dtypes(X_base_test)

        # 5) Align base test to train
        if align_mode == "stability":
            X_base_test, align_base = align_test_to_train(X_train, X_base_test)
        else:
            X_base_test = align_test_to_train_strict(X_train, X_base_test)
            align_base = {"mode": "strict", "missing_in_test": [], "extra_in_test": []}

        rec["align_baseline_test"] = align_base
        rec["baseline_test_na_profile"] = basic_na_profile(X_base_test)

        # 6) Fit
        pipe = build_pipeline(X_train, task_type)
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            raise RuntimeError(f"MODEL_FIT_ERROR: {e}")

        # 7) Baseline predictions
        if task_type == "classification":
            proba_base = pipe.predict_proba(X_base_test)
            pred_base = np.argmax(proba_base, axis=1)
            rec["base_pred_entropy_mean"] = round(float(
                np.mean(-np.sum(proba_base * np.log(np.clip(proba_base, 0.0, 1.0)), axis=1))
            ), 2)
            rec["base_pred_maxprob_mean"] = round(float(np.mean(np.max(proba_base, axis=1))), 2)
        else:
            pred_base = pipe.predict(X_base_test)
            proba_base = None
            rec["base_pred_mean"] = round(float(np.mean(pred_base)), 2)
            rec["base_pred_std"] = round(float(np.std(pred_base)), 2)

        # 8) Build evolved test features using SAME baseline train tables + evolved test tables
        evo_test_tables = load_relational_task(evolved_task, "test")
        rec["evolved_files"] = [file_fingerprint(p) for (p, _) in evo_test_tables]

        # Only need evolved test output; train output is ignored here
        _, _, X_evo_test, evo_join_log = build_relational_features(train_tables, evo_test_tables)
        rec["evolved_join_log"] = evo_join_log
        rec["evolved_test_schema_raw"] = schema_fingerprint(X_evo_test)

        # 9) Apply datetime formats learned on baseline train
        for c, meta in dt_info.items():
            if c in X_evo_test.columns:
                fmt = meta["format"]
                X_evo_test[c] = pd.to_datetime(X_evo_test[c].astype("string"), format=fmt, errors="coerce")

        X_evo_test = datetime_cols_to_features(X_evo_test, dt_cols)
        X_evo_test = normalize_string_dtypes(X_evo_test)

        # 10) Align evolved test to baseline train
        if align_mode == "stability":
            X_evo_aligned, align_evo = align_test_to_train(X_train, X_evo_test)
        else:
            X_evo_aligned = align_test_to_train_strict(X_train, X_evo_test)
            align_evo = {"mode": "strict", "missing_in_test": [], "extra_in_test": []}

        rec["align_evolved_test"] = align_evo
        rec["evolved_test_na_profile"] = basic_na_profile(X_evo_aligned)
        rec["n_evo_test"] = int(len(X_evo_aligned))

        # 11) Predict evolved
        try:
            if task_type == "classification":
                proba_evo = pipe.predict_proba(X_evo_aligned)
                pred_evo = np.argmax(proba_evo, axis=1)
            else:
                pred_evo = pipe.predict(X_evo_aligned)
                proba_evo = None
        except Exception as e:
            raise RuntimeError(f"PREDICT_ERROR: {e}")

        # 12) Drift metrics
        # ---------------------------
        drift: Dict[str, Any] = {}

        if task_type == "classification":

            # 1) Prediction agreement
            drift["agreement"] = round(float(np.mean(pred_evo == pred_base)), 2)

            # 2) Probability divergence
            if (
                    proba_base is not None
                    and proba_evo is not None
                    and proba_base.shape[1] == proba_evo.shape[1]
            ):
                drift["jsd"] = js_divergence(proba_evo, proba_base)
            else:
                drift["jsd"] = None

        else:
            # 1) RMSE
            drift["rmse"] = round(float(np.sqrt(np.mean((pred_evo - pred_base) ** 2))), 2)

            # 2) KS statistic
            drift["ks"] = ks_statistic(pred_evo, pred_base)

            # 3) Spearman correlation
            try:
                drift["spearman"] = round(float(
                    pd.Series(pred_evo).corr(pd.Series(pred_base), method="spearman")
                ), 2)
            except Exception:
                drift["spearman"] = None

        rec.update(drift)

        # Silent failure flag
        silent = False

        if task_type == "classification":
            jsd = rec.get("jsd")
            agreement = rec.get("agreement")

            if jsd is not None and jsd > 0.05:
                silent = True
            if agreement is not None and agreement < 0.95:
                silent = True

        else:
            ks = rec.get("ks")
            rmse = rec.get("rmse")

            if ks is not None and ks > 0.15:
                silent = True
            #Silent failure if drift magnitude exceeds 20% of baseline prediction variance.
            base_std = float(np.std(pred_base))

            if base_std > 0:
                if rmse is not None and rmse / base_std > 0.2:
                    silent = True

        rec["silent_failure_candidate"] = bool(silent)

        rec["status"] = "success"
        return rec

    except FileNotFoundError as e:
        rec["status"] = "crash"
        rec["fail_type"] = FailType.IO_ERROR.value
        rec["error"] = str(e)

    except MemoryError:
        rec["status"] = "crash"
        rec["fail_type"] = FailType.RESOURCE_ERROR.value
        rec["error"] = "MemoryError"

    except RuntimeError as e:
        msg = str(e)
        rec["status"] = "crash"
        if "TARGET_ERROR" in msg:
            rec["fail_type"] = FailType.TARGET_ERROR.value
        elif "MODEL_FIT_ERROR" in msg:
            rec["fail_type"] = FailType.MODEL_FIT_ERROR.value
        elif "PREDICT_ERROR" in msg:
            rec["fail_type"] = FailType.PREDICT_ERROR.value
        elif "STRICT_SCHEMA_FAIL" in msg:
            rec["fail_type"] = FailType.PREPROCESS_ERROR.value
        elif "many-to-many" in msg.lower() or "many_to_many" in msg.lower() or "join" in msg.lower():
            rec["fail_type"] = FailType.JOIN_ERROR.value
        else:
            rec["fail_type"] = FailType.UNKNOWN_ERROR.value
        rec["error"] = msg

    except Exception as e:
        rec["status"] = "crash"
        rec["fail_type"] = FailType.UNKNOWN_ERROR.value
        rec["error"] = str(e)

    rec["traceback_tail"] = "\n".join(traceback.format_exc().splitlines()[-12:])
    return rec


# ---------------------------
# Main runner
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_root", required=True)
    ap.add_argument("--evolved_root", required=True)
    ap.add_argument("--align_mode", choices=["stability", "strict"], default="stability",
                    help="stability: fill missing cols with NaN and drop extras; strict: crash on missing cols")
    ap.add_argument("--offset", type=int, default=0, help="Start from this task index (0-based)")
    ap.add_argument("--resume", action="store_true", help="Skip tasks already in out_jsonl/out_csv")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = run all tasks, otherwise run first N tasks")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--n_jobs", type=int, default=1)
    args = ap.parse_args()

    baseline_root = Path(args.baseline_root)
    evolved_root = Path(args.evolved_root)

    task_dirs = [p for p in sorted(baseline_root.iterdir()) if p.is_dir()]

    # apply offset + limit to apply for starting folder and numbers of folders
    if args.offset > 0:
        task_dirs = task_dirs[args.offset:]
    if args.limit and args.limit > 0:
        task_dirs = task_dirs[:args.limit]

    done_ids = load_done_ids(Path(args.out_jsonl)) if args.resume else set()

    tasks = []
    for task_dir in task_dirs:
        if task_dir.name in done_ids:
            continue
        evo_dir = evolved_root / task_dir.name
        tasks.append((task_dir, evo_dir if evo_dir.exists() and evo_dir.is_dir() else None))

    run_env = env_info()

    def run_one(task_dir: Path, evo_dir: Optional[Path]) -> Dict[str, Any]:
        if evo_dir is None:
            return {
                "task_id": task_dir.name,
                "variant": args.variant,
                "align_mode": args.align_mode,
                "status": "missing_evolved_task",
                "env": run_env,
                "duration_ms": 0,
            }
        rec = evaluate_one_task(task_dir, evo_dir, args.variant, args.align_mode)
        rec["env"] = run_env
        return rec

    results = Parallel(n_jobs=args.n_jobs, prefer="processes")(
        delayed(run_one)(t, e) for (t, e) in tasks
    )

    # Write JSONL
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with open(out_jsonl, mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(safe_json(r), ensure_ascii=False) + "\n")

    # Write CSV summary
    flat = []
    for r in results:
        rr = dict(r)
        for k in [
            "baseline_train_schema", "baseline_test_schema", "evolved_test_schema_raw",
            "align_baseline_test", "align_evolved_test", "dt_info",
            "baseline_test_na_profile", "evolved_test_na_profile", "env",
            "baseline_files", "evolved_files"
        ]:
            if k in rr and isinstance(rr[k], (dict, list)):
                rr[k] = json.dumps(safe_json(rr[k]), ensure_ascii=False)
        flat.append(rr)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    print("Saved:", str(out_csv))
    print("Saved:", str(out_jsonl))


if __name__ == "__main__":
    main()