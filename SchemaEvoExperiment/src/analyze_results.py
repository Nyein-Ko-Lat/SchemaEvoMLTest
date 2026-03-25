"""
analyze_results.py

Analyze JSONL logs produced by runner (one record per task).
- Loads one or many *.jsonl files
- Normalizes fields (variant/status/fail_type)
- Exports:
  * combined_flat.csv  (one row per task)
  * summary_by_variant.csv
  * failures_by_variant.csv
  * (optional) simple plots as PNGs

Usage examples:
  python analyze_results.py --inputs C:/ThesisData/results/drop.jsonl
  python analyze_results.py --inputs C:/ThesisData/results/*.jsonl --out_dir C:/ThesisData/results/analysis
  python analyze_results.py --inputs C:/ThesisData/results/*.jsonl --variants add drop rename --save_plots
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import re

# -------------------------
# Loading
# -------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore bad line
                continue
    return rows


def load_inputs(patterns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    files: List[str] = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(matched)
        else:
            # allow direct file path
            if Path(p).exists():
                files.append(p)

    files = sorted(list(dict.fromkeys(files)))  # unique preserve order
    if not files:
        raise FileNotFoundError("No JSONL files matched --inputs")

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        all_rows.extend(read_jsonl(Path(fp)))

    df = pd.DataFrame(all_rows)
    return df, files


# -------------------------
# Flattening helpers
# -------------------------
NESTED_JSON_COLS = [
    "baseline_train_schema",
    "baseline_test_schema",
    "evolved_test_schema_raw",
    "align_baseline_test",
    "align_evolved_test",
    "dt_info",
    "baseline_test_na_profile",
    "evolved_test_na_profile",
    "env",
    "baseline_files",
    "evolved_files",
]


def safe_json_dumps(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)

def _text(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def classify_crash_row(row: pd.Series) -> Dict[str, str]:
    """
    Classify crash root cause more accurately using error/traceback text.
    Returns: {"fail_bucket": "...", "fail_reason": "..."}.
    """
    fail_type = _text(row.get("fail_type")).upper()
    err = _text(row.get("error"))
    tb = _text(row.get("traceback_tail"))
    txt = (fail_type + " " + err + " " + tb).lower()

    # ---- IO / missing files ----
    if "filenotfounderror" in txt or "missing files" in txt or "no such file" in txt:
        return {"fail_bucket": "IO", "fail_reason": "MISSING_FILE"}

    # ---- Resource / memory ----
    if "memoryerror" in txt or "std::bad_alloc" in txt or "cannot allocate memory" in txt:
        # Often caused by dense OHE or giant join
        if "onehot" in txt or "sparse_output=false" in txt or "dense" in txt:
            return {"fail_bucket": "Resource", "fail_reason": "OHE_DENSE_OOM"}
        if "merge" in txt or "join" in txt:
            return {"fail_bucket": "Resource", "fail_reason": "JOIN_OOM"}
        return {"fail_bucket": "Resource", "fail_reason": "OOM"}

    # ---- Join failures ----
    if "no common columns" in txt:
        return {"fail_bucket": "Join", "fail_reason": "NO_COMMON_COLUMNS"}
    if "many-to-many" in txt:
        return {"fail_bucket": "Join", "fail_reason": "MANY_TO_MANY_RISK"}
    if "join expansion" in txt or "cartesian" in txt:
        return {"fail_bucket": "Join", "fail_reason": "JOIN_EXPLOSION"}

    # ---- Target / label inference ----
    if "no suitable target" in txt or "no suitable" in txt and "target" in txt:
        return {"fail_bucket": "Target", "fail_reason": "NO_TARGET"}
    if "too few usable training rows" in txt or "after cleaning y" in txt:
        return {"fail_bucket": "Target", "fail_reason": "Y_TOO_FEW_ROWS"}
    if "keyerror" in txt and ("y_col" in txt or "label" in txt or "target" in txt):
        return {"fail_bucket": "Target", "fail_reason": "LABEL_COLUMN_MISSING"}

    # ---- Schema evolution / preprocessing ----
    if "strict_schema_fail" in txt:
        return {"fail_bucket": "Schema", "fail_reason": "STRICT_SCHEMA_FAIL"}
    if "missing_in_test" in txt and "strict_schema_fail" in txt:
        return {"fail_bucket": "Schema", "fail_reason": "MISSING_COLUMNS"}
    if "could not convert" in txt or "cannot convert" in txt:
        return {"fail_bucket": "Schema", "fail_reason": "TYPE_CONVERSION_ERROR"}
    if "unknown categories" in txt or "handle_unknown" in txt:
        return {"fail_bucket": "Schema", "fail_reason": "OHE_CATEGORY_ISSUE"}
    if "to_datetime" in txt or "datetime" in txt:
        return {"fail_bucket": "Schema", "fail_reason": "DATETIME_PARSE_ISSUE"}

    # ---- Model fit / predict ----
    if "convergencewarning" in txt or "failed to converge" in txt or "lbfgs" in txt or "saga" in txt:
        return {"fail_bucket": "Model", "fail_reason": "CONVERGENCE"}
    if "model fit failed" in txt or "fit failed" in txt:
        return {"fail_bucket": "Model", "fail_reason": "FIT_ERROR"}
    if "predict failed" in txt:
        return {"fail_bucket": "Model", "fail_reason": "PREDICT_ERROR"}

    # ---- Fallback to existing fail_type ----
    if "target" in fail_type:
        return {"fail_bucket": "Target", "fail_reason": fail_type}
    if "join" in fail_type:
        return {"fail_bucket": "Join", "fail_reason": fail_type}
    if "preprocess" in fail_type:
        return {"fail_bucket": "Schema", "fail_reason": fail_type}
    if "model_fit" in fail_type or "predict" in fail_type:
        return {"fail_bucket": "Model", "fail_reason": fail_type}
    if "resource" in fail_type:
        return {"fail_bucket": "Resource", "fail_reason": fail_type}
    if "io" in fail_type:
        return {"fail_bucket": "IO", "fail_reason": fail_type}

    return {"fail_bucket": "Unknown", "fail_reason": "UNKNOWN"}

def crash_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (variant, align_mode):
      - raw counts
      - baseline counts repeated in every row (for same align_mode)
      - deltas vs baseline (counts + rates)
    """
    g = df.groupby(["variant", "align_mode"], dropna=False)

    out = pd.DataFrame({
        "n_tasks": g.size(),
        "n_success": g.apply(lambda x: (x["status"] == "success").sum()),
        "n_crash": g.apply(lambda x: (x["status"] == "crash").sum()),
        "n_missing_evolved": g.apply(lambda x: (x["status"] == "missing_evolved_task").sum()),
    }).reset_index()

    # rates (optional, but keep them)
    out["crash_rate"] = (out["n_crash"] / out["n_tasks"].replace(0, np.nan)).round(2)
    out["success_rate"] = (out["n_success"] / out["n_tasks"].replace(0, np.nan)).round(2)

    # ---- baseline stats per align_mode (repeat into all rows) ----
    base = out[out["variant"] == "baseline"].copy()
    base = base.rename(columns={
        "n_tasks": "baseline_n_tasks",
        "n_success": "baseline_n_success",
        "n_crash": "baseline_n_crash",
        "n_missing_evolved": "baseline_n_missing_evolved",
        "crash_rate": "baseline_crash_rate",
        "success_rate": "baseline_success_rate",
    })[[
        "align_mode",
        "baseline_n_tasks", "baseline_n_success", "baseline_n_crash", "baseline_n_missing_evolved",
        "baseline_crash_rate", "baseline_success_rate"
    ]]

    out = out.merge(base, on="align_mode", how="left")

    # ---- deltas vs baseline ----
    out["delta_crash_count"] = out["n_crash"] - out["baseline_n_crash"]
    out["delta_success_count"] = out["n_success"] - out["baseline_n_success"]
    out["delta_crash_rate"] = out["crash_rate"] - out["baseline_crash_rate"]

    return out.sort_values(["align_mode", "delta_crash_count", "n_tasks"], ascending=[True, False, False])


def silent_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Silent failures are only meaningful when the pipeline didn't crash.
    So we compute them among successes only.
    Requires a boolean column: silent_failure_candidate.
    """
    dd = df.copy()
    if "silent_failure_candidate" not in dd.columns:
        dd["silent_failure_candidate"] = False

    succ = dd[dd["status"] == "success"].copy()
    if succ.empty:
        return pd.DataFrame(columns=[
            "variant", "align_mode", "n_success", "n_silent", "silent_rate"
        ])

    g = succ.groupby(["variant", "align_mode"], dropna=False)
    out = pd.DataFrame({
        "n_success": g.size(),
        "n_silent": g.apply(lambda x: (x["silent_failure_candidate"] == True).sum()),
    }).reset_index()

    out["silent_rate"] = (out["n_silent"] / out["n_success"].replace(0, np.nan)).round(2)

    # ---- baseline repeated in each row ----
    base = out[out["variant"] == "baseline"].copy()
    base = base.rename(columns={
        "n_success": "baseline_n_success",
        "n_silent": "baseline_n_silent",
        "silent_rate": "baseline_silent_rate",
    })[["align_mode", "baseline_n_success", "baseline_n_silent", "baseline_silent_rate"]]

    out = out.merge(base, on="align_mode", how="left")

    # deltas
    out["delta_silent_count"] = out["n_silent"] - out["baseline_n_silent"]
    out["delta_silent_rate"] = out["silent_rate"] - out["baseline_silent_rate"]

    return out.sort_values(["align_mode", "delta_silent_count"], ascending=[True, False])

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure key columns exist
    for col in ["task_id", "variant", "status", "fail_type"]:
        if col not in df.columns:
            df[col] = None

    # Standardize status
    df["status"] = df["status"].astype(str).str.lower()
    df.loc[df["status"].isin(["ok"]), "status"] = "success"
    df.loc[df["status"].isin(["missing_evolved_task"]), "status"] = "missing_evolved_task"

    # Make numeric columns numeric
    numeric_cols = [
        "n_evo_test",
        "n_base_test",
        "n_train",
        # regression drift
        "rmse",
        "ks",
        "spearman",
        # classification drift
        "agreement",
        "jsd",
        # optional reference distribution stats
        "base_pred_mean",
        "base_pred_std",
        "base_pred_entropy_mean",
        "base_pred_maxprob_mean",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------
    # Normalized RMSE (scale-free)
    # --------------------------------
    if "rmse" in df.columns and "base_pred_std" in df.columns:
        df["nrmse"] = (df["rmse"] / df["base_pred_std"]).round(2)
        # avoid divide-by-zero explosions
        df.loc[df["base_pred_std"] == 0, "nrmse"] = np.nan

    # Flatten nested objects to strings so CSV export is stable
    for c in NESTED_JSON_COLS:
        if c in df.columns:
            df[c] = df[c].map(safe_json_dumps)

    # Add refined crash classification columns (analysis-side taxonomy)
    if "error" not in df.columns:
        df["error"] = None
    if "traceback_tail" not in df.columns:
        df["traceback_tail"] = None

    # Default
    df["fail_bucket"] = None
    df["fail_reason"] = None

    crash_mask = df["status"] == "crash"
    if crash_mask.any():
        classified = df.loc[crash_mask].apply(classify_crash_row, axis=1, result_type="expand")
        df.loc[crash_mask, "fail_bucket"] = classified["fail_bucket"].values
        df.loc[crash_mask, "fail_reason"] = classified["fail_reason"].values

    return df

# --------------------------------
# Remove tasks that are not compatible with supervised ML
# --------------------------------
def keep_only_baseline_valid_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only task_ids whose baseline run succeeded within each align_mode.
    This preserves fair comparison separately for strict and stability analyses.
    """
    dd = df.copy()

    required = {"task_id", "variant", "status", "align_mode"}
    missing = required - set(dd.columns)
    if missing:
        raise ValueError(f"Missing required columns for baseline filtering: {sorted(missing)}")

    dd["task_id"] = dd["task_id"].astype(str)

    baseline_success = (
        dd[
            (dd["variant"] == "baseline") &
            (dd["status"] == "success") &
            (dd["task_id"].notna())
        ][["task_id", "align_mode"]]
        .drop_duplicates()
        .copy()
    )

    dd = dd.merge(
        baseline_success.assign(_keep=1),
        on=["task_id", "align_mode"],
        how="inner"
    )

    dd = dd.drop(columns=["_keep"])
    return dd

# -------------------------
# Summaries
# -------------------------
def crash_rate_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["variant","align_mode"], dropna=False)
    out = pd.DataFrame(
        {
            "n_tasks": g.size(),
            "n_success": g.apply(lambda x: (x["status"] == "success").sum()),
            "n_crash": g.apply(lambda x: (x["status"] == "crash").sum()),
            "n_missing_evolved": g.apply(lambda x: (x["status"] == "missing_evolved_task").sum()),
            "crash_rate": g.apply(lambda x: float((x["status"] == "crash").mean())),
            "success_rate": g.apply(lambda x: float((x["status"] == "success").mean())),
        }
    )
    out = out.reset_index()
    return out.sort_values(["crash_rate", "n_tasks"], ascending=[False, False])


def failure_taxonomy_table(df: pd.DataFrame) -> pd.DataFrame:
    crashes = df[df["status"] == "crash"].copy()
    if crashes.empty:
        return pd.DataFrame(columns=["variant", "fail_bucket", "fail_reason", "count", "share_within_variant"])

    # Prefer refined analysis-side taxonomy; fallback to fail_type
    if "fail_bucket" not in crashes.columns:
        crashes["fail_bucket"] = None
    if "fail_reason" not in crashes.columns:
        crashes["fail_reason"] = crashes.get("fail_type", None)

    grp = (
        crashes
        .groupby(["variant","align_mode", "fail_bucket", "fail_reason"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    totals = crashes.groupby(["variant","align_mode"]).size().reset_index(name="variant_crashes")
    merged = grp.merge(totals, on=["variant","align_mode"], how="left")
    merged["share_within_variant"] = (merged["count"] / merged["variant_crashes"].replace(0, np.nan)).round(2)

    return merged.sort_values(["variant", "count"], ascending=[True, False])


def drift_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    succ = df[df["status"] == "success"].copy()
    if succ.empty:
        return pd.DataFrame()

    reg_metrics = ["nrmse", "ks", "spearman"]
    cls_metrics = ["agreement", "jsd"]
    metrics = [m for m in reg_metrics + cls_metrics if m in succ.columns]

    def agg_block(g: pd.DataFrame) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for m in metrics:
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            if len(v) == 0:
                res[f"{m}_median"] = np.nan
                res[f"{m}_p90"] = np.nan
                res[f"{m}_mean"] = np.nan
            else:
                res[f"{m}_median"] = round(float(v.median()), 2)
                res[f"{m}_p90"] = round(float(v.quantile(0.90)), 2)
                res[f"{m}_mean"] = round(float(v.mean()), 2)

        return res

    out = succ.groupby(["variant", "align_mode"], dropna=False).apply(agg_block).apply(pd.Series).reset_index()
    return out

# -------------------------
# Optional plots (simple, no seaborn)
# -------------------------
def save_plots(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Crash rate bar
    cr = crash_rate_table(df)
    plt.figure()
    plt.bar(cr["variant"].astype(str), cr["crash_rate"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Crash rate")
    plt.title("Crash rate by variant")
    plt.tight_layout()
    plt.savefig(out_dir / "crash_rate_by_variant.png", dpi=150)
    plt.close()

    # Drift metric distributions for successes (if available)
    succ = df[df["status"] == "success"].copy()
    if succ.empty:
        return

    for metric in ["nrmse", "ks", "spearman", "agreement", "jsd"]:
        if metric not in succ.columns:
            continue
        plt.figure()
        # show per-variant boxplot
        variants = [v for v in sorted(succ["variant"].dropna().unique())]
        data = [pd.to_numeric(succ[succ["variant"] == v][metric], errors="coerce").dropna().values for v in variants]
        if not any(len(x) for x in data):
            plt.close()
            continue
        plt.boxplot(data, labels=[str(v) for v in variants], showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} (successes) by variant")
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_by_variant.png", dpi=150)
        plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more JSONL paths or glob patterns")
    ap.add_argument("--out_dir", default="./analysis_out", help="Output directory for tables/CSVs")
    ap.add_argument("--variants", nargs="*", default=None, help="Optional: only include these variants")
    ap.add_argument("--save_plots", action="store_true", help="Save a few basic PNG plots")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw, files = load_inputs(args.inputs)
    df = normalize_df(df_raw)
    df = keep_only_baseline_valid_tasks(df)

    n_remaining_tasks = df["task_id"].nunique() if "task_id" in df.columns else len(df)
    print(f"Baseline-valid tasks retained for analysis: {n_remaining_tasks}")


    # Optional filter by variants
    if args.variants:
        wanted = set(args.variants)
        df = df[df["variant"].isin(wanted)].copy()

    # Save combined flat CSV
    combined_csv = out_dir / "combined_flat.csv"

    cols = df.columns.tolist()


    df.to_csv(combined_csv, index=False)

    # Summaries
    crash_tbl = crash_rate_table(df)
    fail_tbl = failure_taxonomy_table(df)
    drift_tbl = drift_summary_table(df)
    silent_tbl = silent_summary_table(df)
    crash_bucket_tbl = (
        df[df["status"] == "crash"]
        .groupby(["variant","align_mode", "fail_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["variant", "count"], ascending=[True, False])
    )
    # round decimal to 2 places in each table
    for tbl in [crash_tbl, drift_tbl, prof_tbl, silent_tbl]:
        for c in tbl.select_dtypes(include=["float64", "float32"]).columns:
            tbl[c] = tbl[c].round(2)

    crash_tbl.to_csv(out_dir / "crash_rate_by_variant.csv", index=False)
    fail_tbl.to_csv(out_dir / "failures_by_variant.csv", index=False)
    drift_tbl.to_csv(out_dir / "drift_summary_by_variant.csv", index=False)
    crash_bucket_tbl.to_csv(out_dir / "crash_buckets_by_variant.csv", index=False)
    silent_tbl.to_csv(out_dir / "silent_summary_by_variant.csv", index=False)

    # Print a quick console summary
    print("Loaded JSONL files:")
    for f in files:
        print(" -", f)
    print("\nSaved:")
    print(" -", combined_csv)
    print(" -", out_dir / "crash_rate_by_variant.csv")
    print(" -", out_dir / "failures_by_variant.csv")
    if not drift_tbl.empty:
        print(" -", out_dir / "drift_summary_by_variant.csv")


    if args.save_plots:
        save_plots(df, out_dir)
        print("\nSaved plots to:", str(out_dir))

if __name__ == "__main__":
    main()