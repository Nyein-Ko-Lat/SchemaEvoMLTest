import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def read_header_only(csv_path: Path):
    # header only (fast)
    df0 = pd.read_csv(csv_path, nrows=0)
    return list(df0.columns)


def guess_domain_from_columns(cols):
    # simple keyword heuristics (edit anytime)
    text = " ".join(c.lower() for c in cols)

    rules = [
        ("weather", ["temp", "temperature", "humidity", "wind", "rain", "pressure", "station"]),
        ("sales", ["sales", "revenue", "price", "order", "customer", "product", "store", "qty", "quantity"]),
        ("health", ["patient", "diagnosis", "disease", "treatment", "blood", "cholesterol", "heart", "glucose"]),
        ("economic", ["income", "wage", "tax", "agi", "gdp", "unemployment", "inflation","budget"]),
        ("education", ["math", "reading", "science", "score", "sat", "act", "grade"]),
    ]

    hits = []
    for domain, kws in rules:
        count = sum(1 for k in kws if k in text)
        if count:
            hits.append((domain, count))

    if not hits:
        return ("unknown", 0)

    hits.sort(key=lambda x: x[1], reverse=True)
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data/original", help="Root directory containing task folders")
    ap.add_argument("--out", default="./data/metadata/schema_catalog.csv", help="Output root for evolved datasets")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of folders (0=all)")
    args = ap.parse_args()

    root = Path(args.root)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    task_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    if args.limit and args.limit > 0:
        task_folders = task_folders[:args.limit]

    rows = []
    for i, folder in enumerate(task_folders, 1):
        task_id = folder.name

        for f in sorted(folder.glob("*.csv")):
            try:
                cols = read_header_only(f)
                domain, score = guess_domain_from_columns(cols)

                rows.append({
                    "task_id": task_id,
                    "folder": str(folder),
                    "file": f.name,
                    "path": str(f),
                    "n_cols": len(cols),
                    "columns_json": json.dumps(cols),
                    "domain_guess": domain,
                    "domain_score": score,
                })
            except Exception as e:
                rows.append({
                    "task_id": task_id,
                    "folder": str(folder),
                    "file": f.name,
                    "path": str(f),
                    "n_cols": "",
                    "columns_json": "",
                    "domain_guess": "error",
                    "domain_score": 0,
                    "error": repr(e),
                })

        if i % 50 == 0:
            print(f"Scanned {i}/{len(task_folders)} folders...")

    # write catalog
    fieldnames = [
        "task_id", "folder", "file", "path",
        "n_cols", "columns_json", "domain_guess", "domain_score"
    ]
    # include error column only if any
    if any("error" in r for r in rows):
        fieldnames.append("error")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("Catalog written:", out_csv)


if __name__ == "__main__":
    main()
