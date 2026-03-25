# SchemaEvoMLTest

This repository contains the implementation code used in the thesis:

**Evaluating the Effects of Schema Evolution on Machine Learning Pipeline Stability**

The project investigates how schema evolution affects machine learning (ML) pipelines under controlled conditions. The experiments focus on both:

- **operational stability** (e.g., crash vs. non-crash execution), and
- **prediction stability** (e.g., silent output drift after schema changes).

The study applies schema evolution operations to benchmark datasets, evaluates baseline and evolved conditions, and aggregates the resulting execution and prediction-level outcomes.

---

## Repository Overview

The repository contains the main scripts used for:

- generating schema-evolved datasets,
- running ML evaluation on baseline and evolved data,
- and aggregating experiment results.

### Main Scripts

- `evolve_autopipeline_patched.py`  
  Generates schema-evolved task variants from the original benchmark dataset.

- `schema_evolve_ml_testing.py`  
  Runs ML evaluation on baseline and evolved tasks under selected alignment modes.

- `analyze_results.py`  
  Aggregates JSONL experiment outputs and exports summary result tables.

---
## Dataset Availability

This study uses the Auto-Pipeline benchmark dataset introduced by Yang et al. (VLDB 2021), which contains approximately 700 real-world data preparation pipelines.

The dataset is publicly available at:
https://gitlab.com/jwjwyoung/autopipeline-benchmarks/-/tree/main/github-pipelines

The dataset is not redistributed in this repository. Users should obtain it from the original source and place it in the expected input directory before running the experiments.

---
## Experimental Workflow

The experiment pipeline follows three main stages:

1. **Schema Evolution Generation**  
   Original benchmark tasks are transformed using controlled schema evolution operations.

2. **ML Evaluation**  
   A baseline model is trained and evaluated against baseline or schema-evolved task variants.

3. **Result Aggregation and Analysis**  
   Task-level logs are combined and summarized into analysis-ready outputs.

---

## Example Commands Used in the Thesis

### 1. Generate schema-evolved tasks

```bash
python evolve_autopipeline_patched.py --root C:/ThesisData/original --out C:/ThesisData/output/evolved_tasks --variants meaning --limit 100 --offset 0 --seed 2 --resume
