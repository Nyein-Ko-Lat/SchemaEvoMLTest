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
#requirement

```md
Tested with Python 3.9+
```
Install dependencies using:

```bash
pip install -r requirements.txt
```
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
```
### 2. Run ML evaluation
```bash
python schema_evolve_ml_testing.py --baseline_root C:/ThesisData/output/evolved_tasks/baseline --evolved_root C:/ThesisData/output/evolved_tasks/baseline --variant baseline --align_mode stability --limit 100 --offset 0 --resume --out_csv C:/ThesisData/results/baseline_stability.csv --out_jsonl C:/ThesisData/results/baseline_stability.jsonl --n_jobs 2
```
### 3. Aggregate results
```bash
python analyze_results.py --inputs C:/ThesisData/results/*.jsonl --out_dir C:/ThesisData/results/analysis
```
---
#Expected Folder Structure
```bash
C:/ThesisData/
├── original/
├── output/
│   └── evolved_tasks/
├── results/
│   ├── *.csv
│   ├── *.jsonl
│   └── analysis/
```
---
## Dataset Availability

This study uses the Auto-Pipeline benchmark dataset introduced by Yang et al. (VLDB 2021), which contains approximately 700 real-world data preparation pipelines.

The dataset is publicly available at:
https://gitlab.com/jwjwyoung/autopipeline-benchmarks/-/tree/main/github-pipelines

**The dataset is not redistributed in this repository. Users should obtain it from the original source and place it in the expected input directory before running the experiments.**

---
## Sample Files Included

This repository includes a small set of representative sample CSV files and summary outputs to illustrate:
- baseline input structure,
- schema-evolved examples,
- and the format of analysis-ready results.

These files are included for demonstration and reproducibility support only. They are not the full benchmark dataset or the full generated experiment outputs.

---
## Reproducibility Notes
To reproduce the full experiment, users should:
1. obtain the original benchmark dataset from its official source,
2. place the dataset under the expected input directory,
3. generate evolved task variants,
4. run the ML evaluation scripts,
5. aggregate the resulting logs and outputs.

The repository is intended as a research codebase supporting the experimental findings reported in the thesis.

---
## Outputs
Typical outputs produced by the experiments include:

- task-level CSV summaries,
- JSONL execution logs,
- aggregated summary tables,
- and analysis-ready result files.

These outputs are used to evaluate:

- execution failures,
- silent failures,
- and prediction drift under schema evolution.

---
## Notes
This repository is provided as research code developed for controlled experimentation and thesis reproducibility. The implementation prioritizes experimental correctness and traceability over production-level packaging

---
## Citation
If you use or reference this repository, please cite the corresponding thesis.

## Author
Nyein Ko Lat
GISMA University of Applied Science
M.ENG.Computer Science
