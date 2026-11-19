<div align="center">

# 🧬 Machine Learning for Long-Acting Injectable Drug Release

**A reproducible analysis of PLGA-based drug-release prediction**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Focus](https://img.shields.io/badge/Polymer-PLGA-2E8B57?style=flat-square)
![Methods](https://img.shields.io/badge/Methods-ML%20%7C%20SHAP-6C63FF?style=flat-square)

</div>

---

## Overview

This project explores machine-learning approaches for predicting drug release from long-acting injectable (LAI) formulations, with a focus on PLGA. It contains a baseline evaluation pipeline and an extended analysis that adds hyperparameter tuning and SHAP-based interpretation.

The analysis compares zero-shot (14-feature) and few-shot (17-feature) datasets using grouped cross-validation and held-out formulation groups. The scripts generate model metrics, figures, and Excel workbooks.

> **Research context:** This is a research analysis. Predictions are not intended for clinical or regulatory decision-making.

## Project map

```text
GITHUB DATASET/
├── README.md
├── data/
│   ├── baseline/                 # Inputs used by the baseline pipeline
│   └── improved/                 # Inputs used by the extended pipeline
├── scripts/
│   ├── baseline/
│   │   └── LAI_Complete_Pipeline.py
│   └── improved/
│       └── Polymer_Release_timing.py
└── results/
    ├── baseline/                 # Figures and LAI_ML_Full_Results.xlsx
    └── improved/                 # Figures and LAI_Improved_Results.xlsx
```

Each analysis has its own copy of the input workbooks. The copies were retained to make each run's inputs explicit; they were verified as identical by SHA-256.

## Analysis versions

| Version | Script | What it does | Main outputs |
|---|---|---|---|
| **Baseline** | [`scripts/baseline/LAI_Complete_Pipeline.py`](scripts/baseline/LAI_Complete_Pipeline.py) | Compares five regression models, performs grouped cross-validation and external validation, and reports feature importance. | Six figures and [`LAI_ML_Full_Results.xlsx`](results/baseline/LAI_ML_Full_Results.xlsx) |
| **Extended** | [`scripts/improved/Polymer_Release_timing.py`](scripts/improved/Polymer_Release_timing.py) | Adds randomized hyperparameter search and SHAP explanations for tree-based models. | Figures 7–12 and [`LAI_Improved_Results.xlsx`](results/improved/LAI_Improved_Results.xlsx) |

### Figures at a glance

**Baseline:** cross-validation performance, release curves, R² heatmap, feature importance, zero-shot vs few-shot comparison, and best-model scatter plot.

**Extended:** hyperparameter tuning, selected parameters, SHAP beeswarm and bar plots, SHAP dependence plots for both feature sets, and SHAP summaries by test group.

## Data

The analysis uses two Excel workbooks per version:

- `Dataset_14_feat.xlsx` — 14-feature (zero-shot) input.
- `Dataset_17_feat.xlsx` — 17-feature (few-shot) input, including early-time release features.

The scripts select PLGA formulation groups and use the training and held-out groups defined in the source. Data files are preserved as supplied; the pipelines clip the release target to the physical range `[0, 1]` during preparation.

The project scripts cite the source dataset repository and the research paper by Bannigan et al. (2023). See the script headers for the DOI and dataset attribution.

## Setup

Create and activate a Python environment, then install the packages used by the analyses:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install numpy pandas matplotlib scikit-learn openpyxl shap lightgbm
```

The baseline script can fall back to scikit-learn's `GradientBoostingRegressor` if LightGBM is unavailable. The extended analysis uses SHAP and LightGBM-related components; install the dependencies above for the full workflow.

## Run

Run a script from any working directory; input and output paths are anchored to the project structure.

```bash
python scripts/baseline/LAI_Complete_Pipeline.py
python scripts/improved/Polymer_Release_timing.py
```

Generated artifacts are written to the matching `results/` subfolder. Running a pipeline updates the results it produces.

## Validation design

- **Model families:** LightGBM (or its documented fallback), Random Forest, Extra Trees, Support Vector Regression, and Ridge Regression.
- **Cross-validation:** five-fold `GroupKFold`, grouping by experimental index to reduce formulation leakage between folds.
- **External evaluation:** held-out formulation groups are specified in each pipeline.
- **Interpretability:** the extended pipeline uses SHAP TreeExplainer for tree-based models.

Consult the exported workbooks for detailed metrics, predictions, feature rankings, and methodology notes.

## Citation

The scripts identify the project reference as Bannigan et al. (2023), *Nature Communications*, DOI: [10.1038/s41467-022-35343-w](https://doi.org/10.1038/s41467-022-35343-w). Dataset source: [aspuru-guzik-group/long-acting-injectables](https://github.com/aspuru-guzik-group/long-acting-injectables).

Please cite the original study and dataset when reusing these materials, and verify the final bibliographic details against the publication.

---

<div align="center">

**Polymer science × interpretable machine learning**

</div>