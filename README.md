# PUBG Data Analytics & Machine Learning

A comprehensive PySpark ML study on PUBG player behavior — predicting placement, classifying Top 25% finishes, and discovering playstyle archetypes from ~4.45 million player records.

**Course:** DS3003 Data Science Project, Spring 2026  
**Group 13:** Zhengyong Huang, Zhen Liu, Yihan Wei, Zewang Wang

## Overview

This project builds a full PySpark ML pipeline covering:

| Task | Models | Key Metric |
|---|---|---|
| **Regression** — Predict `winPlacePerc` | Linear Regression, GBT Regressor | R² up to 0.893 (duo) |
| **Classification** — Identify Top 25% placement | Logistic Regression, GBT Classifier | ROC-AUC up to 0.967 (duo) |
| **Clustering** — Discover playstyle archetypes | K-Means (K=5) + PCA visualization | Silhouette score |

## Pipeline

```
Raw Data (4.45M rows)
  → Data Cleaning (severity-based anomaly handling)
    → Feature Engineering (mobility, combat, resource, social features)
      → Mode Split (solo / duo / squad)
        → Train-Test Split (by matchId to prevent leakage)
          → Regression → Feature Importance
          → Classification → ROC-AUC
          → Clustering → Cluster Profiles + PCA
```

## Project Structure

```
├── pubg_ml_pipeline.ipynb    # Full PySpark ML pipeline (89 cells, cleaned outputs)
├── PUBG_Report.docx          # Final project report (PDF/Word)
├── outputs/                  # Generated results (CSV tables, charts)
│   ├── regression_results.csv
│   ├── classification_results.csv
│   ├── kmeans_silhouette_scores.csv
│   ├── *_cluster_profile.csv
│   ├── *_feature_importance.csv
│   └── ablation_study.png
└── requirements.txt          # Python dependencies
```

## Key Findings

- **walkDistance** is the dominant predictor of placement across all modes
- **Boosts** carry significantly stronger placement signal than heals
- Five distinct playstyle archetypes discovered: elite fighters, vehicle rotators, balanced survivors, low-activity players, and support-oriented players
- **Tree-based models (GBT)** substantially outperform linear baselines across all tasks
- Match-level train-test split is critical — row-level splitting causes data leakage

## Getting Started

### Prerequisites

- Python 3.8+
- Java 8+ (for PySpark)
- 8GB+ RAM recommended

### Install

```bash
pip install -r requirements.txt
```

### Data

The main dataset (`pubg_main_final.csv`, ~445MB) is sourced from [Kaggle PUBG Match Statistics](https://www.kaggle.com/c/pubg-finish-placement-prediction). Place it in the project root before running the notebook.

### Run

```bash
jupyter notebook pubg_ml_pipeline.ipynb
```

Set `USE_SAMPLE_FOR_DEVELOPMENT = True` in the notebook for quick testing on a reduced dataset.
