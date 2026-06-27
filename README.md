# 🛡️ RiskLens
### Calibrated Default Probability Scoring with Business-Driven Decision Thresholds

> *Most credit models predict. RiskLens decides.*

A production-grade credit risk decisioning system that transforms raw applicant data into calibrated default probabilities, cost-optimized lending thresholds, risk-banded decisions, and feature-level explanations — built to reflect how credit risk actually works in practice, not just how it's taught.

## 🚀 Live Demo

Try RiskLens here: **[https://risklens1.streamlit.app](https://risklens1.streamlit.app/)**

![Streamlit Home Screen](assets/images/streamlit_home.png)

---

## The Problem With Standard Credit ML

Most loan default classifiers are built to maximize accuracy. In practice, that's the wrong objective.

Approving a future defaulter costs a lender far more than rejecting a creditworthy applicant. These costs are not equal — so the decision threshold should not be 0.50.

RiskLens treats threshold selection as a **business decision**, not a statistical default.

---

## What RiskLens Does

```
Raw Applicant Data (255,347 records, 18 features)
        ↓
  Preprocessing Pipeline
  (imputation, encoding, scaling)
        ↓
  Gradient Boosting Classifier
  (selected from 4-model comparison)
        ↓
  Probability Calibration
  (Brier Score: 0.0909)
        ↓
  Business Cost Optimization
  (FN cost: $1,000 | FP cost: $200)
        ↓
  Optimal Threshold: 0.15
  (vs. industry default of 0.50)
        ↓
  Risk Band Assignment
  Low → Moderate → High → Critical
        ↓
  Lending Decision + SHAP Explanation
  Approve | Manual Review | Reject
```

Every stage is deliberate. Every output is explainable.

---

## Results

### Model Comparison

| Model | Accuracy | ROC-AUC | PR-AUC | Recall | Precision |
|---|---|---|---|---|---|
| Logistic Regression | 0.676 | 0.753 | 0.311 | 0.699 | 0.220 |
| Random Forest | 0.885 | 0.744 | 0.304 | 0.012 | 0.710 |
| Extra Trees | 0.884 | 0.731 | 0.280 | 0.005 | 0.574 |
| **Gradient Boosting** ✅ | **0.886** | **0.758** | **0.329** | 0.049 | 0.626 |

> Gradient Boosting selected on ROC-AUC and PR-AUC — the two metrics that matter for imbalanced classification in credit risk. Note: Random Forest and Extra Trees achieve 88%+ accuracy while catching under 2% of actual defaulters. That is a majority-class predictor, not a credit risk model. Evaluation went beyond accuracy precisely because of this.

### Why Gradient Boosting at Default Threshold Still Fails

At threshold 0.50, Gradient Boosting achieves 88.6% accuracy — but catches only **4.9% of actual defaulters**. Moving to the business-optimal threshold changes everything.

### The Threshold That Actually Works

![Threshold Optimization Plot](assets/images/threshold.png)

| Threshold | Recall | Precision | Total Business Cost |
|---|---|---|---|
| 0.10 | 74.7% | 20.9% | $4,861,000 |
| **0.15** ✅ | **56.6%** | **27.2%** | **$4,374,000** |
| 0.20 | 42.0% | 32.7% | $4,465,000 |
| 0.25 | 30.6% | 37.9% | $4,711,000 |
| 0.50 | 4.9% | 62.6% | $5,673,000 |

At threshold **0.15**, RiskLens catches **56.6% of defaulters** while minimizing total expected business cost at $4,374,000 — saving **$1,299,000** vs. the default 0.50 threshold, which misses over 95% of defaulters despite looking accurate on paper.

**Post-threshold performance:** Accuracy 77.4% · Recall 56.6% · Precision 27.2% · F1 0.367 · Brier Score 0.0909

### Risk Band Validation

| Risk Band | Observed Default Rate |
|---|---|
| Low Risk | 2.78% |
| Moderate Risk | 8.37% |
| High Risk | 22.01% |
| Critical Risk | 42.94% |

Critical Risk applicants default at **15.4x the rate** of Low Risk applicants. The segmentation reflects real separability in borrower risk — not arbitrary cutoffs.

---

## Pipeline Architecture

```
risklens/
├── notebooks/
│   ├── 01_data_understanding.ipynb                      # EDA, distributions, class imbalance (255,347 records)
│   ├── 02_data_preparation_feature_engineering.ipynb   # Preprocessing pipeline, encoding, scaling
│   ├── 03_baseline_model_development.ipynb             # 4-model comparison, ROC/PR-AUC evaluation
│   ├── 04_calibration_threshold_optimization.ipynb     # Calibration curve, Brier score, cost sweep
│   ├── 05_risk_segmentation_decision_engine.ipynb      # Risk bands, decision engine, portfolio simulation
│   └── 06_model_explainability_shap.ipynb              # Global SHAP, waterfall plots, per-applicant drivers
├── src/
│   └── decision_functions.py                           # Reusable risk band + decision logic
├── models/
│   ├── best_baseline_model.pkl
│   ├── preprocessor.pkl
│   └── best_business_threshold.pkl
├── apps/
│   └── streamlit_app.py                                # Live decisioning interface with SHAP waterfall
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Key Technical Decisions

**Why PR-AUC alongside ROC-AUC?**
ROC-AUC is insensitive to class imbalance. With an 11.6% default rate across 255,347 records, a model predicting the majority class looks strong on ROC-AUC. PR-AUC focuses on minority-class performance — the class that actually matters for credit risk.

**Why 0.15 and not the F1-maximizing threshold?**
F1 maximization treats false positives and false negatives as equally costly. They are not. A custom cost function with asymmetric FN/FP costs was used to find the threshold that minimizes expected financial loss, not statistical error.

**Why Gradient Boosting and not XGBoost?**
Scikit-learn's GradientBoostingClassifier was used to maintain pipeline consistency and avoid external dependencies for the baseline comparison. XGBoost/LightGBM are natural next steps for production uplift.

**Why SHAP over built-in feature importances?**
Random Forest and Gradient Boosting feature importances measure how often a feature is used for splits — not how much it actually moves predictions. SHAP provides direction-aware, per-applicant attribution grounded in game theory. This is what regulators and credit officers actually need.

---

## Explainability Output — Per Applicant

![SHAP Waterfall](assets/images/shap.png)

For every applicant assessed, RiskLens produces a full decision report:

```python
{
  "Default_Probability": 0.2099,
  "Risk_Band": "High Risk",
  "Decision": "Manual Review",
  "Recommendation": "Applicant shows elevated risk. Perform enhanced credit review before approval.",
  "Top_Risk_Drivers": [
      ("Applicant Age",     +0.54),
      ("Loan Amount",       +0.48),
      ("Interest Rate",     +0.27),
      ("Months Employed",   +0.13),
      ("Credit Score",      +0.12),
  ],
  "Top_Protective_Factors": [
      ("Income",               -0.24),
      ("Full-time Employment", -0.16),
      ("No Co-Signer",         -0.09),
  ]
}
```

This is not a black box. Every decision has a reason.

---

## Probability Calibration

A model that outputs "70% default probability" should be wrong about 30% of the time for that group — no more, no less. RiskLens validates this.

- **Brier Score: 0.0909** (0 = perfect, 1 = worst possible)
- Calibration curve confirms predicted probabilities track observed default rates
- Calibration validated *before* threshold optimization — the threshold operates on trustworthy probabilities, not just ranked scores

---

## Why RiskLens Matters

Traditional loan default classifiers stop after predicting whether an applicant will default.

RiskLens extends this workflow by:

- Producing calibrated default probabilities
- Optimizing decision thresholds using business costs
- Segmenting applicants into interpretable risk bands
- Generating lending recommendations
- Explaining every prediction with SHAP

The result is an explainable credit risk decisioning system rather than a binary classifier.

> Illustrative asymmetric business costs were used to demonstrate cost-sensitive threshold optimization.

---

## What This Is Not

RiskLens is a research and demonstration system built on a public dataset. It is not a production lending system and should not be used as one without:
- Regulatory compliance review (fair lending law, model risk management)
- Validation on institution-specific historical data
- Ongoing monitoring for distribution shift and feature drift
- Human-in-the-loop review for all flagged applicants

Acknowledging these limitations is part of building responsibly.

---

## Notebook Execution Order

```
01 → 02 → 03 → 04 → 05 → 06
```

Each notebook saves artifacts (`.pkl`, `.csv`) consumed by the next. Run in sequence from a single working directory containing `Loan_default.csv`.

---

## Stack

```
Python 3.10+
pandas · numpy · scikit-learn
shap · matplotlib · seaborn
streamlit · joblib
```

---

## How to Run

```bash
git clone https://github.com/siddhimane8/RiskLens
cd RiskLens
pip install -r requirements.txt

# Run notebooks in order
jupyter notebook notebooks/01_data_understanding.ipynb

# Launch the Streamlit app
streamlit run apps/streamlit_app.py
```

---

## What's Next

- [ ] XGBoost / LightGBM comparison for production-grade performance uplift
- [ ] FastAPI endpoint for model serving
- [ ] Distribution shift monitoring (PSI on key features)
- [ ] Docker containerization for deployment

---


## Author

**Siddhi Mane** — Artificial Intelligence and Machine Learning Student

[![GitHub](https://img.shields.io/badge/GitHub-siddhimane8-181717?style=flat&logo=github&logoColor=white)](https://github.com/siddhimane8)

&nbsp;

---

*Built as part of a flagship AI/ML portfolio targeting production-grade, explainable machine learning systems.*
