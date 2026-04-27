# Patient Readmission Risk Predictor

A full-stack machine learning system that predicts 30-day hospital readmission risk for diabetic patients. Given a patient's clinical profile at discharge, the system returns a risk score, an explanation of the key contributing factors, and a recommended care action.

---

## Problem

11.2% of diabetic patients are readmitted to hospital within 30 days of discharge. Many of these readmissions are preventable with timely intervention — but only if the right patients are identified before they leave, not after they return.

This project builds a data-driven scoring system that gives clinical staff an objective, explainable readmission probability for every patient at discharge.

---

## What It Does

- Predicts 30-day readmission probability from a patient's clinical profile
- Explains *why* a patient is flagged using SHAP feature attribution
- Segments the patient population into Low / Medium / High risk tiers via K-Means clustering
- Serves predictions through a REST API
- Displays results in an interactive dashboard with Light, Dark, and Default themes

---

## Architecture

```
Raw Data (Kaggle)
      │
      ▼
Data Cleaning & Feature Engineering   ← data/prepare_data.py
      │
      ▼
XGBoost Classifier + SHAP Explainer   ← notebooks/train_model.py
+ K-Means Segmentation
      │
      ▼
Flask REST API  (POST /predict)        ← api/app.py
      │
      ▼
Streamlit Dashboard                    ← dashboard/app.py
```

---

## Project Structure

```
patient-adherence-predictor/
├── data/
│   ├── prepare_data.py         # Cleaning and feature engineering
│   └── cleaned_data.csv        # Generated — not tracked in git
├── notebooks/
│   └── train_model.py          # XGBoost + SHAP + KMeans training
├── api/
│   ├── app.py                  # Flask REST API
│   ├── model.pkl               # Generated — not tracked in git
│   ├── explainer.pkl           # Generated — not tracked in git
│   ├── feature_names.pkl       # Generated — not tracked in git
│   └── segment_model.pkl       # Generated — not tracked in git
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── report/
│   └── consulting_memo.docx    # Business findings and recommendations
├── requirements.txt
└── README.md
```

---

## Dataset

**Diabetes 130-US Hospitals for Years 1999–2008**
Source: [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes) / UCI Machine Learning Repository

- 101,766 patient encounters across 130 US hospitals
- 50 features covering demographics, diagnoses, medications, lab results, and discharge outcomes
- Target variable: readmitted within 30 days (`<30`) vs. not

---

## Feature Engineering

Five clinically meaningful features were derived from raw columns:

| Feature | Description |
|---|---|
| `med_count` | Total diabetes medications prescribed — proxy for treatment complexity |
| `visit_burden` | Prior outpatient + emergency + inpatient visits combined |
| `procedure_intensity` | Lab + clinical procedures at current visit — case severity indicator |
| `comorbidity_score` | Number of distinct diagnoses — multi-condition complexity |
| `a1c_risk` | HbA1c result encoded as a risk tier — direct glycaemic control measure |

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.66 |
| Top-Decile Precision | 24.75% |
| Baseline Readmission Rate | 11.2% |
| Training Sample | 12,000 patients (stratified) |

Top-decile precision of 24.75% means: flagging the 10% highest-risk patients identifies readmissions at 2.2x the baseline rate.

**Top predictors (by mean SHAP value):**

1. `number_inpatient` — prior hospital stays
2. `discharge_disposition_id` — post-discharge destination
3. `time_in_hospital` — length of current stay
4. `visit_burden` — total prior visit frequency
5. `num_lab_procedures` — clinical complexity at current visit

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/patient-adherence-predictor.git
cd patient-adherence-predictor

# 2. Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# https://www.kaggle.com/datasets/brandao/diabetes
# Place diabetic_data.csv inside /data
```

---

## Running the Project

**Step 1 — Prepare data**
```bash
cd data
python prepare_data.py
```

**Step 2 — Train model**
```bash
cd notebooks
python train_model.py
```

**Step 3 — Start API**
```bash
cd api
python app.py
# Running at http://localhost:5000
```

**Step 4 — Launch dashboard**
```bash
# New terminal
cd dashboard
streamlit run app.py
# Opens at http://localhost:8501
```

---

## API Reference

**Health check**
```
GET /health
```

**List expected features**
```
GET /features
```

**Predict**
```
POST /predict
Content-Type: application/json

{
  "features": {
    "number_inpatient": 2,
    "age_numeric": 65,
    "med_count": 3,
    "visit_burden": 4,
    "time_in_hospital": 5
  }
}
```

**Response**
```json
{
  "risk_score": 0.73,
  "risk_label": "High Risk",
  "segment": "High Risk",
  "top_reasons": [
    {
      "feature": "number_inpatient",
      "value": 2.0,
      "impact": "increases risk",
      "shap": 0.27
    }
  ],
  "recommendation": "Immediate action: assign pharmacist outreach within 48hrs of discharge."
}
```

---

## Dashboard

Three built-in themes switchable from the sidebar: Default, Light, Dark.

- Fill in patient details on the left panel
- Click **Run Prediction**
- Right panel shows risk score, progress bar, top contributing factors, and recommended action
- Sidebar displays live API connection status

---

## Limitations

- Trained on a 15k-row stratified sample due to local compute constraints. Full-dataset training expected to push AUC to ~0.75–0.78
- ICD-9 diagnosis codes excluded due to high cardinality — grouping into CCS categories is a planned improvement
- Dataset covers 1999–2008; retraining on contemporary data recommended before any clinical use
- Demographic fairness analysis not yet conducted

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Model | XGBoost 2.0 |
| Explainability | SHAP (TreeExplainer) |
| Segmentation | K-Means (scikit-learn) |
| API | Flask 3.0 |
| Dashboard | Streamlit 1.32 |
| Data | pandas, numpy |

---

## Author

**Kashish**
[github.com/kashishj16][def]

[def]: https://github.com/kashishj16