"""
STEP 3 — FLASK REST API
========================
Place in : /api/app.py
Run      : python app.py   (from inside /api folder)
Server   : http://localhost:5000

Endpoints:
  GET  /health      → check server is alive
  GET  /features    → list all expected feature names
  POST /predict     → send patient data, get risk score back
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# LOAD ALL MODEL ARTIFACTS AT STARTUP
# These are the 4 .pkl files saved by train_model.py
# ─────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)

print("Loading model artifacts...")
model         = pickle.load(open(os.path.join(BASE, "model.pkl"),          "rb"))
explainer     = pickle.load(open(os.path.join(BASE, "explainer.pkl"),      "rb"))
feature_names = pickle.load(open(os.path.join(BASE, "feature_names.pkl"),  "rb"))
kmeans, scaler, seg_feats, SEGMENT_LABELS = pickle.load(
    open(os.path.join(BASE, "segment_model.pkl"), "rb")
)
print(f"✅ Model loaded | {len(feature_names)} features expected")


# ─────────────────────────────────────────────────────────────
# HELPER: extract top 3 reasons from SHAP values
# ─────────────────────────────────────────────────────────────
def get_top_reasons(shap_vals, feat_values, n=3):
    """
    Returns top N features driving this patient's risk score.
    Example output:
      [{"feature": "number_inpatient", "value": 3.0,
        "impact": "increases risk", "shap": 0.27}]
    """
    triplets = sorted(
        zip(feature_names, shap_vals, feat_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:n]

    return [
        {
            "feature": feat,
            "value":   round(float(val), 2),
            "impact":  "increases risk" if shap > 0 else "decreases risk",
            "shap":    round(float(shap), 4),
        }
        for feat, shap, val in triplets
    ]


# ─────────────────────────────────────────────────────────────
# ROUTE 1: Health check
# Test: open browser → http://localhost:5000/health
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model":  "Patient Adherence Predictor v1.0",
        "features": len(feature_names),
    })


# ─────────────────────────────────────────────────────────────
# ROUTE 2: List feature names
# Test: open browser → http://localhost:5000/features
# ─────────────────────────────────────────────────────────────
@app.route("/features", methods=["GET"])
def features():
    return jsonify({
        "feature_names": feature_names,
        "count": len(feature_names),
    })


# ─────────────────────────────────────────────────────────────
# ROUTE 3: Predict — the main endpoint
#
# Expects POST request with JSON body like:
# {
#   "features": {
#     "number_inpatient": 2,
#     "age_numeric": 65,
#     "med_count": 3,
#     ... (any/all feature names)
#   }
# }
#
# Returns:
# {
#   "risk_score": 0.73,
#   "risk_label": "High Risk",
#   "segment": "High Risk",
#   "top_reasons": [...],
#   "recommendation": "..."
# }
# ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        if not body or "features" not in body:
            return jsonify({"error": "Send JSON with a 'features' key"}), 400

        feat_dict = body["features"]

        # Build feature vector in exact order the model expects
        # Missing features default to -1 (= unknown)
        feat_vector = np.array(
            [float(feat_dict.get(fname, -1)) for fname in feature_names],
            dtype=np.float32
        ).reshape(1, -1)

        # ── Risk score ──────────────────────────────────────
        risk_score = float(model.predict_proba(feat_vector)[0][1])

        if risk_score >= 0.60:
            risk_label     = "High Risk"
            recommendation = (
                "Immediate action: assign pharmacist outreach within 48hrs of discharge. "
                "Schedule follow-up calls at Day 7 and Day 14."
            )
        elif risk_score >= 0.35:
            risk_label     = "Medium Risk"
            recommendation = (
                "Moderate action: send automated medication reminder SMS. "
                "Schedule one follow-up call at Day 14."
            )
        else:
            risk_label     = "Low Risk"
            recommendation = "Standard discharge protocol. No immediate intervention needed."

        # ── SHAP explanation ─────────────────────────────────
        shap_vals   = explainer.shap_values(feat_vector)[0]
        top_reasons = get_top_reasons(shap_vals, feat_vector[0])

        # ── Patient segment ──────────────────────────────────
        seg_input  = np.array([[feat_dict.get(f, 0) for f in seg_feats]], dtype=np.float32)
        seg_scaled = scaler.transform(seg_input)
        seg_id     = int(kmeans.predict(seg_scaled)[0])
        segment    = SEGMENT_LABELS.get(seg_id, "Unknown")

        return jsonify({
            "risk_score":     round(risk_score, 3),
            "risk_label":     risk_label,
            "segment":        segment,
            "top_reasons":    top_reasons,
            "recommendation": recommendation,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting Patient Adherence API...")
    print("   Health check : http://localhost:5000/health")
    print("   Features list: http://localhost:5000/features")
    print("   Predict      : POST http://localhost:5000/predict\n")
    app.run(debug=True, port=5000)