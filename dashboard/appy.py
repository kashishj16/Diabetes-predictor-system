"""
STEP 4 — STREAMLIT DASHBOARD
==============================
Place in : /dashboard/app.py
Run      : streamlit run app.py   (from inside /dashboard folder)
Requires : Flask API running on http://localhost:5000

Features:
  - Patient risk prediction form
  - Risk score display with segment label
  - Top 3 reasons (SHAP-based)
  - Recommendation panel
  - Light / Dark / Default theme switcher
"""

import streamlit as st
import requests
import json

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Patient Adherence Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:5000"

# ─────────────────────────────────────────────────────────────
# THEME DEFINITIONS
# ─────────────────────────────────────────────────────────────
THEMES = {
    "Default": {
        "bg":           "#f5f5f5",
        "card":         "#ffffff",
        "text":         "#1a1a1a",
        "subtext":      "#555555",
        "border":       "#e0e0e0",
        "accent":       "#2c5f8a",
        "low":          "#2e7d32",
        "medium":       "#e65100",
        "high":         "#b71c1c",
        "low_bg":       "#e8f5e9",
        "medium_bg":    "#fff3e0",
        "high_bg":      "#ffebee",
        "bar_bg":       "#e0e0e0",
    },
    "Light": {
        "bg":           "#ffffff",
        "card":         "#f9f9f9",
        "text":         "#111111",
        "subtext":      "#444444",
        "border":       "#d0d0d0",
        "accent":       "#1a4f7a",
        "low":          "#1b5e20",
        "medium":       "#bf360c",
        "high":         "#7f0000",
        "low_bg":       "#f1f8e9",
        "medium_bg":    "#fbe9e7",
        "high_bg":      "#fce4ec",
        "bar_bg":       "#e8e8e8",
    },
    "Dark": {
        "bg":           "#121212",
        "card":         "#1e1e1e",
        "text":         "#e0e0e0",
        "subtext":      "#aaaaaa",
        "border":       "#333333",
        "accent":       "#5b9bd5",
        "low":          "#66bb6a",
        "medium":       "#ffa726",
        "high":         "#ef5350",
        "low_bg":       "#1b2e1c",
        "medium_bg":    "#2e1f0e",
        "high_bg":      "#2e0f0f",
        "bar_bg":       "#2a2a2a",
    },
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    theme_choice = st.radio("Theme", list(THEMES.keys()), index=0)
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This tool predicts 30-day hospital readmission risk "
        "for diabetic patients using an XGBoost model trained "
        "on 15,000 patient records."
    )
    st.markdown("---")
    st.markdown("### API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API connected")
        else:
            st.error("API returned error")
    except:
        st.error("API offline — run app.py in /api first")

T = THEMES[theme_choice]

# ─────────────────────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Global background */
  .stApp {{
      background-color: {T['bg']};
  }}

  /* All text */
  html, body, [class*="css"], .stMarkdown, p, span, label {{
      color: {T['text']} !important;
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: {T['card']};
      border-right: 1px solid {T['border']};
  }}

  /* Input widgets */
  .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
      background-color: {T['card']};
      color: {T['text']};
      border: 1px solid {T['border']};
  }}

  /* Buttons */
  .stButton > button {{
      background-color: {T['accent']};
      color: #ffffff;
      border: none;
      border-radius: 4px;
      padding: 0.5rem 1.5rem;
      font-size: 0.9rem;
      font-weight: 500;
      cursor: pointer;
      transition: opacity 0.2s;
  }}
  .stButton > button:hover {{
      opacity: 0.85;
  }}

  /* Cards */
  .card {{
      background-color: {T['card']};
      border: 1px solid {T['border']};
      border-radius: 6px;
      padding: 1.2rem 1.5rem;
      margin-bottom: 1rem;
  }}

  /* Section headers */
  .section-title {{
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {T['subtext']};
      margin-bottom: 0.8rem;
  }}

  /* Risk badge */
  .badge-low    {{ background:{T['low_bg']};    color:{T['low']};    border:1px solid {T['low']};    border-radius:4px; padding:4px 12px; font-weight:600; display:inline-block; }}
  .badge-medium {{ background:{T['medium_bg']}; color:{T['medium']}; border:1px solid {T['medium']}; border-radius:4px; padding:4px 12px; font-weight:600; display:inline-block; }}
  .badge-high   {{ background:{T['high_bg']};   color:{T['high']};   border:1px solid {T['high']};   border-radius:4px; padding:4px 12px; font-weight:600; display:inline-block; }}

  /* Score number */
  .score-number {{
      font-size: 2.8rem;
      font-weight: 700;
      line-height: 1;
      color: {T['text']};
  }}

  /* Progress bar */
  .bar-track {{
      background: {T['bar_bg']};
      border-radius: 4px;
      height: 8px;
      margin: 0.4rem 0 1rem 0;
  }}

  /* Reason row */
  .reason-row {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0.5rem 0;
      border-bottom: 1px solid {T['border']};
      font-size: 0.88rem;
  }}
  .reason-row:last-child {{ border-bottom: none; }}
  .reason-feat {{ color: {T['text']}; font-weight: 500; }}
  .reason-val  {{ color: {T['subtext']}; font-size: 0.82rem; }}
  .tag-up   {{ background:{T['high_bg']};  color:{T['high']};   border-radius:3px; padding:2px 8px; font-size:0.78rem; }}
  .tag-down {{ background:{T['low_bg']};   color:{T['low']};    border-radius:3px; padding:2px 8px; font-size:0.78rem; }}

  /* Recommendation box */
  .rec-box {{
      background: {T['card']};
      border-left: 3px solid {T['accent']};
      border-radius: 0 4px 4px 0;
      padding: 0.9rem 1.2rem;
      font-size: 0.9rem;
      color: {T['text']};
      margin-top: 0.5rem;
  }}

  /* Divider */
  hr {{ border-color: {T['border']}; margin: 1.2rem 0; }}

  /* Hide Streamlit chrome */
  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"<h2 style='color:{T['text']}; font-weight:600; margin-bottom:0.2rem;'>Patient Adherence Predictor</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{T['subtext']}; font-size:0.9rem; margin-bottom:1.5rem;'>Predicts 30-day hospital readmission risk for diabetic patients</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LAYOUT: INPUT LEFT | RESULTS RIGHT
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

# ── INPUT FORM ───────────────────────────────────────────────
with left:
    st.markdown(f"<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)

    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            age          = st.number_input("Age",                    min_value=0,   max_value=100, value=65)
            inpatient    = st.number_input("Prior inpatient visits",  min_value=0,   max_value=20,  value=1)
            outpatient   = st.number_input("Prior outpatient visits", min_value=0,   max_value=30,  value=2)
            emergency    = st.number_input("Prior emergency visits",  min_value=0,   max_value=20,  value=0)
            time_in_hosp = st.number_input("Days in hospital",        min_value=1,   max_value=30,  value=4)
        with c2:
            num_diagnoses   = st.number_input("Number of diagnoses",    min_value=1,  max_value=16,  value=6)
            num_meds        = st.number_input("Number of medications",  min_value=0,  max_value=30,  value=5)
            num_lab_procs   = st.number_input("Lab procedures",         min_value=0,  max_value=100, value=40)
            num_procs       = st.number_input("Other procedures",       min_value=0,  max_value=20,  value=1)
            med_count       = st.number_input("Diabetes meds count",    min_value=0,  max_value=10,  value=2)

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            diabetes_med = st.selectbox("On diabetes medication", ["Yes", "No"])
            insulin_used = st.selectbox("Insulin prescribed",     ["Yes", "No"])
        with c4:
            a1c_result   = st.selectbox("HbA1c result", [">8 (poor control)", ">7 (borderline)", "Normal", "Not tested"])
            gender       = st.selectbox("Gender", ["Female", "Male", "Other"])

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("Run Prediction")


# ── RESULTS PANEL ────────────────────────────────────────────
with right:
    st.markdown(f"<div class='section-title'>Prediction Results</div>", unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown(
            f"<div class='card' style='color:{T['subtext']}; font-size:0.9rem;'>"
            "Fill in the patient details on the left and click <strong>Run Prediction</strong>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # Build feature dict — map form inputs to model feature names
        a1c_map  = {">8 (poor control)": 2, ">7 (borderline)": 1, "Normal": 0, "Not tested": -1}
        gen_map  = {"Female": 0, "Male": 1, "Other": -1}
        bool_map = {"Yes": 1, "No": 0}

        visit_burden       = inpatient + outpatient + emergency
        procedure_intensity = num_procs + num_lab_procs
        comorbidity_score  = num_diagnoses

        features = {
            "age_numeric":              age,
            "number_inpatient":         inpatient,
            "number_outpatient":        outpatient,
            "number_emergency":         emergency,
            "time_in_hospital":         time_in_hosp,
            "number_diagnoses":         num_diagnoses,
            "num_medications":          num_meds,
            "num_lab_procedures":       num_lab_procs,
            "num_procedures":           num_procs,
            "med_count":                med_count,
            "visit_burden":             visit_burden,
            "procedure_intensity":      procedure_intensity,
            "comorbidity_score":        comorbidity_score,
            "a1c_risk":                 a1c_map[a1c_result],
            "diabetesMed":              bool_map[diabetes_med],
            "insulin_used":             bool_map[insulin_used],
            "gender":                   gen_map[gender],
        }

        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json={"features": features},
                timeout=5,
            )
            data = resp.json()

            if "error" in data:
                st.error(f"API error: {data['error']}")
            else:
                score   = data["risk_score"]
                label   = data["risk_label"]
                segment = data["segment"]
                reasons = data["top_reasons"]
                rec     = data["recommendation"]

                # Badge class
                badge_class = {"Low Risk": "badge-low", "Medium Risk": "badge-medium", "High Risk": "badge-high"}.get(label, "badge-low")
                bar_color   = {"Low Risk": T["low"],     "Medium Risk": T["medium"],    "High Risk": T["high"]}.get(label, T["low"])

                # Score card
                st.markdown(f"""
                <div class='card'>
                    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                        <div>
                            <div style='font-size:0.78rem; color:{T['subtext']}; margin-bottom:0.3rem;'>30-day readmission risk</div>
                            <div class='score-number'>{score*100:.1f}%</div>
                        </div>
                        <div>
                            <span class='{badge_class}'>{label}</span>
                            <div style='font-size:0.78rem; color:{T['subtext']}; margin-top:0.5rem; text-align:right;'>Segment: {segment}</div>
                        </div>
                    </div>
                    <div class='bar-track'>
                        <div style='width:{min(score*100, 100):.1f}%; background:{bar_color}; height:8px; border-radius:4px; transition:width 0.4s;'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.75rem; color:{T['subtext']};'>
                        <span>0%</span><span>50%</span><span>100%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top reasons
                st.markdown(f"<div class='section-title' style='margin-top:0.8rem;'>Key factors</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                for r in reasons:
                    tag_class = "tag-up" if r["impact"] == "increases risk" else "tag-down"
                    tag_text  = "raises risk" if r["impact"] == "increases risk" else "lowers risk"
                    st.markdown(f"""
                    <div class='reason-row'>
                        <div>
                            <div class='reason-feat'>{r['feature'].replace('_', ' ').title()}</div>
                            <div class='reason-val'>Value: {r['value']}</div>
                        </div>
                        <span class='{tag_class}'>{tag_text}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # Recommendation
                st.markdown(f"<div class='section-title' style='margin-top:0.8rem;'>Recommended action</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='rec-box'>{rec}</div>", unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot reach API. Make sure Flask is running: cd ../api && python app.py")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align:center; font-size:0.78rem; color:{T['subtext']};'>"
    "Patient Adherence Predictor &nbsp;|&nbsp; XGBoost + SHAP &nbsp;|&nbsp; "
    "Trained on Diabetes 130-US Hospitals dataset"
    "</p>",
    unsafe_allow_html=True,
)