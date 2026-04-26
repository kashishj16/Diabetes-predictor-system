"""
STEP 1 — DATA CLEANING & FEATURE ENGINEERING (FIXED)
=====================================================
Place this file in: /data/prepare_data.py
Run with         : python prepare_data.py   (from inside /data folder)
Output           : cleaned_data.csv
"""

import pandas as pd
import numpy as np

# 1. LOAD
df = pd.read_csv("diabetic_data.csv", na_values=["?"], low_memory=False)
print(f"✅ Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. DROP USELESS COLUMNS
DROP_COLS = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
df.drop(columns=DROP_COLS, inplace=True)

# 3. TARGET VARIABLE
df["non_adherent"] = (df["readmitted"] == "<30").astype(int)
df.drop(columns=["readmitted"], inplace=True)
pos = df["non_adherent"].sum()
print(f"✅ Target created → Readmitted <30 days: {pos} ({pos/len(df):.1%})")

# 4. FIX AGE
AGE_MAP = {
    "[0-10)": 5,   "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95,
}
df["age_numeric"] = df["age"].map(AGE_MAP)
df.drop(columns=["age"], inplace=True)

# 5. CONVERT ALL MEDICATION COLUMNS → BINARY
#    Includes combo meds like glyburide-metformin that caused the error
MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone",
    "rosiglitazone", "acarbose", "insulin", "tolbutamide",
    "acetohexamide", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
for col in MED_COLS:
    if col in df.columns:
        df[col + "_used"] = (df[col] != "No").astype(int)
        df.drop(columns=[col], inplace=True)
print(f"✅ All medication columns → binary")

# 6. FEATURE ENGINEERING
med_used_cols = [c for c in df.columns if c.endswith("_used")]
df["med_count"]           = df[med_used_cols].sum(axis=1)
df["visit_burden"]        = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
df["procedure_intensity"] = df["num_procedures"] + df["num_lab_procedures"]
df["comorbidity_score"]   = df["number_diagnoses"]

# HbA1c blood sugar control
HBA1C_MAP = {">8": 2, ">7": 1, "Norm": 0, "None": -1}
if "A1Cresult" in df.columns:
    df["a1c_risk"] = df["A1Cresult"].map(HBA1C_MAP).fillna(-1)
    df.drop(columns=["A1Cresult"], inplace=True)

# Glucose serum level
if "max_glu_serum" in df.columns:
    GLU_MAP = {">300": 2, ">200": 1, "Norm": 0, "None": -1}
    df["glu_serum_risk"] = df["max_glu_serum"].map(GLU_MAP).fillna(-1)
    df.drop(columns=["max_glu_serum"], inplace=True)

print(f"✅ Feature engineering done")

# 7. ENCODE STANDARD CATEGORICALS
CAT_COLS = ["race", "gender", "change", "diabetesMed"]
for col in CAT_COLS:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.codes

# 8. DROP DIAGNOSIS CODE COLUMNS
df.drop(columns=["diag_1", "diag_2", "diag_3"], errors="ignore", inplace=True)

# 9. CATCH-ALL — encode ANY remaining object column
#    This is the fix that prevents the XGBoost dtype error
obj_cols = df.select_dtypes(include="object").columns.tolist()
if obj_cols:
    print(f"⚠️  Encoding remaining object columns: {obj_cols}")
    for col in obj_cols:
        df[col] = df[col].astype("category").cat.codes

# 10. FILL NULLS
df.fillna(-1, inplace=True)

# 11. FINAL CHECK
obj_remaining = df.select_dtypes(include="object").columns.tolist()
if obj_remaining:
    print(f"❌ Still has object columns: {obj_remaining}")
else:
    print(f"✅ All columns numeric — XGBoost ready!")

# 12. SAVE
df.to_csv("cleaned_data.csv", index=False)
print(f"\n{'='*50}")
print(f"✅ DONE! Saved → cleaned_data.csv")
print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"{'='*50}")
print(f"\nNext: go to /notebooks → python train_model.py")