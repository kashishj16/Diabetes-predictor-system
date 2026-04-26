"""
STEP 2 — MODEL TRAINING (ULTRA LOW RAM)
========================================
Close Chrome, VS Code, everything before running.
Place in : /notebooks/train_model.py
Run      : python train_model.py
Output   : model.pkl, explainer.pkl, feature_names.pkl, segment_model.pkl → saved to /api/
"""

import pandas as pd
import numpy as np
import pickle, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

os.makedirs("../api", exist_ok=True)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# 1. LOAD ONLY 15,000 ROWS — using nrows, load as float32
#    nrows loads from top of file — fast, no full file read
#    dtype=float32 uses HALF the RAM of default float64
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(
    "../data/cleaned_data.csv",
    nrows=15000,
    dtype=np.float32
)
print(f"✅ Loaded {len(df)} rows  |  RAM used: ~{df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"   Target rate: {df['non_adherent'].mean():.1%} readmitted <30 days")

X = df.drop(columns=["non_adherent"])
y = df["non_adherent"].astype(np.int8)
feature_names = list(X.columns)
del df  # free RAM immediately

# ─────────────────────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"✅ Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN XGBOOST
#    hist method = fastest + lowest RAM
#    n_estimators=100, max_depth=3 = minimal memory footprint
# ─────────────────────────────────────────────────────────────
spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
print(f"⚙️  Class imbalance: {spw:.1f}x")

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    tree_method="hist",
    nthread=1,
    eval_metric="auc",
    random_state=42,
    verbosity=0,
)

print("⏳ Training XGBoost...")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print("✅ Training done!")

# ─────────────────────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
ap  = average_precision_score(y_test, y_prob)
top10_thresh = np.percentile(y_prob, 90)
top10_prec   = y_test.values[y_prob >= top10_thresh].mean()

print(f"\n{'='*45}")
print(f"📊 RESULTS")
print(f"   ROC-AUC              : {auc:.4f}")
print(f"   Avg Precision        : {ap:.4f}")
print(f"   Top-decile precision : {top10_prec:.2%}  ← for ZS memo")
print(f"\n{classification_report(y_test, y_pred, target_names=['Adherent','Non-adherent'])}")
print(f"{'='*45}")

# ─────────────────────────────────────────────────────────────
# 5. SHAP — on only 200 rows to save RAM
# ─────────────────────────────────────────────────────────────
print("⏳ Building SHAP explainer (200 rows)...")
explainer   = shap.TreeExplainer(model)
shap_sample = X_test.iloc[:200]
shap_vals   = explainer.shap_values(shap_sample)

mean_abs    = np.abs(shap_vals).mean(axis=0)
top_feats   = sorted(zip(feature_names, mean_abs), key=lambda x: -x[1])[:8]

print("✅ SHAP done!")
print(f"\n📊 TOP 8 FEATURES:")
print(f"{'Feature':<30} {'SHAP Score':>10}")
print("-" * 42)
for f, v in top_feats:
    print(f"  {f:<28} {v:.4f}  {'█' * int(v*60)}")

# ─────────────────────────────────────────────────────────────
# 6. K-MEANS SEGMENTATION
# ─────────────────────────────────────────────────────────────
print("\n⏳ K-Means segmentation...")
seg_feats = [f for f in ["age_numeric","med_count","visit_burden",
                          "comorbidity_score","procedure_intensity"] if f in X.columns]

scaler   = StandardScaler()
X_seg    = scaler.fit_transform(X[seg_feats])
kmeans   = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_seg)

all_probs = model.predict_proba(X)[:, 1]
cluster_risk   = {c: all_probs[clusters == c].mean() for c in range(3)}
sorted_c       = sorted(cluster_risk, key=cluster_risk.get)
SEGMENT_LABELS = {sorted_c[0]: "Low Risk", sorted_c[1]: "Medium Risk", sorted_c[2]: "High Risk"}

print("✅ Segments done!")
print(f"\n{'Segment':<14} {'Count':>6} {'Avg Risk':>9}")
print("-" * 32)
for cid, label in SEGMENT_LABELS.items():
    m = clusters == cid
    print(f"  {label:<12} {m.sum():>6} {all_probs[m].mean():>9.2%}")

# ─────────────────────────────────────────────────────────────
# 7. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
pickle.dump(model,          open("../api/model.pkl",         "wb"))
pickle.dump(explainer,      open("../api/explainer.pkl",     "wb"))
pickle.dump(feature_names,  open("../api/feature_names.pkl", "wb"))
pickle.dump(
    (kmeans, scaler, seg_feats, SEGMENT_LABELS),
    open("../api/segment_model.pkl", "wb")
)

print(f"\n{'='*45}")
print(f"✅ All saved to /api/")
print(f"   model.pkl  |  explainer.pkl")
print(f"   feature_names.pkl  |  segment_model.pkl")
print(f"{'='*45}")
print(f"\nNext: cd ../api  →  python app.py")