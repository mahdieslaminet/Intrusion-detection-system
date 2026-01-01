import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# =========================
# Fixed (per your request)
# =========================
DATASET_NAME = "CIC-IDS-2017"
PREFIX = "ids2017"
MODEL_NAME = "LightGBM"
LABEL_COL = "Label"  # expected label column in uploaded test CSV (if exists)


# ---------------- Paths ----------------
BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "outputs" / "models"
FS_DIR = BASE / "data" / "fs"
SPLITS_DIR = BASE / "data" / "splits"        # optional scaler
MAPPINGS_DIR = BASE / "data" / "mappings"    # optional label maps


# ---------------- Streamlit page ----------------
st.set_page_config(page_title="IDS Web Tester", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .stMetric { background: #0b1220; padding: 10px 12px; border-radius: 10px; }
      div[data-testid="stMetricValue"] { font-size: 24px; }
      .small-note { opacity: 0.75; font-size: 12px; }
      .pill {
        display:inline-block; padding: 6px 10px; border-radius:999px;
        background:#111a2e; border:1px solid #22304f; margin-right:8px;
        font-size: 12px; opacity: .95;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ IDS Web Tester")
st.caption("Upload the CIC-IDS-2017 test CSV → run LightGBM detection → view predictions + model metrics (if Label exists).")

st.markdown(
    f"""
    <div>
      <span class="pill">Dataset: <b>{DATASET_NAME}</b></span>
      <span class="pill">Model: <b>{MODEL_NAME}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- Caching ----------------
@st.cache_data
def load_kept_columns(prefix: str) -> list[str]:
    p = FS_DIR / f"{prefix}_kept_columns.txt"
    if not p.exists():
        raise FileNotFoundError(f"Missing kept columns file: {p}")
    cols = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cols


@st.cache_resource
def load_model(prefix: str, model_name: str):
    p = MODELS_DIR / f"{prefix}_{model_name}.joblib"
    if not p.exists():
        raise FileNotFoundError(f"Missing model file: {p}")
    return load(p)


@st.cache_resource
def load_optional_scaler(prefix: str):
    p = SPLITS_DIR / f"{prefix}_scaler.joblib"
    return load(p) if p.exists() else None


@st.cache_data
def load_optional_label_map(prefix: str):
    """
    Optional mapping file:
      data/mappings/{prefix}_label_map.json
    Example:
      {"0":"BENIGN", "1":"DoS", ...}
    """
    p = MAPPINGS_DIR / f"{prefix}_label_map.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            mp = json.load(f)
        out = {}
        for k, v in mp.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                pass
        return out if out else None
    return None


# ---------------- Helpers ----------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def read_csv_robust(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1")


def predict_pipeline(df_raw: pd.DataFrame, prefix: str, model_name: str):
    kept_cols = load_kept_columns(prefix)
    model = load_model(prefix, model_name)
    scaler = load_optional_scaler(prefix)

    df = clean_columns(df_raw)

    missing = [c for c in kept_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Your uploaded CSV is missing required feature columns.\n"
            "Missing examples:\n- " + "\n- ".join(missing[:25]) +
            ("\n..." if len(missing) > 25 else "")
        )

    X = df[kept_cols]
    X = coerce_numeric(X)

    nan_ratio = float(X.isna().mean().mean())
    X = X.fillna(0.0)

    used_scaler = False
    X_arr = X.values

    if scaler is not None:
        try:
            X_arr = scaler.transform(X_arr)
            used_scaler = True
        except Exception:
            used_scaler = False

    y_pred = model.predict(X_arr)

    return y_pred, nan_ratio, used_scaler, kept_cols


def try_build_y_true(df_out: pd.DataFrame, label_col: str, label_map: dict | None):
    """
    Supports:
      - numeric Label
      - string Label (mapped via label_map if available)
    """
    if label_col not in df_out.columns:
        return None, "No Label column found in your CSV."

    y = df_out[label_col]

    y_num = pd.to_numeric(y, errors="coerce")
    if not y_num.isna().any():
        return y_num.astype(int).values, "Label is numeric."

    if label_map:
        inv = {v: k for k, v in label_map.items()}
        y_mapped = y.astype(str).map(inv)
        if y_mapped.isna().any():
            return None, "Label exists but cannot be fully mapped to class IDs."
        return y_mapped.astype(int).values, "Label mapped using label_map.json."

    return None, "Label exists but is not numeric. Provide label_map.json to evaluate."


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")

    st.write(f"**Dataset:** {DATASET_NAME}")
    st.write(f"**Model:** {MODEL_NAME}")

    st.markdown("---")
    show_attack_only = st.checkbox("Show only suspected attacks", value=False)
    max_rows_show = st.slider("Max rows to display", 50, 2000, 200, step=50)

    st.markdown("---")
    st.subheader("Optional files (recommended)")
    st.markdown(
        """
        <div class="small-note">
        • data/mappings/ids2017_label_map.json  → show class names + enable evaluation for string labels<br/>
        • data/splits/ids2017_scaler.joblib     → apply training scaler (best fidelity)<br/>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- Main: Upload ----------------
uploaded = st.file_uploader("Upload CIC-IDS-2017 test CSV", type=["csv"])

if uploaded is None:
    st.info("Upload the CIC-IDS-2017 test CSV to start.")
    st.stop()

df_raw = read_csv_robust(uploaded)
df_raw = clean_columns(df_raw)

left, right = st.columns([1.2, 0.8], gap="large")

with left:
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

with right:
    st.subheader("Quick Info")
    st.write(f"**Rows:** {len(df_raw):,}")
    st.write(f"**Columns:** {df_raw.shape[1]:,}")
    st.write("**Detected columns (first 12):**")
    st.code(", ".join(df_raw.columns[:12]) + (" ..." if len(df_raw.columns) > 12 else ""))

run = st.button("🚀 Run Detection", type="primary")

if not run:
    st.stop()


# ---------------- Run Prediction ----------------
label_map = load_optional_label_map(PREFIX)

try:
    y_pred, nan_ratio, used_scaler, kept_cols = predict_pipeline(df_raw, PREFIX, MODEL_NAME)
except ModuleNotFoundError as e:
    # This is the common case when lightgbm is not installed but model is LightGBM
    st.error(
        "A required Python module is missing.\n\n"
        f"**Details:** {e}\n\n"
        "If you are using the LightGBM model, install it first:\n"
        "- `pip install lightgbm`\n"
        "- On macOS (Apple Silicon) you may need: `brew install libomp` then `pip install lightgbm`"
    )
    st.stop()
except Exception as e:
    st.error(str(e))
    st.stop()

df_out = df_raw.copy()
df_out["predicted_class_id"] = y_pred.astype(int)

if label_map:
    df_out["predicted_label"] = df_out["predicted_class_id"].map(label_map).fillna(df_out["predicted_class_id"].astype(str))
else:
    df_out["predicted_label"] = df_out["predicted_class_id"].astype(str)

# Optional: Attack-only filter
if show_attack_only:
    if label_map and any(str(v).upper() == "BENIGN" for v in label_map.values()):
        benign_ids = [k for k, v in label_map.items() if str(v).upper() == "BENIGN"]
        df_view = df_out[~df_out["predicted_class_id"].isin(benign_ids)]
    else:
        df_view = df_out[df_out["predicted_class_id"] != 0]
else:
    df_view = df_out


# ---------------- Summary metrics ----------------
st.markdown("## Results")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total rows", f"{len(df_out):,}")
m2.metric("Features used", f"{len(kept_cols):,}")
m3.metric("NaN ratio (after numeric coercion)", f"{nan_ratio:.3f}")
m4.metric("Scaler applied", "Yes ✅" if used_scaler else "No ⚠️")

if not used_scaler:
    st.warning("Scaler was not applied (missing or incompatible scaler file). Tree-based models often work fine, but for maximum fidelity, save and load the training scaler.")


# ---------------- Predicted distribution ----------------
st.subheader("Predicted Class Distribution")
vc = df_out["predicted_class_id"].value_counts().sort_index()

fig = plt.figure()
plt.bar(vc.index.astype(str), vc.values)
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.title(f"{DATASET_NAME} • {MODEL_NAME}")
st.pyplot(fig)

if label_map:
    with st.expander("Show class name mapping"):
        st.json({str(k): v for k, v in sorted(label_map.items(), key=lambda x: x[0])})


# ---------------- Evaluation (Accuracy/Precision/etc) ----------------
st.subheader("Model Performance (if your CSV contains Label)")
y_true, why = try_build_y_true(df_out, LABEL_COL, label_map)

if y_true is None:
    st.info(f"Performance metrics not available: {why}")
else:
    y_pred_int = df_out["predicted_class_id"].astype(int).values

    acc = accuracy_score(y_true, y_pred_int)
    prec_macro = precision_score(y_true, y_pred_int, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred_int, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_int, average="macro", zero_division=0)

    prec_w = precision_score(y_true, y_pred_int, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred_int, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred_int, average="weighted", zero_division=0)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Accuracy", f"{acc:.4f}")
    r2.metric("Precision (macro)", f"{prec_macro:.4f}")
    r3.metric("Recall (macro)", f"{rec_macro:.4f}")
    r4.metric("F1 (macro)", f"{f1_macro:.4f}")

    r5, r6, r7 = st.columns(3)
    r5.metric("Precision (weighted)", f"{prec_w:.4f}")
    r6.metric("Recall (weighted)", f"{rec_w:.4f}")
    r7.metric("F1 (weighted)", f"{f1_w:.4f}")

    with st.expander("Classification report"):
        st.text(classification_report(y_true, y_pred_int, zero_division=0))

    with st.expander("Confusion matrix"):
        cm = confusion_matrix(y_true, y_pred_int)
        st.write(cm)


# ---------------- Output table + downloads ----------------
st.subheader("Output Table")
st.dataframe(df_view.head(max_rows_show), use_container_width=True)

csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "⬇️ Download CSV with predictions",
    data=csv_bytes,
    file_name=f"{PREFIX}_{MODEL_NAME}_predictions.csv",
    mime="text/csv",
)

summary = {
    "dataset": PREFIX,
    "dataset_display": DATASET_NAME,
    "model": MODEL_NAME,
    "rows": int(len(df_out)),
    "features_used": int(len(kept_cols)),
    "nan_ratio": float(nan_ratio),
    "scaler_applied": bool(used_scaler),
    "predicted_distribution": {str(int(k)): int(v) for k, v in vc.items()},
}

# add metrics to summary (if available)
if y_true is not None:
    summary["metrics"] = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
    }

summary_bytes = json.dumps(summary, indent=2).encode("utf-8")
st.download_button(
    "⬇️ Download JSON summary",
    data=summary_bytes,
    file_name=f"{PREFIX}_{MODEL_NAME}_summary.json",
    mime="application/json",
)