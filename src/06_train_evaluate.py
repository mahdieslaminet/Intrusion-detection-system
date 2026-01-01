from pathlib import Path
import json
import numpy as np

# ✅ مهم: جلوگیری از خطای tkinter روی ویندوز/ترمینال
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# Optional models
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False


BASE = Path(__file__).resolve().parent.parent
BAL_DIR = BASE / "data" / "balanced"

OUT_MODELS = BASE / "outputs" / "models"
OUT_REPORTS = BASE / "outputs" / "reports"
OUT_PLOTS = BASE / "outputs" / "plots"
OUT_MODELS.mkdir(parents=True, exist_ok=True)
OUT_REPORTS.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def save_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_path, dpi=200)
    plt.close()

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    rep = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": float(acc),
        "precision_weighted": float(pr),
        "recall_weighted": float(rc),
        "f1_weighted": float(f1),
        "report_dict": rep,
        "confusion_matrix": cm.tolist(),
    }

def get_models(num_classes: int):
    models = {}

    # Random Forest
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None
    )

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # LightGBM
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    # CatBoost
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=800,
            learning_rate=0.05,
            depth=8,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False
        )

    return models

def run_dataset(prefix: str):
    npz_path = BAL_DIR / f"{prefix}_balanced.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing balanced file: {npz_path}")

    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    num_classes = len(np.unique(y_train))
    print(f"\n[Dataset] {prefix}")
    print("Train:", X_train.shape, "Test:", X_test.shape, "Classes:", num_classes)

    models = get_models(num_classes)
    results = {}

    for name, model in models.items():
        print(f"  -> Training: {name}")
        model.fit(X_train, y_train)

        res = evaluate_model(model, X_test, y_test)
        results[name] = {
            "accuracy": res["accuracy"],
            "precision_weighted": res["precision_weighted"],
            "recall_weighted": res["recall_weighted"],
            "f1_weighted": res["f1_weighted"],
        }

        # ذخیره report
        report_path = OUT_REPORTS / f"{prefix}_{name}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(res["report_dict"], f, ensure_ascii=False, indent=2)

        # ذخیره confusion matrix
        cm = np.array(res["confusion_matrix"])
        cm_path = OUT_PLOTS / f"{prefix}_{name}_cm.png"
        save_confusion_matrix(cm, f"{prefix} - {name} Confusion Matrix", cm_path)

        # ذخیره مدل
        model_path = OUT_MODELS / f"{prefix}_{name}.joblib"
        dump(model, model_path)

        print(f"     acc={res['accuracy']:.4f} f1_w={res['f1_weighted']:.4f} | saved model+report+cm")

    # خلاصه
    summary_path = OUT_REPORTS / f"{prefix}_SUMMARY.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Saved summary:", summary_path.name)

def main():
    # ✅ IDS2017 قبلاً انجام شده، پس فعلاً کامنتش کردیم
    # run_dataset("ids2017")

    # ✅ ادامه پروژه
    run_dataset("ids2018")
    run_dataset("ddos2019_5pct")

if __name__ == "__main__":
    main()
