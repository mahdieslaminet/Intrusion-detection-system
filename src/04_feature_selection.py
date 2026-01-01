from pathlib import Path
import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

BASE = Path(__file__).resolve().parent.parent
SPLITS_DIR = BASE / "data" / "splits"
OUT_DIR = BASE / "data" / "fs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 250

def load_columns(columns_path: Path) -> list[str]:
    cols = columns_path.read_text(encoding="utf-8").splitlines()
    cols = [c.strip() for c in cols if c.strip()]
    return cols

def feature_select_one(
    prefix: str,
    importance_threshold: float,
    max_train_rows: int | None = 300_000,  # برای سرعت؛ None یعنی کل Train
):
    """
    - NPZ (X_train, X_test, y_train, y_test) را می‌خواند
    - RFR (250 trees) را فقط روی Train fit می‌کند
    - فیچرهای با importance >= threshold را نگه می‌دارد
    - خروجی NPZ جدید + mask + لیست فیچرها + importances را ذخیره می‌کند
    """
    npz_path = SPLITS_DIR / f"{prefix}_split_scaled.npz"
    cols_path = SPLITS_DIR / f"{prefix}_columns.txt"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Missing: {cols_path}")

    data = np.load(npz_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    cols = load_columns(cols_path)
    if len(cols) != X_train.shape[1]:
        raise ValueError(
            f"Columns count ({len(cols)}) != X features ({X_train.shape[1]}) for {prefix}"
        )

    print(f"\n[FS] {prefix}")
    print("Train:", X_train.shape, "Test:", X_test.shape, "Features:", X_train.shape[1])
    print("Threshold:", importance_threshold)

    # برای سرعت، از train یک نمونه می‌گیریم (فقط برای fit RFR)
    if max_train_rows is not None and X_train.shape[0] > max_train_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(X_train.shape[0], size=max_train_rows, replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
        print(f"Using sampled train for RFR fit: {max_train_rows:,} rows")
    else:
        X_fit = X_train
        y_fit = y_train

    # RandomForestRegressor طبق مقاله (y عددی است)
    rfr = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rfr.fit(X_fit, y_fit)

    importances = rfr.feature_importances_
    if importances.shape[0] != X_train.shape[1]:
        raise RuntimeError("Importances length mismatch.")

    mask = importances >= importance_threshold
    kept = int(mask.sum())

    # اگر آستانه خیلی سخت بود و چیزی باقی نماند، حداقل top-10 را نگه داریم
    if kept == 0:
        topk = min(10, len(importances))
        top_idx = np.argsort(importances)[-topk:]
        mask[top_idx] = True
        kept = int(mask.sum())
        print(f"⚠️ Threshold too high; keeping top-{topk} features instead.")

    kept_cols = [c for c, m in zip(cols, mask) if m]

    X_train_fs = X_train[:, mask]
    X_test_fs = X_test[:, mask]

    # ذخیره
    out_npz = OUT_DIR / f"{prefix}_fs.npz"
    np.savez_compressed(
        out_npz,
        X_train=X_train_fs,
        X_test=X_test_fs,
        y_train=y_train,
        y_test=y_test
    )

    # ذخیره mask و importances و ستون‌ها
    np.save(OUT_DIR / f"{prefix}_mask.npy", mask)
    np.save(OUT_DIR / f"{prefix}_importances.npy", importances)

    (OUT_DIR / f"{prefix}_kept_columns.txt").write_text(
        "\n".join(kept_cols),
        encoding="utf-8"
    )

    dump(rfr, OUT_DIR / f"{prefix}_rfr_250.joblib")

    print("Kept features:", kept, "/", len(cols))
    print("Saved:", out_npz.name)
    print("Done ✅")

def main():
    # مطابق مقاله: آستانه‌ها دیتاست‌محورند.
    # (اگر خواستی دقیقاً مثل مقاله تنظیم کنیم، همین‌ها رو نگه دار. اگر خیلی کم/زیاد شد، بعداً tune می‌کنیم.)
    thresholds = {
        "ids2017": 0.015,
        "ids2018": 0.019,
        "ddos2019_5pct": 0.0016,
    }

    # اگر سیستم قوی داری و می‌خوای دقیق‌تر: max_train_rows=None
    feature_select_one("ids2017", thresholds["ids2017"], max_train_rows=300_000)
    feature_select_one("ids2018", thresholds["ids2018"], max_train_rows=300_000)
    feature_select_one("ddos2019_5pct", thresholds["ddos2019_5pct"], max_train_rows=300_000)

if __name__ == "__main__":
    main()
