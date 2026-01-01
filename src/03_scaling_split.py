from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from joblib import dump

BASE = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE / "data" / "processed"
OUT_DIR = BASE / "data" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

def robust_read_csv(path: Path, chunksize: int):
    """
    CSV reader مقاوم برای فایل‌های بزرگ و خط‌های خراب.
    """
    return pd.read_csv(
        path,
        chunksize=chunksize,
        engine="python",          # مقاوم‌تر برای خطوط خراب
        on_bad_lines="skip"       # رد کردن خطوط مشکل‌دار
    )

def prepare_and_split_chunked(
    in_file: Path,
    out_prefix: str,
    scaler_type: str,           # "minmax" or "zscore"
    chunksize: int = 300_000,
    max_rows: int | None = None
):
    print(f"\n[Split+Scale] {in_file.name} -> prefix: {out_prefix} (scaler={scaler_type})")

    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    # 1) خواندن به صورت chunk و جمع‌کردن (با امکان محدودسازی)
    parts = []
    total_rows = 0

    for chunk in robust_read_csv(in_file, chunksize=chunksize):
        # strip columns
        chunk.columns = [c.strip() for c in chunk.columns]

        if "Label" not in chunk.columns:
            continue

        # اگر max_rows تعیین شده، محدود کن
        if max_rows is not None and total_rows >= max_rows:
            break

        if max_rows is not None and (total_rows + len(chunk)) > max_rows:
            chunk = chunk.iloc[: (max_rows - total_rows)]

        parts.append(chunk)
        total_rows += len(chunk)

    if not parts:
        raise ValueError("No valid data read from file (after skipping bad lines).")

    df = pd.concat(parts, ignore_index=True)
    print(f"Loaded rows (after skipping bad lines): {len(df):,}")

    # 2) جدا کردن X/y
    df.columns = [c.strip() for c in df.columns]
    y_raw = df["Label"].astype(str)
    X_df = df.drop(columns=["Label"])

    # 3) تبدیل همه ستون‌ها به عددی
    for c in X_df.columns:
        if X_df[c].dtype == "object":
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # حذف NaN های تولیدشده
    mask = X_df.notna().all(axis=1)
    X_df = X_df.loc[mask]
    y_raw = y_raw.loc[mask]

    # 4) به numpy
    X = X_df.astype(np.float32).to_numpy()

    # 5) Label encode
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # 6) Split stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 7) Scaling
    if scaler_type.lower() == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type.lower() in ["zscore", "standard", "standardscaler"]:
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be 'minmax' or 'zscore'")

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 8) Save
    npz_path = OUT_DIR / f"{out_prefix}_split_scaled.npz"
    np.savez_compressed(
        npz_path,
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train,
        y_test=y_test
    )

    dump(scaler, OUT_DIR / f"{out_prefix}_scaler.joblib")
    dump(le, OUT_DIR / f"{out_prefix}_label_encoder.joblib")

    (OUT_DIR / f"{out_prefix}_columns.txt").write_text(
        "\n".join(X_df.columns.tolist()),
        encoding="utf-8"
    )

    print("Saved:", npz_path.name)
    print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
    print("Classes:", len(le.classes_))
    print("Done ✅")

def main():
    ids2017 = PROCESSED_DIR / "ids2017_clean.csv"
    ids2018 = PROCESSED_DIR / "ids2018_clean.csv"
    ddos2019 = PROCESSED_DIR / "ddos2019_clean_5pct.csv"

    # اگر سیستم ضعیف بود، برای تست می‌تونی max_rows بذاری (مثلاً 2_000_000)
    prepare_and_split_chunked(ids2017, "ids2017", "minmax", chunksize=300_000, max_rows=None)
    prepare_and_split_chunked(ids2018, "ids2018", "minmax", chunksize=300_000, max_rows=None)
    prepare_and_split_chunked(ddos2019, "ddos2019_5pct", "zscore", chunksize=300_000, max_rows=None)

if __name__ == "__main__":
    main()
