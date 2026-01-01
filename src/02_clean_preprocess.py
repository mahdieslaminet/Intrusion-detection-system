from pathlib import Path
import pandas as pd
import numpy as np

# ریشه پروژه
BASE = Path(__file__).resolve().parent.parent

# ورودی‌ها (خروجی مرحله merge)
MERGED_DIR = BASE / "data" / "merged"

# خروجی‌ها (دیتای تمیز)
PROCESSED_DIR = BASE / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ستون‌های رایج که باعث leakage/overfitting یا بی‌فایده هستند (مطابق رویکرد مقاله)
DROP_COLS_CANDIDATES = [
    "Flow ID", "Source IP", "Destination IP", "Timestamp",
    "SimillarHTTP", "SimilarHTTP", "Fwd Header Length.1"
]

def clean_dataset(
    in_file: Path,
    out_file: Path,
    sample_frac: float | None = None,
    chunksize: int = 300_000,
):
    """
    پاکسازی مرحله 2 مقاله:
    - strip کردن نام ستون‌ها (حل مشکل ' Label' و ... )
    - حذف ستون‌های شناسه‌ای/لیکج
    - تبدیل inf به NaN و حذف NaN
    - حذف ستون‌های ثابت (no-variance)
    - (اختیاری) نمونه‌گیری تصادفی برای دیتاست‌های خیلی بزرگ (مثل ddos2019)
    """
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")

    # اگر قبلاً خروجی وجود دارد، پاکش کن تا append خراب نشود
    if out_file.exists():
        out_file.unlink()

    print(f"\n[Clean] {in_file.name} -> {out_file.name}")
    first_write = True

    total_in = 0
    total_out = 0

    for chunk in pd.read_csv(in_file, low_memory=False, chunksize=chunksize):
        total_in += len(chunk)

        # ✅ حل مشکل فاصله‌های اول/آخر نام ستون‌ها (مثل ' Label')
        chunk.columns = [c.strip() for c in chunk.columns]

        # (اختیاری) نمونه‌گیری برای کاهش حجم
        if sample_frac is not None:
            chunk = chunk.sample(frac=sample_frac, random_state=RANDOM_STATE)

        # حذف ستون‌های لیکج/اضافی اگر وجود دارند
        drop_cols = [c for c in DROP_COLS_CANDIDATES if c in chunk.columns]
        if drop_cols:
            chunk.drop(columns=drop_cols, inplace=True, errors="ignore")

        # Infinity -> NaN
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

        # حذف ردیف‌های ناقص
        chunk.dropna(inplace=True)

        # حذف ستون‌های ثابت (هیچ تنوعی ندارند)
        nunique = chunk.nunique(dropna=False)
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            # Label را اشتباهی حذف نکن
            const_cols = [c for c in const_cols if c != "Label"]
            if const_cols:
                chunk.drop(columns=const_cols, inplace=True, errors="ignore")

        # چک وجود Label
        if "Label" not in chunk.columns:
            raise ValueError(
                "Column 'Label' not found after stripping column names. "
                f"Available columns: {chunk.columns.tolist()}"
            )

        # ذخیره خروجی
        chunk.to_csv(out_file, mode="a", index=False, header=first_write)
        first_write = False

        total_out += len(chunk)

    print(f"Rows in : {total_in:,}")
    print(f"Rows out: {total_out:,}")
    print("Done ✅")

def main():
    ids2017_in = MERGED_DIR / "ids2017_merged.csv"
    ids2018_in = MERGED_DIR / "ids2018_merged.csv"
    ddos2019_in = MERGED_DIR / "ddos2019_merged.csv"

    ids2017_out = PROCESSED_DIR / "ids2017_clean.csv"
    ids2018_out = PROCESSED_DIR / "ids2018_clean.csv"
    ddos2019_out = PROCESSED_DIR / "ddos2019_clean_5pct.csv"

    # 2017 و 2018 بدون sampling
    clean_dataset(ids2017_in, ids2017_out, sample_frac=None)
    clean_dataset(ids2018_in, ids2018_out, sample_frac=None)

    # 2019 خیلی بزرگه → طبق مقاله 5% نمونه‌گیری
    clean_dataset(ddos2019_in, ddos2019_out, sample_frac=0.05)

if __name__ == "__main__":
    main()
