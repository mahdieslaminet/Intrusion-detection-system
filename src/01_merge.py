from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "merged"
OUT.mkdir(parents=True, exist_ok=True)

def merge_folder(folder: Path, out_file: Path, chunksize: int = 200_000):
    csv_files = sorted([p for p in folder.glob("*.csv")
                        if not p.name.startswith(".~lock")
                        and not p.name.endswith(".tmp")])

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    print(f"\n[Merging] {folder.name} -> {out_file.name}")
    print("Files:", len(csv_files))

    wrote_header = False
    total_rows = 0

    for i, f in enumerate(csv_files, 1):
        print(f"  ({i}/{len(csv_files)}) {f.name}")

        reader = pd.read_csv(f, low_memory=False, chunksize=chunksize)

        for chunk in reader:
            # بعضی فایل‌ها ممکنه ستون‌های اضافی/متفاوت داشته باشن
            # اینجا همون ستون‌های موجود رو می‌نویسیم؛ در مرحله Clean استانداردش می‌کنیم
            chunk.to_csv(out_file, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            total_rows += len(chunk)

    print(f"Done. Total rows written: {total_rows:,}")

if __name__ == "__main__":
    merge_folder(DATA / "ids2017", OUT / "ids2017_merged.csv")
    merge_folder(DATA / "ids2018", OUT / "ids2018_merged.csv")
    merge_folder(DATA / "ddos2019", OUT / "ddos2019_merged.csv")
