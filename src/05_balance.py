from pathlib import Path
import numpy as np

from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

BASE = Path(__file__).resolve().parent.parent
FS_DIR = BASE / "data" / "fs"
OUT_DIR = BASE / "data" / "balanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def bincount_summary(y: np.ndarray, top: int = 15) -> str:
    y = np.asarray(y)
    counts = np.bincount(y)
    pairs = [(i, int(c)) for i, c in enumerate(counts) if c > 0]
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:top]
    return ", ".join([f"class{i}={c}" for i, c in pairs])

def stratified_limit_train(X, y, max_rows, min_per_class=10, seed=42):
    """
    محدودسازی train به صورت stratified:
    - از هر کلاس تا حد min_per_class نمونه (اگر موجود باشد)
    - سپس تا سقف max_rows با نمونه‌های تصادفی تکمیل می‌کند
    """
    if max_rows is None or X.shape[0] <= max_rows:
        return X, y

    rng = np.random.default_rng(seed)
    y = np.asarray(y)

    chosen = []

    # حداقل از هر کلاس
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        take = min(len(idx_cls), min_per_class)
        if take > 0:
            chosen.extend(rng.choice(idx_cls, size=take, replace=False))

    chosen = np.array(sorted(set(chosen)))
    remaining_budget = max_rows - len(chosen)

    # تکمیل تا سقف max_rows
    if remaining_budget > 0:
        remaining_idx = np.setdiff1d(np.arange(X.shape[0]), chosen)
        add_size = min(remaining_budget, len(remaining_idx))
        if add_size > 0:
            add = rng.choice(remaining_idx, size=add_size, replace=False)
            chosen = np.concatenate([chosen, add])

    rng.shuffle(chosen)
    return X[chosen], y[chosen]

def choose_k_neighbors(y_train, default_k=5):
    """
    k_neighbors باید < کمترین تعداد نمونه در کلاس‌های train باشد.
    برای SMOTE حداقل 2 نمونه لازم است.
    """
    y_train = np.asarray(y_train)
    counts = np.bincount(y_train)
    positive = counts[counts > 0]

    if len(positive) == 0:
        return 1

    min_count = int(positive.min())

    # اگر min_count = 1 => SMOTE ممکن نیست (ولی با stratified معمولاً رخ نمی‌دهد)
    # اینجا k را طوری انتخاب می‌کنیم که خطا ندهد
    if min_count <= 1:
        return 1

    # k <= min_count-1
    k = min(default_k, min_count - 1)
    k = max(1, k)
    return k

def balance_one(prefix: str, method: str):
    in_path = FS_DIR / f"{prefix}_fs.npz"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing: {in_path}")

    data = np.load(in_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"\n[Balance] {prefix} | method={method}")
    print("Before - train shape:", X_train.shape, "classes:", len(np.unique(y_train)))
    print("Before - train dist:", bincount_summary(y_train))

    # محدودسازی برای سرعت + جلوگیری از کم‌نمونه شدن کلاس‌ها
    MAX_TRAIN_ROWS = {
        "ids2017": 300_000,
        "ids2018": 300_000,
        "ddos2019_5pct": None  # برای ddos معمولاً سبک‌تر است
    }.get(prefix, None)

    X_use, y_use = stratified_limit_train(
        X_train, y_train,
        max_rows=MAX_TRAIN_ROWS,
        min_per_class=10,
        seed=RANDOM_STATE
    )

    print("Using train subset:", X_use.shape, "classes:", len(np.unique(y_use)))
    print("Subset dist:", bincount_summary(y_use))

    # k_neighbors داینامیک برای جلوگیری از ValueError
    k = choose_k_neighbors(y_use, default_k=5)
    print("SMOTE k_neighbors:", k)

    if method.lower() == "smoteenn":
        sampler = SMOTEENN(
            smote=SMOTE(random_state=RANDOM_STATE, k_neighbors=k),
            random_state=RANDOM_STATE
        )
    elif method.lower() == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    else:
        raise ValueError("method must be 'smoteenn' or 'smote'")

    Xb, yb = sampler.fit_resample(X_use, y_use)

    print("After  - train shape:", Xb.shape, "classes:", len(np.unique(yb)))
    print("After  - train dist:", bincount_summary(yb))

    out_path = OUT_DIR / f"{prefix}_balanced.npz"
    np.savez_compressed(
        out_path,
        X_train=Xb,
        y_train=yb,
        X_test=X_test,
        y_test=y_test
    )

    print("Saved:", out_path.name)
    print("Done ✅")

def main():
    # مطابق مقاله
    balance_one("ids2017", method="smoteenn")
    balance_one("ids2018", method="smoteenn")
    balance_one("ddos2019_5pct", method="smote")

if __name__ == "__main__":
    main()
