"""
Preprocess raw UCI datasets/ → data_clean/ với format CMAR-ready.

Pipeline cho mỗi file:
  1. Parse với delimiter đúng (comma/space)
  2. Skip ID columns nếu có
  3. Replace missing '?' bằng giá trị riêng "missing"
  4. Discretize continuous columns bằng equal-frequency 5 bins (pandas qcut)
  5. Add header row
  6. Output ra data_clean/ với tên match benchmark expectations

Reference: Liu et al. 1998 (CBA) và Li et al. 2001 (CMAR) — discretize trước khi mining.
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path("datasets")
DST = Path("data_clean")
DST.mkdir(exist_ok=True)

N_BINS = 5

# Per-dataset config:
#   src_file:         tên file trong datasets/
#   out_name:         tên file output trong data_clean/ (match benchmark)
#   delim:            "," hoặc " " hoặc r"\s+"
#   skip_cols:        list column indices (0-based) cần bỏ — vd ID column
#   class_col:        index của class column (-1 = last column)
#   col_names:        list tên cột (sẽ làm header). None → tự generate "col0,col1,..."
#   merge_with:       tên file thứ 2 để concat (cho train/test split datasets)
CONFIGS = [
    {
        "src": "breast-cancer-wisconsin.csv", "out": "breast-w.csv",
        "delim": ",", "skip_cols": [0], "class_col": -1,
        "col_names": ["clump_thickness","cell_size","cell_shape","marginal_adh",
                      "single_epith","bare_nuclei","bland_chrom","normal_nucleoli",
                      "mitoses"],
    },
    {
        "src": "processed.cleveland.csv", "out": "cleve.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                      "exang","oldpeak","slope","ca","thal"],
        # Paper CMAR 2001 dùng binary classification: 0→healthy, 1-4→sick
        "binarize_class": {"0": "healthy", "1": "sick", "2": "sick", "3": "sick", "4": "sick"},
    },
    {
        "src": "crx.csv", "out": "crx.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(15)],
    },
    {
        "src": "diabetes.csv", "out": "diabetes.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": ["pregnant","glucose","bp","skin","insulin","bmi","pedigree","age"],
    },
    {
        "src": "german.csv", "out": "german.csv",
        "delim": r"\s+", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(20)],
    },
    {
        "src": "glass.csv", "out": "glass.csv",
        "delim": ",", "skip_cols": [0], "class_col": -1,
        "col_names": ["RI","Na","Mg","Al","Si","K","Ca","Ba","Fe"],
    },
    {
        "src": "heart.csv", "out": "heart.csv",
        "delim": r"\s+", "skip_cols": [], "class_col": -1,
        "col_names": ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                      "exang","oldpeak","slope","ca","thal"],
    },
    {
        "src": "hepatitis.csv", "out": "hepatitis.csv",
        "delim": ",", "skip_cols": [], "class_col": 0,  # CLASS IS FIRST COL
        "col_names": ["age","sex","steroid","antivirals","fatigue","malaise","anorexia",
                      "liver_big","liver_firm","spleen_palp","spiders","ascites","varices",
                      "bilirubin","alk_phos","sgot","albumin","protime","histology"],
    },
    {
        "src": "horse-colic.csv", "out": "horse.csv",
        "delim": r"\s+", "skip_cols": [2], "class_col": 22,  # outcome (alive/died/euth)
        "merge_with": "horse-colic.test.csv",
        "col_names": [f"a{i}" for i in range(27)],
    },
    {
        "src": "iris.csv", "out": "iris.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": ["sepal_len","sepal_wid","petal_len","petal_wid"],
    },
    {
        "src": "labor-neg.csv", "out": "labor.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(16)],
    },
    {
        "src": "led7.csv", "out": "led7.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"led{i}" for i in range(7)],
    },
    {
        "src": "lymphography.csv", "out": "lymph.csv",
        "delim": ",", "skip_cols": [], "class_col": 0,  # CLASS IS FIRST COL in lymph UCI
        "col_names": ["lymphatics","block_affere","bl_lymph_c","bl_lymph_s","by_pass",
                      "extravasates","regen","early_uptake","lym_nodes_dim","lym_nodes_enl",
                      "change_lym","defect_in_node","change_in_node","change_in_stru",
                      "special_forms","dislocation","exclusion","no_of_nodes"],
    },
    {
        "src": "sonar.csv", "out": "sonar.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"f{i}" for i in range(60)],
    },
    {
        "src": "tic-tac-toe.csv", "out": "tic-tac-toe.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"pos{i}" for i in range(9)],
    },
    {
        "src": "vehicle.csv", "out": "vehicle.csv",
        "delim": r"\s+", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(18)],
    },
    {
        "src": "waveform.csv", "out": "waveform.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(21)],
    },
    {
        "src": "wine.csv", "out": "wine.csv",
        "delim": ",", "skip_cols": [], "class_col": 0,  # CLASS IS FIRST COL in wine UCI
        "col_names": ["alcohol","malic","ash","alcal","mg","phenols","flav","nonflav",
                      "proantho","color","hue","od280","proline"],
    },
    {
        "src": "zoo.csv", "out": "zoo.csv",
        "delim": ",", "skip_cols": [0], "class_col": -1,  # skip animal name
        "col_names": ["hair","feathers","eggs","milk","airborne","aquatic","predator",
                      "toothed","backbone","breathes","venomous","fins","legs","tail",
                      "domestic","catsize"],
    },
    # BONUS datasets có thể thêm
    {
        "src": "australian.csv", "out": "australian.csv",
        "delim": r"\s+", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(14)],
    },
    {
        "src": "ionosphere.csv", "out": "ionosphere.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"f{i}" for i in range(34)],
    },
    {
        "src": "anneal.csv", "out": "anneal.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "merge_with": "anneal.test.csv",
        "col_names": [f"a{i}" for i in range(38)],
    },
    {
        "src": "hypothyroid.csv", "out": "hypothyroid.csv",
        "delim": ",", "skip_cols": [], "class_col": 0,  # CLASS IS FIRST COL
        "col_names": [f"a{i}" for i in range(25)],
    },
    {
        "src": "sick.csv", "out": "sick.csv",
        "delim": ",", "skip_cols": [], "class_col": -1,
        "col_names": [f"a{i}" for i in range(29)],
    },
]


def is_continuous(series):
    """Coi là continuous nếu hầu hết giá trị là numeric HOẶC có >5 unique values.
    Threshold 5 (chứ không phải 15) để match paper CMAR 2001 preprocessing —
    breast-w có 1-10 integer codes (10 unique) nên cần discretize."""
    s = series.replace('?', np.nan).dropna()
    if len(s) == 0:
        return False
    # Thử convert sang numeric
    nums = pd.to_numeric(s, errors='coerce')
    if nums.isna().sum() > len(s) * 0.1:  # >10% không phải số → categorical
        return False
    # Là numeric — check unique count. > 5 để discretize cả integer codes.
    return nums.nunique() > 5


def process_one(cfg):
    src_path = SRC / cfg["src"]
    if not src_path.exists():
        print(f"  ⚠ SKIP {cfg['out']} — source not found: {src_path}")
        return False, None

    # Đọc với delimiter đúng
    delim = cfg["delim"]
    df = pd.read_csv(src_path, sep=delim, header=None, na_values=['?'],
                     skipinitialspace=True, engine='python')

    # Merge thêm test file nếu có (anneal, horse-colic)
    if "merge_with" in cfg:
        merge_path = SRC / cfg["merge_with"]
        if merge_path.exists():
            df_test = pd.read_csv(merge_path, sep=delim, header=None, na_values=['?'],
                                  skipinitialspace=True, engine='python')
            df = pd.concat([df, df_test], ignore_index=True)

    # Drop skip_cols
    if cfg["skip_cols"]:
        df = df.drop(columns=cfg["skip_cols"]).reset_index(drop=True)
        df.columns = range(len(df.columns))  # reset column indices

    # Tách class column
    class_idx = cfg["class_col"]
    if class_idx == -1:
        class_idx = len(df.columns) - 1
    class_col_name = df.columns[class_idx]
    y = df[class_col_name]
    X = df.drop(columns=[class_col_name])
    X.columns = range(len(X.columns))  # reset

    # Discretize continuous columns
    n_disc = 0
    for col in X.columns:
        if is_continuous(X[col]):
            try:
                X[col] = pd.qcut(pd.to_numeric(X[col], errors='coerce'),
                                 q=N_BINS, duplicates='drop',
                                 labels=[f"bin{i}" for i in range(N_BINS)])
                X[col] = X[col].astype(str)  # convert categorical → string
                n_disc += 1
            except Exception:
                # Fallback: equal-width
                X[col] = pd.cut(pd.to_numeric(X[col], errors='coerce'),
                                bins=N_BINS,
                                labels=[f"bin{i}" for i in range(N_BINS)])
                X[col] = X[col].astype(str)
                n_disc += 1

    # Replace NaN / nan → "missing"
    X = X.replace(['nan', 'NaN'], 'missing').fillna('missing')
    y = y.replace(['nan', 'NaN', '?'], 'missing').fillna('missing').astype(str)

    # Strip leading/trailing whitespace
    X = X.apply(lambda c: c.astype(str).str.strip())
    y = y.astype(str).str.strip()

    # Cleanup class column: strip record IDs (e.g., "negative.|3733" → "negative")
    # và remove surrounding quotes (e.g., "'good'" → "good")
    y = y.str.replace(r'\.\|\d+$', '', regex=True)
    y = y.str.replace(r"^'|'$", '', regex=True)

    # Binarize / remap class nếu config có (vd. cleve 0→healthy, 1-4→sick để match paper)
    if "binarize_class" in cfg:
        y = y.map(cfg["binarize_class"]).fillna(y)

    # Tạo header — col_names provided + "class"
    col_names = cfg["col_names"]
    if len(col_names) != len(X.columns):
        # Auto-generate nếu mismatch
        col_names = [f"col{i}" for i in range(len(X.columns))]

    # Build output dataframe: X + class column
    out_df = X.copy()
    out_df.columns = col_names
    out_df["class"] = y.values

    # Write CSV
    out_path = DST / cfg["out"]
    out_df.to_csv(out_path, index=False)
    print(f"  ✓ {cfg['out']:25s} | {len(out_df)} rows | {len(col_names)} attrs | "
          f"{n_disc} discretized | classes: {sorted(set(y))[:5]}")
    return True, len(out_df)


def main():
    print(f"=== Preprocess UCI datasets → {DST}/ ===")
    print(f"Discretization: equal-frequency {N_BINS} bins (Li et al. 2001 CMAR)\n")
    n_ok = 0
    n_fail = 0
    total_rows = 0
    for cfg in CONFIGS:
        ok, rows = process_one(cfg)
        if ok:
            n_ok += 1
            total_rows += rows
        else:
            n_fail += 1
    print(f"\n=== DONE: {n_ok} files preprocessed, {n_fail} skipped, {total_rows} total rows ===")


if __name__ == "__main__":
    main()
