#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân tích thống kê kết quả CMAR cải tiến vs baseline (cho khóa luận).
- Win-Tie-Loss table
- Wilcoxon signed-rank test (Demšar 2006) cho macroF1, recall, accuracy
- Regression: ΔF1 ~ log(imbalance ratio) (cải tiến tăng theo độ khó = lý tưởng)

Đọc: result/thesis_baseline_metrics.csv + result/thesis_mcwcar_metrics.csv
Chạy: py report/wilcoxon_test.py
"""
import sys, math
import pandas as pd
from scipy import stats

BASE = "result/thesis_baseline_metrics.csv"
IMPR = "result/thesis_mcwcar_metrics.csv"

# imbalance ratio thật (max_class/min_class) từ phân tích trước
RATIO = {
    "lymph":40.5,"zoo":10.3,"glass":8.4,"hepatitis":3.8,"german":2.3,
    "breast-w":1.9,"tic-tac-toe":1.9,"diabetes":1.9,"labor":1.9,"horse":1.7,
    "wine":1.5,"heart":1.3,"crx":1.2,"cleve":1.2,"sonar":1.1,"vehicle":1.1,
    "led7":1.1,"waveform":1.0,"iris":1.0,
}
TIE = 0.005  # ngưỡng |Δ| coi là tie

def load(p):
    df = pd.read_csv(p).set_index("dataset")
    return df

def main():
    b = load(BASE); m = load(IMPR)
    ds = [d for d in b.index if d in m.index]
    print("="*70)
    print("PHÂN TÍCH THỐNG KÊ — CMAR cải tiến vs Baseline ({} datasets)".format(len(ds)))
    print("="*70)

    for metric in ["macroF1","accuracy"]:
        bv = b.loc[ds, metric].values
        mv = m.loc[ds, metric].values
        diff = mv - bv
        win  = int((diff >  TIE).sum())
        tie  = int((abs(diff) <= TIE).sum())
        loss = int((diff < -TIE).sum())
        # Wilcoxon: bỏ cặp bằng nhau (zero diff) — dùng 'wilcox' zero_method
        try:
            stat, p = stats.wilcoxon(mv, bv, zero_method="wilcox", alternative="greater")
            pstr = f"{p:.4f}"
        except ValueError as e:
            stat, pstr = float("nan"), f"N/A ({e})"
        print(f"\n--- {metric} ---")
        print(f"  Win/Tie/Loss : {win} / {tie} / {loss}")
        print(f"  Mean Δ       : {diff.mean():+.4f}")
        print(f"  Wilcoxon (H1: improved > baseline, one-sided): p = {pstr}")
        if isinstance(pstr,str) and pstr.replace('.','').isdigit():
            print(f"  → {'CÓ Ý NGHĨA thống kê (p<0.05)' if float(pstr)<0.05 else 'CHƯA đạt p<0.05 (N nhỏ → underpowered, Demšar 2006)'}")

    # Chỉ nhóm imbalanced (ratio>=2)
    imb = [d for d in ds if RATIO.get(d,1) >= 2.0]
    print("\n" + "="*70)
    print(f"CHỈ NHÓM IMBALANCED (ratio≥2, n={len(imb)})")
    print("="*70)
    for metric in ["macroF1","accuracy"]:
        bv = b.loc[imb, metric].values; mv = m.loc[imb, metric].values
        diff = mv - bv
        try:
            _, p = stats.wilcoxon(mv, bv, alternative="greater")
            pstr=f"{p:.4f}"
        except ValueError as e:
            pstr=f"N/A ({e})"
        print(f"  {metric:9s}: meanΔ={diff.mean():+.4f}  Wilcoxon p={pstr}")

    # Regression ΔF1 ~ log(IR)
    print("\n" + "="*70)
    print("REGRESSION: ΔmacroF1 ~ log(imbalance ratio)")
    print("="*70)
    xs = [math.log(RATIO.get(d,1)) for d in ds]
    ys = [m.loc[d,"macroF1"] - b.loc[d,"macroF1"] for d in ds]
    sl, ic, r, p, se = stats.linregress(xs, ys)
    print(f"  slope={sl:+.4f} (slope>0 = cải tiến TĂNG theo độ khó = LÝ TƯỞNG)")
    print(f"  r={r:.3f}  p={p:.4f}")
    print(f"  → {'XÁC NHẬN: cải tiến nhắm đúng dataset khó' if sl>0 else 'slope≤0'}")

    # Bảng markdown per-dataset
    print("\n" + "="*70)
    print("BẢNG per-dataset (markdown, copy vào khóa luận)")
    print("="*70)
    print("| Dataset | IR | Base F1 | Impr F1 | ΔF1 | Kết quả |")
    print("|---------|---:|--------:|--------:|----:|---------|")
    for d in sorted(ds, key=lambda x:-RATIO.get(x,1)):
        bf=b.loc[d,"macroF1"]; mf=m.loc[d,"macroF1"]; df=mf-bf
        verdict = "Win" if df>TIE else ("Loss" if df<-TIE else "Tie")
        print(f"| {d} | {RATIO.get(d,1):.1f} | {bf:.3f} | {mf:.3f} | {df:+.3f} | {verdict} |")

if __name__ == "__main__":
    main()
