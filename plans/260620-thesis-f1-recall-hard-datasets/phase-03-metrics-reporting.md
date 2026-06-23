# Phase 3 — Metric chuẩn imbalanced + báo cáo (G-mean/MCC/bal-acc, win-tie-loss, Wilcoxon)

## Context links
- Plan: [plan.md](plan.md)
- Research: [researcher-02-benchmark-standards.md](research/researcher-02-benchmark-standards.md) (metric, win-tie-loss, Wilcoxon, regression-vs-IR)
- Scout: [scout-01-codebase.md](scout/scout-01-codebase.md) (EvalMetrics CHƯA có G-mean/MCC/bal-acc)

## Overview
- **Date:** 2026-06-20 · **Priority:** CAO · **Status:** TODO
- Cung cấp công cụ đo CHUẨN để (a) đánh giá Phase 1-2 đúng tinh thần imbalanced (accuracy
  một mình gây hiểu lầm), (b) chứng minh trong khóa luận: cải tiến đáng kể trên khó +
  không regression trên cân bằng. PHẦN "thêm metric" nên làm TRƯỚC Phase 1 để đo chính xác.

## Key Insights
1. EvalMetrics đã có per-class TP/FP/FN + macroF1/weightedF1 + average() cộng dồn qua fold.
   → G-mean, MCC, balanced-acc tính TRỰC TIẾP từ TP/FP/FN đã có (DRY, không cần đổi pipeline).
2. Multi-class: G-mean = (∏ recall_c)^(1/k); balanced-acc = mean(recall_c); MCC multi-class
   dùng công thức tổng quát Gorodkin (confusion matrix) — cần ma trận nhầm lẫn KxK, hiện
   chỉ có TP/FP/FN per-class (KHÔNG đủ cho MCC multi-class chính xác). Xem Decision dưới.
3. Win-tie-loss + Wilcoxon là phân tích NGOÀI Java (chạy trên CSV kết quả). Làm script
   Python nhỏ (đã có report/*.py tiền lệ) đọc CSV baseline vs improved.

## Decision (KISS + brutal honest)
- **G-mean, balanced-acc**: thêm vào EvalMetrics (rẻ, chỉ cần recall per-class đã có).
- **MCC**: multi-class MCC cần full confusion matrix. Hai lựa chọn:
  - (A) Thêm confusion matrix KxK vào compute()/average() → MCC chuẩn Gorodkin. Effort MED.
  - (B) Bỏ MCC, dùng G-mean + balanced-acc + macroF1 (researcher-02 chấp nhận
    "G-mean + minority F1" là đủ cho thesis). YAGNI → **mặc định chọn (B)**; chỉ làm (A)
    nếu hội đồng yêu cầu MCC. Ghi rõ trong báo cáo lý do chọn G-mean over MCC (Chicco 2020
    nói MCC mạnh cho BINARY; multi-class G-mean phổ biến hơn trong AC literature).

## Requirements
- R1: EvalMetrics có field `gMean`, `balancedAcc` (+ optional `mcc` nếu chọn (A)).
- R2: compute() và average() điền các field này từ recall per-class đã có.
- R3: Script báo cáo: đọc 2 CSV (baseline, improved) → bảng per-dataset (Acc, macroF1,
  G-mean, recall, minority-F1) nhóm theo IR tier + win-tie-loss + Wilcoxon p + regression
  Δ vs log(IR).
- R4: KHÔNG đổi chữ ký compute()/average() public theo cách phá caller hiện có (chỉ THÊM field).

## Architecture
```
EvalMetrics (THÊM field, không phá API):
  double gMean;        // (∏_c recall_c)^(1/k), bỏ qua class support=0
  double balancedAcc;  // mean(recall_c) over classes có support>0
  [optional] double mcc;  // chỉ nếu Decision (A)
compute(): sau vòng tính P/R/F1, thêm vòng gom recall_c → gMean, balancedAcc.
average(): tính lại từ recall per-class đã cộng dồn (giống cách f1 đang làm).

report/ (Python, ngoài Java):
  analyze_results.py: pandas đọc baseline_metrics.csv + p1_metrics.csv
     → win/tie/loss, scipy.stats.wilcoxon, linregress(Δ ~ log(IR)), bảng markdown.
```

## Related code files
- `src/EvalMetrics.java` — thêm field + tính trong compute() & average().
- `src/BenchmarkImbalanced.java`, `src/Benchmark.java` — in/lưu thêm G-mean, bal-acc vào CSV.
- `report/analyze_results.py` — script mới (theo mẫu report/generate_improvement_csv_and_plots.py).

## Implementation Steps
1. **EvalMetrics field + compute()**:
   ```java
   public double gMean, balancedAcc;
   // trong compute(), sau khi điền cm.recall:
   double prod = 1.0, sumR = 0.0; int k = 0;
   for (ClassMetrics cm : byClass.values()) if (cm.support > 0){
      prod *= cm.recall; sumR += cm.recall; k++;
   }
   m.gMean = k==0 ? 0.0 : Math.pow(prod, 1.0/k);
   m.balancedAcc = k==0 ? 0.0 : sumR/k;
   ```
   (Lưu ý: nếu 1 recall_c=0 → gMean=0; đúng tinh thần "phạt nặng class bị bỏ" — đây là
   GIÁ TRỊ chẩn đoán cho glass class3.)
2. **average()**: tính gMean/balancedAcc từ recall per-class đã cộng dồn (đặt sau vòng f1).
3. **Driver**: thêm cột G-mean, balancedAcc vào CSV xuất; in ra console.
4. **(Optional A)** confusion matrix + MCC nếu cần — tách commit riêng.
5. **report/analyze_results.py**:
   - input: 2 CSV (baseline vs improved) + cột IR.
   - output: bảng markdown nhóm IR tier (low<5 / med 5-20 / high≥20), win-tie-loss
     (tie nếu |Δ macroF1|<0.005), `wilcoxon(improved, baseline)` p-value, `linregress`
     của Δ G-mean theo log(IR) (slope>0 = cải tiến tăng theo độ khó = LÝ TƯỞNG).
6. Chạy với output Phase 1+2 → sinh bảng cuối cho khóa luận.

## Todo list
- [ ] EvalMetrics: gMean + balancedAcc (compute + average)
- [ ] (optional) confusion matrix + MCC — chỉ nếu yêu cầu
- [ ] driver lưu G-mean/bal-acc vào CSV
- [ ] analyze_results.py: win-tie-loss + Wilcoxon + regression-vs-IR + bảng markdown
- [ ] sinh bảng cuối từ kết quả Phase 1+2

## Success Criteria
- EvalMetrics in được G-mean/balanced-acc cho mọi dataset; glass class3 recall=0 → gMean
  phản ánh đúng (thấp) ở baseline, tăng rõ sau Phase 1.
- Bảng khóa luận: 5 imbalanced G-mean tăng vs baseline; win-tie-loss vs CMAR gốc trên 19
  datasets có Win ≥ Loss; Wilcoxon p báo cáo (kể cả nếu p>0.05, báo trung thực + lưu ý
  N nhỏ underpowered — researcher-02).
- regression-vs-IR slope ≥ 0 (cải tiến không tệ đi trên set khó).

## Risk Assessment
- **gMean=0 khi 1 class recall=0**: đúng về lý thuyết nhưng làm trung bình nhiễu. Báo cáo
  CẢ macroF1 (mượt hơn) lẫn gMean. Không che giấu.
- **Wilcoxon underpowered** với 5-19 datasets: nêu rõ giới hạn, không thổi phồng p-value.
- **MCC multi-class sai** nếu cố tính từ TP/FP/FN per-class (không đủ thông tin) → vì vậy
  Decision (B) bỏ MCC trừ khi thêm confusion matrix đầy đủ (A).

## Security Considerations
N/A. Script Python chỉ đọc CSV nội bộ.

## Next steps
→ Dùng bảng này quyết định có cần Phase 4 không: nếu minority recall vẫn thấp / còn
regression → Phase 4. Nếu đạt Success Criteria tổng → kết thúc, viết chương kết quả.
