# Phase 2 — Regression Control (CIR gating thống nhất + per-dataset param)

## Context links
- Plan: [plan.md](plan.md)
- Research: [researcher-01-techniques.md](research/researcher-01-techniques.md) (Technique 4 CIR Adaptive Trigger)
- Scout: [scout-01-codebase.md](scout/scout-01-codebase.md)
- Phụ thuộc: Phase 1 (per-class minSup là cờ MỚI cần đưa vào cùng gate).

## Overview
- **Date:** 2026-06-20 · **Priority:** CAO · **Status:** TODO
- Mục tiêu #2 của khóa luận: dataset cân bằng KHÔNG giảm. Hiện gating chỉ áp cho AV
  voting (CMARClassifier dòng 360-409, dùng `originalImbalanceRatio` vs `adaptiveAvThreshold`).
  Per-class minSup (Phase 1), cost-sensitive β đang KHÔNG nằm dưới gate chung →
  rủi ro làm tụt tic-tac-toe/horse. Phase 2 hợp nhất MỌI cải tiến dưới 1 cổng CIR duy nhất.

## Key Insights
1. Bottleneck regression thật (scout): tic-tac-toe F1 0.970→0.895 (−0.075), horse −0.043
   KHI cải tiến luôn bật. Gating hiện đã chặn AV trên các set này → giữ baseline. Nhưng
   Phase 1 thêm 1 nguồn regression mới (per-class minSup) cần CÙNG gate.
2. CIR = max_class_size / min_class_size, tính 1 lần từ train distribution GỐC (trước SMOTE)
   = `originalImbalanceRatio` đã có sẵn (CMARClassifier dòng 82, set qua setter dòng 135).
   → Tái dùng, KHÔNG thêm field mới (DRY).
3. 5 imbalanced: lymph 40.5, zoo 10.3, glass 8.4, hepatitis 3.8, german 2.3. Ngưỡng CIR
   hiện = 2 (ratio≥2 bật). Vấn đề: german (2.3) sát ngưỡng, tic-tac-toe ~1.4 / horse ~1.7
   nằm dưới → đã an toàn. Giữ ngưỡng=2 nhưng cần xác nhận lại bằng số sau Phase 1.

## Requirements
- R1: MỘT điểm quyết định duy nhất `improvementsEnabled()` = `cirRatio >= cirThreshold`,
  dùng chung cho: AV voting, cost-sensitive β, per-class minSup.
- R2: cirThreshold là field cấu hình (default 2.0, giữ hành vi hiện tại).
- R3: per-dataset override: cho phép tắt cải tiến thủ công cho 1 dataset nếu CIR sát ngưỡng
  mà vẫn regress (escape hatch, dùng tối thiểu — KISS, không over-engineer per-dataset table).
- R4: Khi gate OFF → chạy ĐÚNG baseline CMAR gốc (per-class minSup map = null, AV off,
  β=0). Phải verify Δ = 0 trên dataset cân bằng.

## Architecture
```
CMARClassifier:
  double cirThreshold = 2.0;                       // field mới, configurable
  boolean improvementsEnabled(){
     return originalImbalanceRatio >= cirThreshold; // 1 chân lý duy nhất
  }
  // train(): if (!improvementsEnabled()) → fp.setMinSupByClass(null), không SMOTE-extra
  // classify(): AV + cost-sensitive chỉ khi improvementsEnabled()
```
Gate áp ở 2 nơi: (a) TRAIN (per-class minSup, SMOTE), (b) CLASSIFY (AV, β).
QUAN TRỌNG: CIR phải biết TRƯỚC train → CrossValidator set originalImbalanceRatio sớm.

## Related code files
- `src/CMARClassifier.java` — thêm `cirThreshold` + `improvementsEnabled()`; refactor
  gating dòng 360-409 dùng helper; gate per-class minSup ở train(); gate β ở classifyByAV.
- `src/CrossValidator.java` — đảm bảo `setOriginalImbalanceRatio` gọi TRƯỚC train với
  ratio tính từ distribution GỐC (trước SMOTE).
- `src/BenchmarkImbalanced.java` / `src/Benchmark.java` — verify cả 2 bộ dataset.

## Implementation Steps
1. **Thêm field + helper** trong CMARClassifier:
   ```java
   private double cirThreshold = 2.0;
   public void setCirThreshold(double t){ this.cirThreshold = t; }
   private boolean improvementsEnabled(){ return originalImbalanceRatio >= cirThreshold; }
   ```
2. **Refactor gating dòng 360-409**: thay các check `ratio < adaptiveAvThreshold` rải rác
   bằng `if (!improvementsEnabled()) { fallback baseline; }`. Giữ logic vote cũ khi enabled.
3. **Gate train()**: trước khi gọi `fp.setMinSupByClass(...)` (Phase 1), bọc
   `if (usePerClassMinSup && improvementsEnabled())`. Nếu không → set null.
4. **Gate cost-sensitive** (dòng 483): `if (costSensitiveBeta>0 && improvementsEnabled() && ...)`.
5. **CrossValidator**: tính `cir = maxClassCount/minClassCount` từ trainData GỐC (trước
   SMOTE), `clf.setOriginalImbalanceRatio(cir)` TRƯỚC `clf.train(...)`.
6. **Escape hatch (R3)**: optional `Set<String> disabledDatasets` ở Benchmark driver; nếu
   tên dataset nằm trong set → `clf.setCirThreshold(Double.MAX_VALUE)` (ép baseline).
   Chỉ dùng nếu Step 7 còn regression.
7. **Verify regression**: chạy Benchmark.java (19 datasets). So macroF1 mỗi dataset cân
   bằng (ratio<2) với baseline CMAR gốc. Yêu cầu |Δ| ≤ 0.005. Nếu tic-tac-toe/horse vẫn
   tụt → dùng escape hatch hoặc nâng cirThreshold.

## Todo list
- [ ] field cirThreshold + improvementsEnabled()
- [ ] refactor gating 360-409 qua helper (DRY)
- [ ] gate per-class minSup (Phase 1) + cost-sensitive β dưới cùng helper
- [ ] CrossValidator set originalImbalanceRatio (pre-SMOTE) trước train
- [ ] escape hatch disabledDatasets (chỉ nếu cần)
- [ ] chạy Benchmark 19 datasets, bảng Δ macroF1 vs baseline
- [ ] xác nhận 14 dataset cân bằng |Δ|≤0.005

## Success Criteria
- tic-tac-toe macroF1: bằng baseline CMAR gốc (±0.005), KHÔNG còn −0.075.
- horse macroF1: ±0.005 vs baseline, KHÔNG còn −0.043.
- 14 dataset ratio<2: |Δ macroF1| ≤ 0.005 mỗi cái.
- 5 imbalanced: giữ nguyên gain từ Phase 1 (gate enabled → cải tiến vẫn chạy).

## Risk Assessment
- **CIR sát ngưỡng** (german 2.3): nếu german regress, hạ kỳ vọng hoặc per-dataset off.
  Brutal honest: với ratio 2.3 gain thường nhỏ; chấp nhận tie là OK cho mục tiêu #2.
- **Refactor làm vỡ AV logic hiện đang đúng**: giữ test trước-sau trên 5 imbalanced để
  đảm bảo gain cũ (macroF1 0.751) KHÔNG mất sau refactor. Đây là regression 2 chiều.
- **originalImbalanceRatio chưa set kịp trước train**: nếu quên, gate đọc 0.0 → luôn OFF
  → mất hết cải tiến. Verify bằng assert/log ratio mỗi fold.

## Security Considerations
N/A. Chỉ guard chia 0 khi minClassCount=0 (không xảy ra với stratified CV, nhưng thêm
`Math.max(1,min)`).

## Next steps
→ Phase 3 dùng G-mean/MCC để chứng minh "ổn hơn, đều hơn" + win-tie-loss chứng minh
no-regression bằng số. Nếu sau Phase 2 minority recall vẫn yếu hoặc còn regression →
Phase 4.
