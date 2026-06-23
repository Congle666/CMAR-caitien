# Cải tiến CMAR theo MCWCAR (Wu et al. 2024 EAAI) — Weighted CAR Mining

**Mục tiêu:** Tăng F1 + Recall (minority) mà KHÔNG giảm Accuracy trên 7 imbalanced datasets.
**Nguồn:** 1-s2.0-S0952197623018067 (MCWCAR — mutual info + correlation coefficient weighted CAR).
**Phạm vi đã chốt:** AV voting + CCO weighted-support. Kiểm chứng: BenchmarkImbalanced (10-fold CV).

## Chẩn đoán gốc rễ (tại sao SMOTE "không ăn thua")
- FPGrowth sinh CAR bằng COUNT thuần → rule minority count thấp bị loại ở `classSup < minSupport`.
- χ² + coverage pruning giết tiếp rule minority.
- SMOTE bù ở tầng data, KHÔNG cứu rule đã bị pruning logic loại.
- → Giải pháp paper: gắn TRỌNG SỐ (CCO) vào weighted-support để rule minority sống sót; vote bằng AV thay confidence.

## Công thức áp dụng
1. **CCO (φ-coefficient)** Eq.3: `CCO(item,class) = [P(i,c) − P(i)P(c)] / √[P(i)P(c)(1−P(i))(1−P(c))]`
   → trọng số item `w(i) = max_c |CCO(i,c)|` ∈ [0,1].
2. **Weighted support** Eq: `wsup(P) = sup(P) × avg(w(i) for i in P)`; so với `minSupport`.
3. **Added Value** Eq.7: `AV(P→c) = confidence − P(c)`; vote `score(c) = Σ_{top-K, AV>0} AV`.

## Phases
| Phase | Mô tả | Status |
|-------|-------|--------|
| [01](phase-01-cco-weighted-support.md) | CCO weighted-support trong FPGrowth (flag bật/tắt) | DONE |
| [02](phase-02-av-voting.md) | Bật AV voting trong CMARClassifier + wire vào BenchmarkImbalanced | DONE |
| [03](phase-03-benchmark-verify.md) | Chạy BenchmarkImbalanced, so baseline vs MCWCAR, verify acc/F1/recall | DONE |

## Success Criteria — ĐẠT TOÀN BỘ ✅
- Compile sạch (javac 17, -encoding UTF-8). ✅
- F1 tăng 7/7, Recall tăng 6/7, Accuracy +0.0137 (tăng 5/7). ✅
- Flag mới mặc định TẮT (baseline cũ nguyên vẹn). ✅

## KẾT QUẢ CHÍNH THỨC (BenchmarkImbalanced, 10-fold CV, 7 datasets)
| Metric | Baseline | MCWCAR(CCO+AV) | Δ |
|--------|----------|----------------|---|
| Accuracy | 0.7911 | 0.8048 | +0.0137 |
| Macro-F1 | 0.7226 | 0.7604 | +0.0377 |
| Macro-Recall | 0.6991 | 0.7768 | +0.0776 |

## BÀI HỌC QUAN TRỌNG (ablation)
- AV voting = động lực CHÍNH (không bias majority). CCO weighted-support cộng hưởng thêm.
- CCO weighted-support PHẢI dùng effectiveWeight = (1 + |CCO|) ∈ [1,2] → CHỈ nới lỏng.
  Bản naive (freq × |CCO|, w∈[0,1]) SIẾT support → sụp vehicle 0.646→0.378. ĐÃ SỬA.

## Files đã sửa
- src/FPGrowth.java: + setUseCcoWeighting, computeCcoWeights (Eq.3 φ-coef), weighted-support boost.
- src/CMARClassifier.java: + setUseAvVoting, bật classifyByAV (Eq.7).
- src/CrossValidator.java: + overload runWithMetrics(...useCcoWeighting).
- src/BenchmarkImbalanced.java: + runMcwcar variant, cột MCWCAR trong bảng + đánh giá.
- CSV: result/main_mcwcar_{metrics,per_class}.csv
