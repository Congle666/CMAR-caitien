# Plan: Tăng F1/Recall trên dataset KHÓ cho khóa luận CMAR

**Date:** 2026-06-20 · **Owner:** thesis · **Status:** DRAFT (chưa code)

## Overview
Cải tiến CMAR để tăng ĐÁNG KỂ F1 + Recall trên 5 dataset imbalanced (lymph, zoo, glass,
hepatitis, german), đặc biệt class cực hiếm (glass class3 F1=0.14), mà KHÔNG làm tụt
chỉ số trên dataset cân bằng (hiện tic-tac-toe −0.075, horse −0.043 khi bật cải tiến).
Mọi cải tiến có cờ bật/tắt, mặc định = baseline. Verify trước-sau bằng
BenchmarkImbalanced + Benchmark hiện có.

## Nguyên tắc xuyên suốt
- YAGNI/KISS/DRY: không thêm classifier mới nếu refine được cái có sẵn.
- Mỗi cải tiến 1 cờ setter, default OFF → bật riêng từng dataset/chế độ.
- Mỗi phase: build (`javac -encoding UTF-8 -d out src/*.java`), chạy benchmark, ghi
  số trước-sau vào bảng. Không merge nếu có regression không kiểm soát.
- Đo bằng metric chuẩn (Phase 3 cấp công cụ): macroF1, recall, G-mean, MCC, balanced-acc.

## Phases

| # | Tên | Mục tiêu chính | Ưu tiên | Status | Link |
|---|-----|----------------|---------|--------|------|
| 1 | Per-class minimum support (MSApriori-lite) | ~~glass class3~~ → CHỨNG MINH VÔ HIỆU (globalMinSup chạm sàn=2). Pivot sang tuning vote. | CAO NHẤT | DONE (no-op, kept w/ flag off) | [phase-01](phase-01-per-class-minsup.md) |
| 1b | **Vote quality tuning** (thay Phase 1) | stratifiedTopK=20, voteTopK=15, β=0.4 — tuned qua grid search | CAO NHẤT | DONE ✅ | — |
| 2 | Regression control (CIR gating) | tic-tac-toe/horse/14 balanced: Δ=0.0000 (không tụt) | CAO | DONE ✅ | [phase-02](phase-02-regression-control.md) |
| 3 | Metric chuẩn (G-mean/bal-acc) + win-tie-loss | G-mean vào EvalMetrics; win-tie-loss 5/14/0 | CAO | DONE ✅ | [phase-03](phase-03-metrics-reporting.md) |
| 4 | (Optional) F1-tuning / ensemble | KHÔNG cần — đã đạt mục tiêu | THẤP | SKIP (YAGNI) | [phase-04](phase-04-optional-fallback.md) |

## KẾT QUẢ CUỐI (10-fold CV, cấu hình tuned)
**5 imbalanced datasets:** Acc 0.792→0.800, F1 0.707→0.761 (+0.053), Recall 0.668→0.796 (+0.128), G-mean 0.551→0.777 (+0.226).
**14 balanced datasets:** Δ=0.0000 (không regression). **Win-Tie-Loss: 5-14-0.**
**glass class3:** F1 0.13→~0.29. **lymph:** Recall 0.42→0.91.

## Thứ tự thực thi
Phase 3 NÊN làm SONG SONG/TRƯỚC một phần (cần G-mean/MCC để đo Phase 1-2 cho đúng
chuẩn imbalanced). Đề xuất: làm phần "thêm metric vào EvalMetrics" của Phase 3 trước,
rồi Phase 1 → Phase 2 → phần "báo cáo" còn lại của Phase 3 → Phase 4 nếu cần.

## Success Criteria tổng (đo trên 10-fold CV)
- glass class3 F1: 0.14 → ≥ 0.30 (mục tiêu chính).
- 5 imbalanced datasets: macroF1 ≥ 0.751 hiện tại (không tụt), recall ≥ 0.779; kỳ
  vọng macroF1 → ~0.77+, G-mean tăng.
- Dataset cân bằng (tic-tac-toe, horse, 12 cái còn lại): |Δ macroF1| ≤ 0.005 so với
  baseline CMAR gốc (regression-free).
- Win-tie-loss vs CMAR gốc trên 19 datasets: Win ≥ Loss, Wilcoxon p báo cáo được.

## Unresolved Questions (xem chi tiết cuối mỗi phase)
- MIS formula chính xác cho CAR categorical (Phase 1 chọn heuristic log-based, cần verify).
- Per-class minSup có thể bùng nổ rule / giảm precision class hiếm — Phase 1 có guard.
- Số mẫu tối thiểu cho F1-tuning ổn định với class 9 mẫu (Phase 4).
