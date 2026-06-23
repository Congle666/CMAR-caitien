# Phase 4 (OPTIONAL) — Class-wise F1-tuning / Ensemble fallback

## Context links
- Plan: [plan.md](plan.md)
- Research: [researcher-01-techniques.md](research/researcher-01-techniques.md) (Technique 5 Ensemble fallback, Technique 6 Auto F1 tuning, Technique 3 OVE)

## Overview
- **Date:** 2026-06-20 · **Priority:** THẤP (gate-kept) · **Status:** TODO (chỉ làm nếu cần)
- **CHỈ thực thi nếu** sau Phase 1-3: (a) minority recall còn < 0.40 trên class mục tiêu,
  HOẶC (b) còn regression không kiểm soát được trên dataset cân bằng. YAGNI: nếu Phase
  1-2 đã đạt Success Criteria tổng → BỎ QUA Phase 4 hoàn toàn.

## Key Insights
1. Ba kỹ thuật ứng viên, chọn 1 theo triệu chứng:
   - **(4a) Class-wise minConf F1-tuning** — nếu precision class hiếm SỤP sau Phase 1
     (nhiều luật yếu). Nâng minConf riêng class khó dựa trên validation F1.
   - **(4b) Ensemble fallback (baseline ⊕ improved)** — nếu vẫn regression trên cân bằng
     dù đã gate. Dùng baseline làm lưới an toàn. Effort CAO (2 classifier).
   - **(4c) One-Versus-Each post-filter** — nếu rule xung đột (1 antecedent fire nhiều class).
2. Ưu tiên 4a (rẻ nhất, fix đúng triệu chứng precision) > 4c > 4b (đắt nhất).

## Requirements
- R1: Mỗi kỹ thuật có cờ riêng, default OFF, nằm dưới CIR gate (Phase 2).
- R2: KHÔNG làm cả 3; chọn TỐI ĐA 1 dựa trên dữ liệu chẩn đoán từ Phase 3.

## Architecture (chỉ phác, chốt khi đến đây)
```
(4a) FPGrowth/CMARClassifier: minConfByClass map (song song minSupByClass Phase 1).
     difficulty_c = 1 - valF1_c; nếu khó → minConf_c *= 1.2 (siết, tăng precision).
(4b) CMARClassifier giữ 2 rule-set; predict: improvementsEnabled()? weighted vote
     (improved 0.6 / baseline 0.4) : baseline. Cần holdout chia trong CrossValidator.
(4c) Sau mine: nếu antecedent → nhiều class, giữ rule có confidence cao nhất nếu gap>0.15.
```

## Related code files
- `src/FPGrowth.java` (4a, 4c), `src/CMARClassifier.java` (4b, 4c), `src/CrossValidator.java`
  (holdout cho 4a/4b).

## Implementation Steps
1. Đọc bảng Phase 3 → xác định triệu chứng → chọn 1 kỹ thuật.
2. (4a) thêm minConfByClass + tính validation F1 trên holdout nội bộ của trainData → siết
   minConf class precision thấp. (4b/4c xem Architecture.)
3. Build + verify lại TOÀN BỘ (5 imbalanced + 19) — không được phá Phase 1-2.

## Todo list
- [ ] Quyết định CÓ cần Phase 4 không (dựa Phase 3) — nếu không, đóng phase
- [ ] Nếu có: chọn 1 trong 4a/4b/4c
- [ ] Implement + verify không regression

## Success Criteria
- Nếu kích hoạt: minority recall mục tiêu ≥ 0.40 HOẶC regression còn lại bị triệt tiêu,
  ĐỒNG THỜI không phá kết quả Phase 1-2.

## Risk Assessment
- 4b nhân đôi chi phí train + phức tạp → chỉ khi thật cần.
- 4a có thể GIẢM recall khi siết minConf → trade-off precision/recall, đo cả hai.
- Over-engineering: rủi ro lớn nhất là làm Phase 4 khi không cần → vi phạm YAGNI.

## Security Considerations
N/A.

## Next steps
→ Hoàn tất → cập nhật plan.md status, viết chương kết quả khóa luận với bảng Phase 3.
