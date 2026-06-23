# Báo cáo cải tiến CMAR — Tóm tắt, thay đổi và so sánh kết quả

Tệp này tóm tắt những cải tiến được hiện thực trong mã nguồn, các tài liệu tham khảo chính, và so sánh kết quả trước/sau cải tiến (Baseline → Full) trên 19 UCI datasets (bảng lấy từ `result/final_*.csv`).

## Những cải tiến chính (tóm tắt)

- **SMOTE / Borderline-SMOTE (data-level balancing)**: áp dụng SMOTE-N (categorical) và Borderline-SMOTE-N để cân bằng lớp thiểu số trước khi mining (tham khảo: Chawla et al., 2002; Han et al., 2005).
- **H4 — Stratified coverage**: trong pruning (database coverage) giữ bảo vệ top‑K luật cho mỗi lớp để đảm bảo class thiểu số có đại diện (giảm tình trạng minority starved).
- **H5 — Lift filter & composite voting (conf × lift)**: sau chi-square pruning có option loại luật có lift < 1; voting có thể dùng trọng số `confidence × lift` (H5b) hoặc Laplace voting (H5c) hoặc ensemble (H5d).
- **H6 — Top‑K voting**: lựa chọn chỉ dùng top‑K luật mạnh nhất per class khi vote.
- **H2 (class-specific minSup)**: framework hỗ trợ nhưng một số pipeline (NoH2) tắt H2 và dùng adaptive SMOTE thay thế.

Các tham khảo được trích trong code: Li, Han & Pei (2001) CMAR; Chawla et al. (2002) SMOTE; Han et al. (2005) Borderline-SMOTE; Brin et al. (1997) (lift); Yin & Han (2003) CPAR; Antonie & Zaïane (2002) (ensemble ideas).

---

## Bảng so sánh (Baseline → Full)

Các cột: `accuracy`, `macroF1`, `macroRecall` (macro-average recall tính bằng trung bình recall từng lớp có support>0). Giá trị hiển thị 4 chữ số thập phân; `Δ` = Full − Baseline.

| Dataset | Baseline Acc | Full Acc | Δ Acc | Baseline MacroF1 | Full MacroF1 | Δ MacroF1 | Baseline MacroRecall | Full MacroRecall | Δ Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast-w | 0.9298 | 0.9298 | 0.0000 | 0.9185 | 0.9185 | 0.0000 | 0.9052 | 0.9052 | 0.0000 |
| cleve | 0.8051 | 0.8051 | 0.0000 | 0.8015 | 0.8015 | 0.0000 | 0.8015 | 0.8015 | 0.0000 |
| crx | 0.8670 | 0.8655 | -0.0015 | 0.8659 | 0.8646 | -0.0013 | 0.8680 | 0.8673 | -0.0007 |
| diabetes | 0.7631 | 0.7631 | 0.0000 | 0.7291 | 0.7291 | 0.0000 | 0.7245 | 0.7245 | 0.0000 |
| german | 0.7320 | 0.7290 | -0.0030 | 0.6440 | 0.6453 | 0.0013 | 0.6372 | 0.6398 | 0.0027 |
| glass | 0.6563 | 0.6915 | +0.0352 | 0.5752 | 0.6544 | +0.0792 | 0.6503 | 0.7270 | +0.0767 |
| heart | 0.8296 | 0.8296 | 0.0000 | 0.8273 | 0.8273 | 0.0000 | 0.8292 | 0.8292 | 0.0000 |
| hepatitis | 0.8186 | 0.7931 | -0.0255 | 0.7282 | 0.7102 | -0.0180 | 0.7591 | 0.7543 | -0.0048 |
| horse | 0.8397 | 0.8397 | 0.0000 | 0.8302 | 0.8302 | 0.0000 | 0.8363 | 0.8363 | 0.0000 |
| iris | 0.9267 | 0.9267 | 0.0000 | 0.9223 | 0.9223 | 0.0000 | 0.9267 | 0.9267 | 0.0000 |
| labor | 0.9467 | 0.9467 | 0.0000 | 0.9457 | 0.9457 | 0.0000 | 0.9595 | 0.9595 | 0.0000 |
| led7 | 0.7259 | 0.7265 | +0.0006 | 0.7170 | 0.7172 | +0.0002 | 0.7238 | 0.7243 | +0.0004 |
| lymph | 0.8058 | 0.8607 | +0.0549 | 0.6864 | 0.8549 | +0.1685 | 0.4209 | 0.8667 | +0.4458 |
| sonar | 0.8071 | 0.8071 | 0.0000 | 0.8041 | 0.8041 | 0.0000 | 0.8061 | 0.8061 | 0.0000 |
| tic-tac-toe | 0.9729 | 0.9729 | 0.0000 | 0.9701 | 0.9701 | 0.0000 | 0.9700 | 0.9700 | 0.0000 |
| vehicle | 0.6465 | 0.6477 | +0.0012 | 0.6042 | 0.6038 | -0.0004 | 0.6497 | 0.6508 | +0.0011 |
| waveform | 0.8390 | 0.8316 | -0.0074 | 0.8383 | 0.8306 | -0.0077 | 0.8387 | 0.8312 | -0.0075 |
| wine | 0.9663 | 0.9663 | 0.0000 | 0.9685 | 0.9685 | 0.0000 | 0.9709 | 0.9709 | 0.0000 |
| zoo | 0.9490 | 0.9650 | +0.0160 | 0.9019 | 0.9273 | +0.0254 | 0.8715 | 0.9286 | +0.0571 |

---

## Nhận xét chính

- Một số dataset có cải thiện lớn rõ rệt: **lymph** (Δ Accuracy +5.49%, Δ MacroF1 +16.85%, Δ MacroRecall +44.6%), **glass** (Δ Acc +3.52%, Δ MacroF1 +7.92%), **zoo** (Δ Acc +1.60%, Δ MacroF1 +2.54%). Những cải thiện này phù hợp kỳ vọng khi dùng adaptive Borderline‑SMOTE và stratified coverage — giúp lớp thiểu số được đại diện tốt hơn trong rule set.
- Một vài dataset có thay đổi nhỏ âm/không đáng kể (ví dụ `waveform`, `german`, `hepatitis`) — điều này là bình thường khi áp các thao tác balancing / voting: trade-off giữa accuracy trên majority và recall/F1 của minority.
- Các thay đổi chủ yếu đến từ hai cơ chế:
  - Data-level balancing (SMOTE / Borderline) — làm tăng khả năng sinh luật cho lớp thiểu số → tăng recall và macro‑F1.
  - Pruning stratified + top‑K voting/conf×lift — giảm thiểu việc loại bỏ luật của lớp thiểu số trong pruning và tập trung vote bằng các luật có correlation mạnh.

## Tài liệu & nguồn tham khảo (trích trong code)

- Li, Han & Pei (2001). CMAR: Accurate and Efficient Classification Based on Multiple Class‑Association Rules (ICDM 2001).
- Chawla et al. (2002). SMOTE: Synthetic Minority Over‑sampling Technique (JAIR).
- Han, Wang & Mao (2005). Borderline‑SMOTE.
- Brin et al. (1997). Lift / association insights.
- Yin & Han (2003). CPAR: Classification based on Predictive Association Rules (SIAM SDM 2003).
- Antonie & Zaïane (2002). Ensemble ideas (referenced in code comments).

---

## Gợi ý cho báo cáo giáo viên / bước tiếp theo

- Tôi có thể mở rộng báo cáo này: thêm biểu đồ (accuracy/F1/recall trước‑sau), trích xuất top‑k luật thay đổi, hoặc tạo bản PDF/PNG.
- Nếu bạn muốn, tôi sẽ tạo `report/IMPROVEMENT_REPORT_FULL.md` (hoặc PNG chart) và file CSV tóm tắt để chèn vào báo cáo giáo viên.

— Kết thúc báo cáo tóm tắt —
