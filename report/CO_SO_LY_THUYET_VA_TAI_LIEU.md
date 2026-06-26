# Cơ sở lý thuyết & Tài liệu tham khảo — Khóa luận cải tiến CMAR

**Đề tài:** Cải tiến thuật toán CMAR cho phân lớp dữ liệu mất cân bằng
**Ngày:** 20/06/2026

---

## 1. CẤU TRÚC TÀI LIỆU — Phân biệt "Nền tảng" và "Cải tiến"

Khóa luận sử dụng nhiều bài báo, nhưng chúng đóng **2 vai trò hoàn toàn khác nhau**:

```
┌─────────────────────────────────────────────────────────────┐
│  ĐỀ TÀI: Cải tiến CMAR cho dữ liệu mất cân bằng              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   NỀN TẢNG (1 bài)          CÁI ĐƯỢC cải tiến               │
│   └── CMAR (Li, Han, Pei 2001)                              │
│                                                              │
│              ▲ cải tiến bằng ▲                              │
│                                                              │
│   CÁC KỸ THUẬT CẢI TIẾN (5 bài)   CÔNG CỤ để cải tiến       │
│   ├── MWMOTE (Barua 2014)                                   │
│   ├── CCO + AV (MCWCAR, Wu 2024)                            │
│   ├── WAR weighted-support (Wang 2000)                      │
│   ├── Cost-sensitive (King & Zeng 2001)                     │
│   └── Adaptive gating (He & Garcia 2009)                    │
│                                                              │
│   CÔNG CỤ ĐÁNH GIÁ (2 bài)                                  │
│   ├── G-mean (Kubat & Matwin 1997)                          │
│   └── Wilcoxon test (Demšar 2006)                           │
└─────────────────────────────────────────────────────────────┘
```

**Ví dụ dễ hiểu:** Đề tài "Nâng cấp xe Honda" thì *Honda là nền tảng* (cái được nâng cấp), còn *turbo, phun xăng điện tử* là công nghệ thêm vào. Không xếp Honda chung nhóm với turbo.

→ Tương tự: **CMAR là nền tảng** (cái được cải tiến), **5 bài kia là kỹ thuật** lấy ý tưởng để cải tiến CMAR.

---

## 2. NỀN TẢNG — Thuật toán CMAR

**Li, W., Han, J., Pei, J. (2001).** *CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules.* IEEE ICDM 2001, pp. 369–376.
🔗 https://doi.org/10.1109/ICDM.2001.989541

**Là gì:** Thuật toán phân lớp dựa trên luật kết hợp (Associative Classification). Quy trình:
1. Khai phá Class Association Rule (CAR) từ FP-tree.
2. Cắt tỉa luật theo chi-square, correlation, database coverage.
3. Phân lớp bằng weighted chi-square trên nhiều luật mạnh.

**Tại sao là nền tảng:** Đây chính là **đối tượng nghiên cứu** của khóa luận. Toàn bộ cải tiến đều xây dựng TRÊN CMAR — sửa cách nó khai phá luật, vote, và cắt tỉa. CMAR gốc hoạt động tốt trên dữ liệu cân bằng nhưng **yếu trên dữ liệu mất cân bằng** (F1/Recall lớp thiểu số thấp) — đó là vấn đề khóa luận giải quyết.

---

## 3. CÁC KỸ THUẬT CẢI TIẾN (5 bài) — Mỗi bài 1 vai trò

> Mỗi kỹ thuật xử lý mất cân bằng ở MỘT TẦNG khác nhau, không trùng lặp.

### 3.1. Tầng DỮ LIỆU — MWMOTE
**Barua, S., Islam, M.M., Yao, X., Murase, K. (2014).** *MWMOTE — Majority Weighted Minority Oversampling Technique.* IEEE TKDE, 26(2), pp. 405–425.
🔗 https://doi.org/10.1109/TKDE.2012.232
- **Cải tiến:** Sinh thêm mẫu nhân tạo cho lớp cực hiếm (vd lymph có lớp chỉ 2 mẫu).
- **Tăng:** Recall.

### 3.2. Tầng LUẬT — CCO Weighted-Support
**Wu, W., et al. (2024).** *MCWCAR.* Engineering Applications of AI, 129, 107622. (Eq.3)
🔗 https://doi.org/10.1016/j.engappai.2023.107622
+ **Wang, W., Yang, J., Yu, P.S. (2000).** *Efficient mining of weighted association rules (WAR).* ACM SIGKDD 2000, pp. 270–274. 🔗 https://doi.org/10.1145/347090.347149
- **Cải tiến:** Gán trọng số (φ-coefficient) cho item → luật chất lượng cao của lớp thiểu số sống sót qua cắt tỉa.
- **Tăng:** F1, Accuracy.

### 3.3. Tầng VOTE — Added Value
**Wu, W., et al. (2024).** *MCWCAR.* (Eq.7)
- **Cải tiến:** Vote bằng `confidence − P(c)` thay vì confidence thuần → chống thiên vị lớp đa số.
- **Tăng:** F1 + Recall.

### 3.4. Tầng VOTE — Cost-Sensitive
**King, G., Zeng, L. (2001).** *Logistic Regression in Rare Events Data.* Political Analysis, 9(2), pp. 137–163.
🔗 https://doi.org/10.1093/oxfordjournals.pan.a004868
- **Cải tiến:** Boost điểm vote lớp hiếm bằng `(N/count)^β`.
- **Tăng:** Recall.

### 3.5. Tầng KIỂM SOÁT — Adaptive Gating
**He, H., Garcia, E.A. (2009).** *Learning from Imbalanced Data.* IEEE TKDE, 21(9), pp. 1263–1284.
🔗 https://doi.org/10.1109/TKDE.2008.239
- **Cải tiến:** Chỉ bật cải tiến khi imbalance ratio ≥ 2 → không gây hại dataset cân bằng.
- **Bảo vệ:** Accuracy (không giảm).

---

## 4. CÔNG CỤ ĐÁNH GIÁ (2 bài)

### 4.1. G-mean
**Kubat, M., Matwin, S. (1997).** *Addressing the Curse of Imbalanced Training Sets: One-Sided Selection.* ICML 1997, pp. 179–186.
🔗 https://dblp.org/rec/conf/icml/KubatM97.html
- Bài gốc đề xuất G-mean = `(∏ recall_c)^(1/k)` — thước đo chuẩn cho dữ liệu mất cân bằng.

### 4.2. Kiểm định thống kê
**Demšar, J. (2006).** *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR, 7, pp. 1–30.
🔗 https://www.jmlr.org/papers/v7/demsar06a.html
- Wilcoxon signed-rank test — chứng minh cải tiến có ý nghĩa thống kê (p=0.0216).

---

## 5. BẢNG TỔNG HỢP

| # | Bài báo | Năm | Vai trò | Tầng | Cải thiện |
|---|---------|-----|---------|------|-----------|
| [1] | **CMAR** (Li, Han, Pei) | 2001 | **NỀN TẢNG** | — | (cái được cải tiến) |
| [2] | **MCWCAR** (Wu et al.) | 2024 | Cải tiến — CCO + AV | Luật + Vote | F1, Recall |
| [3] | **WAR** (Wang et al.) | 2000 | Cải tiến — weighted-support | Luật | F1 |
| [4] | **MWMOTE** (Barua et al.) | 2014 | Cải tiến — oversampling | Dữ liệu | Recall |
| [5] | **King & Zeng** | 2001 | Cải tiến — cost-sensitive | Vote | Recall |
| [6] | **He & Garcia** | 2009 | Cải tiến — adaptive gating | Kiểm soát | Bảo vệ Accuracy |
| [7] | **Kubat & Matwin** | 1997 | Đánh giá — G-mean | Metric | — |
| [8] | **Demšar** | 2006 | Đánh giá — Wilcoxon | Thống kê | — |

---

## 6. TÓM TẮT (1 câu cho hội đồng)

> **"CMAR (Li 2001) là thuật toán nền tảng được cải tiến. Trên cơ sở đó, khóa luận tích hợp 5 kỹ thuật xử lý mất cân bằng ở 4 tầng: MWMOTE (tầng dữ liệu), CCO weighted-support + Added Value (tầng luật/vote, từ MCWCAR 2024), cost-sensitive (King & Zeng 2001), và adaptive gating (He & Garcia 2009) — đánh giá bằng G-mean và kiểm định Wilcoxon."**

---

> **Lưu ý citation:** Các bài [1][2][3][4][5][6][7][8] đã được xác minh đường dẫn thật trên web/cơ sở dữ liệu học thuật. Bài [2] (MCWCAR) đọc trực tiếp file gốc.
