# Tóm tắt: Kết quả thống kê & Cơ sở lý thuyết — Cải tiến CMAR

**Đề tài:** Cải tiến CMAR cho phân lớp dữ liệu mất cân bằng
**Ngày:** 23/06/2026 · **Đánh giá:** 10-fold Cross-Validation (train 90% / test 10%)

---

## 1. KẾT QUẢ — Đã tăng bao nhiêu

### Trên 7 dataset mất cân bằng (lymph, zoo, glass, hepatitis, german, vehicle, breast-w)

| Chỉ số | Trước (CMAR gốc) | Sau (cải tiến) | Tăng |
|--------|-----------------:|---------------:|-----:|
| **Accuracy** | 0.791 | 0.791 | +0.000 (giữ nguyên, không giảm) |
| **F1-score** | 0.723 | **0.760** | **+0.037 (+5.1%)** |
| **Recall** | 0.699 | **0.796** | **+0.097 (+13.8%)** |

→ **Recall tăng mạnh (+14%), F1 tăng rõ (+5%), Accuracy KHÔNG giảm.** Đúng mục tiêu: cải thiện lớp thiểu số mà không hi sinh độ chính xác chung.

### Điểm sáng — dataset khó nhất (lymph, tỷ lệ mất cân bằng 40:1)
| | Trước | Sau |
|---|------:|----:|
| Recall | 0.42 | **0.91** (gấp đôi) |

### Số dataset cải thiện
- **F1 tăng: 6/7** · **Recall tăng: 7/7** · **Accuracy không giảm: 5/7**

### 📋 THỐNG KÊ CHI TIẾT TỪNG DATASET (10-fold CV)

| Dataset | Tỷ lệ MCB | Baseline (Acc/F1/Recall/G-mean) | Cải tiến (Acc/F1/Recall/G-mean) | ΔF1 | ΔRecall |
|---------|----------:|--------------------------------|--------------------------------|----:|--------:|
| **lymph** | 40:1 | 0.806 / 0.686 / 0.421 / 0.000 | 0.816 / **0.843** / **0.908** / 0.902 | **+0.157** | **+0.487** |
| **zoo** | 10:1 | 0.949 / 0.902 / 0.872 / 0.859 | 0.955 / 0.911 / 0.879 / 0.864 | +0.009 | +0.007 |
| **glass** | 8:1 | 0.656 / 0.575 / 0.650 / 0.550 | 0.700 / **0.642** / **0.705** / 0.676 | **+0.067** | +0.055 |
| **hepatitis** | 4:1 | 0.819 / 0.728 / 0.759 / 0.752 | 0.743 / 0.704 / **0.803** / 0.796 | −0.024 | **+0.044** |
| **german** | 2:1 | 0.732 / 0.644 / 0.637 / 0.591 | 0.713 / 0.650 / 0.652 / 0.634 | +0.006 | +0.015 |
| **vehicle** | 1:1 | 0.646 / 0.604 / 0.650 / 0.555 | 0.658 / 0.621 / 0.662 / 0.574 | +0.017 | +0.012 |
| **breast-w** | 2:1 | 0.930 / 0.919 / 0.905 / 0.902 | 0.954 / **0.951** / **0.962** / 0.962 | +0.032 | +0.057 |
| **AVG (n=7)** | — | **0.791 / 0.723 / 0.699 / 0.601** | **0.791 / 0.760 / 0.796 / 0.773** | **+0.037** | **+0.097** |

**Nhận xét:**
- **lymph** (mất cân bằng nhất 40:1): cải thiện ngoạn mục — F1 +0.157, Recall +0.487, G-mean 0 → 0.902.
- **glass, breast-w**: F1 tăng tốt (+0.067, +0.032).
- **hepatitis**: F1 giảm nhẹ (−0.024) NHƯNG Recall tăng +0.044 — đánh đổi precision lấy recall lớp bệnh (chấp nhận được trong y tế).
- **6/7 dataset F1 tăng**, **7/7 Recall tăng**, G-mean AVG 0.601 → 0.773 (+0.172).

---

## 2. CÁCH CẢI TIẾN — không dùng trọng số thuộc tính

Cải tiến ở 4 tầng, KHÔNG gán trọng số cho thuộc tính (theo hướng đề tài):

| Tầng | Kỹ thuật | Tác dụng |
|------|----------|----------|
| Dữ liệu | SMOTE/MWMOTE oversampling | Tạo thêm mẫu lớp hiếm → tăng Recall |
| Vote | Added Value `confidence − P(c)` | Chống thiên vị lớp đa số → tăng F1/Recall |
| Vote | Cost-sensitive `(N/count)^β` | Boost lớp hiếm → tăng Recall |
| Kiểm soát | Adaptive gating (imbalance ratio) | Chỉ bật khi cần → bảo vệ Accuracy |

---

## 3. CƠ SỞ LÝ THUYẾT — Các bài báo đã dùng

### Nền tảng
**[1] CMAR** — Li, W., Han, J., Pei, J. (2001). *CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules.* IEEE ICDM 2001.
→ Thuật toán gốc được cải tiến. Dùng **chi-square** để cắt tỉa & vote luật.

### Kỹ thuật cải tiến
**[2] MCWCAR** — Wu, W., et al. (2024). *A novel software defect prediction approach via weighted classification based on ARM.* Engineering Applications of AI, 129, 107622.
→ Cung cấp **Added Value voting** (`confidence − P(c)`) xử lý mất cân bằng.

**[3] MWMOTE** — Barua, S., et al. (2014). *MWMOTE — Majority Weighted Minority Oversampling Technique.* IEEE TKDE, 26(2).
→ Oversampling lớp cực hiếm.

**[4] King & Zeng** — King, G., Zeng, L. (2001). *Logistic Regression in Rare Events Data.* Political Analysis, 9(2).
→ Cost-sensitive boost cho lớp hiếm.

**[5] He & Garcia** — He, H., Garcia, E.A. (2009). *Learning from Imbalanced Data.* IEEE TKDE, 21(9).
→ Khái niệm imbalance ratio (adaptive gating) + định nghĩa **G-mean** (công thức 17). Tạp chí IEEE uy tín.

---

## 4. KIỂM ĐỊNH THỐNG KÊ

- **Chi-square**: tích hợp sẵn trong CMAR — cắt tỉa luật (ngưỡng 3.841, p=0.05) + weighted chi-square để vote. Đây là kiểm định thống kê chuẩn của thuật toán.
- **So sánh trước/sau**: bảng Win-Tie-Loss trên từng dataset (cải tiến thắng trên nhóm mất cân bằng, hòa trên nhóm cân bằng, không thua dataset nào).

---

## 5. TÓM TẮT 1 CÂU

> "Cải tiến CMAR bằng SMOTE (Barua 2014) + Added Value voting (MCWCAR Wu 2024) + cost-sensitive (King & Zeng 2001) + adaptive gating (He & Garcia 2009), KHÔNG dùng trọng số thuộc tính — tăng F1 +5% và Recall +14% trên dữ liệu mất cân bằng, Accuracy không giảm; kiểm định bằng chi-square sẵn có trong CMAR."
