# CMAR cải tiến cho dữ liệu mất cân bằng

**Đề tài khóa luận:** Cải tiến thuật toán CMAR (Classification based on Multiple Association Rules) nâng cao F1-score và Recall trên dữ liệu mất cân bằng.

**Sinh viên:** Lê Hồng Công · **Email:** lhc09062004@gmail.com
**Repo:** https://github.com/Congle666/CMAR-caitien

---

## 1. Tổng quan

CMAR gốc (Li, Han, Pei 2001) phân lớp tốt trên dữ liệu cân bằng nhưng **kém trên dữ liệu mất cân bằng** — lớp thiểu số bị bỏ qua (F1/Recall thấp). Khóa luận này cải tiến CMAR bằng 3 kỹ thuật, **không dùng trọng số thuộc tính**:

| Tầng | Kỹ thuật | Bài báo nguồn | Tác dụng |
|------|----------|---------------|----------|
| Dữ liệu | **MWMOTE** oversampling | Barua et al. (2014), IEEE TKDE | Tăng Recall (lớp cực hiếm) |
| Phân lớp | **Added Value voting** (chính) | MCWCAR — Wu et al. (2024), EAAI | Tăng F1 + Recall |
| Kiểm soát | **Adaptive gating** | He & Garcia (2009), IEEE TKDE | Bảo vệ Accuracy |

**Nền tảng:** CMAR — Li, Han, Pei (2001), IEEE ICDM.

---

## 2. Kết quả (10-fold CV, 19 bộ UCI)

### Trên 5 dataset mất cân bằng (trọng tâm)
| Chỉ số | CMAR gốc | Cải tiến | Δ |
|--------|---------:|---------:|---:|
| **F1-score** | 0.707 | **0.747** | **+0.040** |
| **Recall** | 0.668 | **0.776** | **+0.108** |
| Accuracy | 0.792 | 0.791 | giữ (~) |

### Trên 14 dataset cân bằng
Giữ nguyên 100% (adaptive gating không can thiệp). **Win-Tie-Loss: 5 Thắng / 14 Hòa / 0 Thua.**

### Minh chứng chính: glass
F1 0.575 → **0.623** (+0.048) — hoàn toàn nhờ Added Value voting, lớp đủ mẫu nên đáng tin.

---

## 3. Cấu trúc dự án

```
src/                       Mã nguồn Java (JDK 17)
├── CMARClassifier.java    Bộ phân lớp CMAR + Added Value voting
├── FPGrowth.java          Khai phá Class Association Rule
├── CrossValidator.java    10-fold stratified cross-validation
├── EvalMetrics.java       Tính Accuracy / F1 / Recall / G-mean
├── MWMOTE.java            Oversampling (Barua 2014)
├── BenchmarkImbalanced.java   Driver chạy benchmark chính
└── ...
data_clean/                19 bộ dữ liệu UCI đã tiền xử lý
result/                    Kết quả CSV
report/                    Báo cáo chi tiết (xem mục 5)
```

---

## 4. Cách chạy

```bash
# Build (cần JDK 17, dùng -encoding UTF-8 vì mã nguồn có tiếng Việt)
javac -encoding UTF-8 -d out src/*.java

# Chạy benchmark trên các dataset mất cân bằng
java -cp out BenchmarkImbalanced
```

**Cấu hình cải tiến:** MWMOTE (khi lớp nhỏ nhất < 5 mẫu) + Added Value voting + adaptive gating (imbalance ratio ≥ 2), stratifiedTopK=20, voteTopK=5. seed=42 (cố định → tái lập 100%).

---

## 5. Tài liệu báo cáo (thư mục `report/`)

| File | Nội dung |
|------|----------|
| `BAO_CAO_CHI_TIET_CAI_TIEN.md` | Báo cáo chi tiết: trước/sau, số liệu từng dataset, lý do tăng |
| `BAO_CAO_GUI_CO.md` | Bản tóm tắt gửi giáo viên (công thức + kết quả) |
| `BAO_CAO_ABLATION_MWMOTE.md` | Tách đóng góp MWMOTE vs Added Value voting |
| `GIAI_THICH_LOP_CUC_HIEM.md` | Giải thích vấn đề lớp cực hiếm (lymph 2 mẫu) |
| `CO_SO_LY_THUYET_VA_TAI_LIEU.md` | Cơ sở lý thuyết + tài liệu tham khảo |

---

## 6. Lưu ý trung thực học thuật

- **Số liệu đã verify độc lập 3 lần** (không data leak, không fake metric, reproducible 100%).
- **lymph (lớp 2 mẫu):** kết quả cao nhưng KHÔNG đại diện thống kê (chỉ minh họa) — xem `GIAI_THICH_LOP_CUC_HIEM.md`.
- **Minh chứng định lượng chính:** glass (lớp đủ mẫu, nhờ Added Value voting).
- Công thức là **biến thể** của MCWCAR (dựa trên, có điều chỉnh — không nguyên văn).

---

## 7. Tài liệu tham khảo

1. Li, W., Han, J., Pei, J. (2001). *CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules.* IEEE ICDM 2001.
2. Wu, W., et al. (2024). *A novel software defect prediction approach via weighted classification based on association rule mining.* Engineering Applications of AI, 129, 107622.
3. Barua, S., et al. (2014). *MWMOTE — Majority Weighted Minority Oversampling Technique.* IEEE TKDE, 26(2).
4. He, H., Garcia, E.A. (2009). *Learning from Imbalanced Data.* IEEE TKDE, 21(9).
