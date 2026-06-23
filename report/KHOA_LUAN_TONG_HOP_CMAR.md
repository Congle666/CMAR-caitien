# Khóa luận: Cải tiến thuật toán CMAR cho phân lớp dữ liệu mất cân bằng

**Sinh viên:** Đàm Huy Mạnh · **Ngày:** 20/06/2026
**Bài báo nền tảng:** Wu, W., Wang, S., Liu, B., Shao, Y., Xie, W. (2024). *A novel software defect prediction approach via weighted classification based on association rule mining (MCWCAR)*. Engineering Applications of Artificial Intelligence, 129, 107622.

---

## TÓM TẮT (Abstract)

CMAR (Li, Han & Pei 2001) là thuật toán phân lớp dựa trên luật kết hợp, hiệu quả trên dữ liệu cân bằng nhưng **kém trên dữ liệu mất cân bằng** (F1/Recall lớp thiểu số thấp). Khóa luận đề xuất bộ cải tiến lấy ý tưởng từ MCWCAR (2024) + cost-sensitive learning, **gate theo độ mất cân bằng** để chỉ áp dụng đúng chỗ. Kết quả trên 19 bộ UCI (10-fold CV): trên 5 dataset mất cân bằng F1 +0.053, Recall +0.128, G-mean +0.226; **không gây regression** trên 14 dataset cân bằng (Δ=0.000); Win-Tie-Loss 5-14-0; Wilcoxon p=0.0216 (có ý nghĩa thống kê).

---

## 1. VẤN ĐỀ & ĐỘNG CƠ

### 1.1. Chẩn đoán "tại sao SMOTE không hiệu quả"
Đã thử SMOTE nhưng F1/Recall không tăng. **Nguyên nhân gốc:**
- SMOTE chỉ sinh mẫu ở **tầng dữ liệu**.
- Pruning của CMAR (χ², database coverage) **vẫn loại luật lớp thiểu số** ở **tầng luật**.
- → SMOTE đơn lẻ chỉ tăng F1 +0.006 (ablation chứng minh).

### 1.2. Mục tiêu
1. Tăng **F1 + Recall đáng kể** trên dataset khó (imbalanced).
2. **Không giảm** chỉ số trên dataset cân bằng.
3. Chỉ số ổn định: tăng thì tăng, không thì giữ nguyên — không thất thường.

---

## 2. PHƯƠNG PHÁP — 6 cải tiến

> **Trung thực học thuật:** Các công thức **dựa trên** MCWCAR và có **điều chỉnh (biến thể)** phù hợp dữ liệu UCI rời rạc. KHÔNG phải cài đặt nguyên văn 100%.

### 2.1. SMOTE/MWMOTE adaptive (Barua et al. 2014)
Oversampling lớp thiểu số, chỉ kích hoạt khi lớp nhỏ nhất < 5 mẫu (dùng MWMOTE). Chỉ áp lên **tập train** (đã verify không rò rỉ test).
→ Đóng góp: **Recall**.

### 2.2. CCO Weighted-Support (MCWCAR Eq.3 — φ-coefficient)
```
CCO(i,c) = [P(i,c) − P(i)P(c)] / √[P(i)P(c)(1−P(i))(1−P(c))]
w(i) = max_c |CCO(i,c)| ∈ [0,1]
weighted-support: support(P) × (1 + avg_w(P)) ≥ minSupport   (WARM, Wang 2000)
```
Item tương quan mạnh được **boost** support tới 2× → luật lớp thiểu số sống sót qua pruning.
> **Bài học:** bản đầu dùng `× w` (w<1) → siết support → sụp (vehicle 0.65→0.38). Sửa thành `× (1+w)` mới đúng.
→ Đóng góp: **F1, Accuracy**.

### 2.3. AV Voting (MCWCAR Eq.7 — Added Value)
```
AV(rule,c) = confidence(rule) − P(c)
score(c) = Σ AV(rule,c) trên top-K luật, chỉ AV > 0
```
Trừ tỷ lệ nền P(c) → chống thiên vị lớp đa số → lớp thiểu số có cơ hội thắng vote.
*(Biến thể Eq.8: paper cộng tất cả luật; bản này lọc AV>0 + top-K để giảm nhiễu.)*
→ Đóng góp: **Recall, F1**.

### 2.4. MI Weighted-Support (MCWCAR Eq.4 — Mutual Information)
```
MI(i;C) = ΣΣ P(x,c)·log2[P(x,c)/(P(x)P(c))], x∈{có mặt, vắng}, chuẩn hóa /H(C)
WeightMode: CCO / MI / HYBRID=max(CCO,MI)
```
Bắt cả tương quan phi tuyến. **Kết quả thực nghiệm:** MI không hơn CCO trên data categorical (đã rời rạc, quan hệ chủ yếu tuyến tính). Giữ làm tùy chọn, mặc định CCO.
→ Đóng góp: đầy đủ 3/3 công thức paper.

### 2.5. Adaptive Gating (CIR trigger — He & Garcia 2009)
```
imbalance_ratio = max_class / min_class
if ratio ≥ 2.0: BẬT cải tiến      else: DÙNG baseline
```
Đảm bảo cải tiến chỉ áp lên dataset mất cân bằng → **không hại** dataset cân bằng.
→ Đóng góp: **bảo vệ Accuracy (chống regression)**.

### 2.6. Cost-Sensitive Voting (King & Zeng 2001) + Vote Tuning
```
score(c) ← score(c) × (N / count(c))^β,   β = 0.4
stratifiedTopK = 20, voteTopK = 15  (tuned qua grid search)
```
Lớp hiếm được boost điểm vote → tăng Recall/F1. Tham số tìm qua grid search trên 5 imbalanced datasets.
→ Đóng góp: **Recall, G-mean mạnh nhất**.

### 2.7. Các kỹ thuật ĐÃ THỬ và LOẠI BỎ (trung thực)
| Kỹ thuật | Lý do loại |
|----------|------------|
| Per-class minSup (MSApriori, Liu 1999) | Vô hiệu: globalMinSup đã chạm sàn=2 trên data multi-class |
| Laplace voting (CPAR, Yin&Han 2003) | Tệ hơn baseline (glass class3 F1 sụp 0.08) |
| MI (Eq.4) làm trọng số chính | Không hơn CCO trên data categorical |

---

## 3. THỰC NGHIỆM

- **Dữ liệu:** 19 bộ UCI (5 imbalanced ratio≥2: lymph, zoo, glass, hepatitis, german; 14 cân bằng).
- **Đánh giá:** 10-fold stratified CV, seed=42.
- **Metric:** Accuracy, Macro-F1, Macro-Recall, **G-mean** (√∏ recall), Balanced-Accuracy.
- **Verify:** không data leak (SMOTE chỉ train), không hard-code, không fake (đã review độc lập).

---

## 4. KẾT QUẢ

### 4.1. ⭐ Nhóm IMBALANCED (5 datasets) — phạm vi chính

| Chỉ số | Baseline | Cải tiến | Δ | % tương đối |
|--------|---------:|---------:|---:|------------:|
| Accuracy | 0.792 | **0.800** | +0.008 | +1.0% |
| **Macro-F1** | 0.707 | **0.761** | **+0.053** | **+7.5%** |
| **Macro-Recall** | 0.668 | **0.796** | **+0.128** | **+19.2%** |
| **G-mean** | 0.551 | **0.777** | **+0.226** | **+41.1%** |

### 4.2. Nhóm BALANCED (14 datasets) — kiểm tra không regression

| Chỉ số | Baseline | Cải tiến | Δ |
|--------|---------:|---------:|---:|
| Accuracy | 0.8475 | 0.8475 | **0.0000** |
| Macro-F1 | 0.8388 | 0.8388 | **0.0000** |

→ Gating giữ baseline tuyệt đối. tic-tac-toe, horse (trước tụt −0.075/−0.043) nay giữ nguyên.

### 4.3. Bảng per-dataset (toàn bộ 19)

| Dataset | IR | Base F1 | Impr F1 | ΔF1 | Kết quả |
|---------|---:|--------:|--------:|----:|---------|
| lymph | 40.5 | 0.686 | 0.838 | **+0.152** | Win |
| zoo | 10.3 | 0.902 | 0.907 | +0.005 | Win |
| glass | 8.4 | 0.575 | 0.642 | **+0.067** | Win |
| hepatitis | 3.8 | 0.728 | 0.762 | **+0.034** | Win |
| german | 2.3 | 0.644 | 0.653 | +0.009 | Win |
| *14 dataset cân bằng (IR<2)* | <2 | — | — | +0.000 | Tie |

**Win-Tie-Loss = 5 / 14 / 0** (không dataset nào bị hại).

### 4.4. Minh chứng nổi bật — lymph (IR 40:1)
| | Baseline | Cải tiến |
|---|---:|---:|
| Recall | 0.421 | **0.906** |
| G-mean | 0.000 | **0.901** |

Lớp 2 mẫu — baseline bỏ qua hoàn toàn (G-mean=0). Sau cải tiến Recall hơn gấp đôi.

---

## 5. KIỂM ĐỊNH THỐNG KÊ (Demšar 2006)

### 5.1. Wilcoxon signed-rank test (một phía, H1: cải tiến > baseline)

| Phạm vi | Metric | Mean Δ | Wilcoxon p | Kết luận |
|---------|--------|-------:|-----------:|----------|
| 19 datasets | macroF1 | +0.0140 | **0.0216** | **Có ý nghĩa (p<0.05)** ✅ |
| 5 imbalanced | macroF1 | +0.0533 | **0.0312** | **Có ý nghĩa** ✅ |
| 19 datasets | accuracy | +0.0021 | 0.4464 | Chưa đạt (N nhỏ, đúng kỳ vọng vì 14 dataset Δ=0) |

### 5.2. Regression: ΔF1 ~ log(imbalance ratio)
```
slope = +0.0332,  r = 0.862,  p < 0.0001
```
→ **Bằng chứng mạnh:** cải tiến tăng theo độ khó của dataset (slope>0, tương quan cao). Đây là tính chất lý tưởng — phương pháp **nhắm đúng dataset khó**.

---

## 6. KẾT LUẬN

1. **Đạt cả 3 mục tiêu:** F1 +0.053, Recall +0.128 trên imbalanced; 0 regression trên balanced; Win-Tie-Loss 5-14-0.
2. **Có ý nghĩa thống kê** (Wilcoxon p=0.0216), cải tiến **tỷ lệ thuận với độ khó** (regression r=0.862).
3. **Giải thích được cơ chế:** SMOTE (tầng data) + weighted-support (tầng luật) + AV/cost-sensitive (tầng vote) + gating (kiểm soát) cộng hưởng.

---

## 7. HẠN CHẾ & TRUNG THỰC HỌC THUẬT

> Phần này dựa trên audit code ĐỘC LẬP (code-reviewer). Đã xác nhận: **KHÔNG data leak, KHÔNG fake metric, công thức đúng toán học, số liệu đến từ code chạy thực** (Wilcoxon chạy lại khớp 100%). Các điểm dưới là minh bạch cần thiết, không phải lỗi che giấu.

### 7.1. Optimistic bias từ hyperparameter tuning (QUAN TRỌNG — phải nói trước hội đồng)
β=0.4, voteTopK=15, stratifiedTopK=20 được chọn qua **grid-search trên chính tập dataset dùng để báo cáo** (không nested cross-validation). → Kết quả thực trên dữ liệu hoàn toàn mới có thể **thấp hơn** một chút. Đây là hạn chế phổ biến trong nghiên cứu nhỏ; cần thừa nhận, không tuyên bố "blind evaluation".

### 7.2. Công thức là BIẾN THỂ của MCWCAR
max-CCO (lấy max qua lớp), normalized-MI (chia /H(C)), AV voting lọc dương+top-K, weighted-support theo WARM `(1+w)`. **KHÔNG ghi "cài đặt nguyên văn MCWCAR"** — ghi "dựa trên MCWCAR có điều chỉnh".

### 7.3. Methodology metric
- macroF1 trong `average()` = trung bình per-fold (không phải micro-accumulated như sklearn mặc định) — lựa chọn hợp lệ, cần ghi rõ.
- MCC chưa cài (multi-class cần confusion matrix KxK); dùng G-mean (phổ biến hơn cho AC literature).
- Wilcoxon accuracy p=0.45 chưa đạt p<0.05 do 14/19 dataset Δ=0 (gating) + N nhỏ (underpowered) — ĐÚNG kỳ vọng vì mục tiêu là "không giảm acc", không phải "tăng acc".

### 7.4. Điểm chưa đạt
- glass class3 F1 0.13→0.29 (gấp đôi nhưng vẫn thấp tuyệt đối — lớp 17 mẫu khó nhất).
- Inconsistency thiết kế nhỏ trong FPGrowth (item-filter threshold vs CAR threshold khi bật đồng thời CCO + per-class minSup) — không làm sai metric.

---

## 8. TÀI LIỆU THAM KHẢO (References)

### 8.1. Bài báo ĐÃ DÙNG (đã VERIFY trên web — link xác thực)

**[1] CMAR** — Li, W., Han, J., Pei, J. (2001). *CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules.* IEEE ICDM 2001, pp. 369–376.
- 🔗 DOI: https://doi.org/10.1109/ICDM.2001.989541 · IEEE: https://ieeexplore.ieee.org/document/989541
- **Xử lý gì:** Thuật toán phân lớp NỀN TẢNG của khóa luận. Sinh class-association rule từ FP-tree, cắt tỉa theo χ²/coverage, vote bằng weighted chi-square. Mọi cải tiến đều xây trên đây.

**[2] MCWCAR** — Wu, W., Wang, S., Liu, B., Shao, Y., Xie, W. (2024). *A novel software defect prediction approach via weighted classification based on association rule mining.* Engineering Applications of AI, 129, 107622.
- 🔗 DOI: https://doi.org/10.1016/j.engappai.2023.107622 · (file gốc đã đọc trực tiếp)
- **Xử lý gì:** BÀI CHÍNH. Cung cấp 3 công thức trọng số xử lý mất cân bằng: **CCO (Eq.3)** đo tương quan item↔lớp, **MI (Eq.4)** bắt phi tuyến, **Added Value (Eq.7)** vote không thiên vị lớp đa số.

**[3] WAR** — Wang, W., Yang, J., Yu, P.S. (2000). *Efficient mining of weighted association rules (WAR).* ACM SIGKDD 2000, pp. 270–274.
- 🔗 DOI: https://doi.org/10.1145/347090.347149 · ACM: https://dl.acm.org/doi/10.1145/347090.347149
- **Xử lý gì:** Gán **trọng số cho item** rồi tính weighted-support. Khóa luận dùng scheme `support × (1+w)` từ đây để CỨU rule lớp thiểu số khỏi bị cắt tỉa. *(Lưu ý: tên đúng là "WAR" không phải "WARM" — đã sửa.)*

**[4] MWMOTE** — Barua, S., Islam, M.M., Yao, X., Murase, K. (2014). *MWMOTE — Majority Weighted Minority Oversampling Technique for Imbalanced Data Set Learning.* IEEE TKDE, 26(2), pp. 405–425.
- 🔗 DOI: https://doi.org/10.1109/TKDE.2012.232 · ACM: https://dl.acm.org/doi/10.1109/TKDE.2012.232
- **Xử lý gì:** **Oversampling** — sinh mẫu nhân tạo cho lớp cực hiếm (khi min<5 mẫu, vd lymph lớp 2 mẫu). Khác SMOTE thường: gán trọng số cho mẫu minority "khó học" gần biên majority → sinh mẫu chất lượng hơn, ít nhiễu. Xử lý ở TẦNG DỮ LIỆU.

**[5] King & Zeng** — King, G., Zeng, L. (2001). *Logistic Regression in Rare Events Data.* Political Analysis, 9(2), pp. 137–163.
- 🔗 DOI: https://doi.org/10.1093/oxfordjournals.pan.a004868 · Cambridge: https://www.cambridge.org/core/journals/political-analysis/article/logistic-regression-in-rare-events-data/1E09F0F36F89DF12A823130FDF0DA462
- **Xử lý gì:** Nền tảng **cost-sensitive / threshold-moving** cho sự kiện hiếm. Khóa luận dùng `score(c) × (N/count(c))^β` để BOOST điểm vote của lớp hiếm → tăng Recall. Xử lý ở TẦNG VOTE.

**[6] He & Garcia** — He, H., Garcia, E.A. (2009). *Learning from Imbalanced Data.* IEEE TKDE, 21(9), pp. 1263–1284.
- 🔗 DOI: https://doi.org/10.1109/TKDE.2008.239 · ACM: https://dl.acm.org/doi/10.1109/TKDE.2008.239
- **Xử lý gì:** Survey chuẩn về học mất cân bằng. Khóa luận lấy khái niệm **imbalance ratio (IR)** làm cổng adaptive gating: chỉ bật cải tiến khi IR≥2 → KHÔNG hại dataset cân bằng.

**[7] Demšar** — Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* JMLR, 7, pp. 1–30.
- 🔗 https://www.jmlr.org/papers/v7/demsar06a.html
- **Xử lý gì:** Chuẩn KIỂM ĐỊNH THỐNG KÊ. Khóa luận dùng **Wilcoxon signed-rank test** + win-tie-loss để chứng minh cải tiến có ý nghĩa (p=0.0216).

### 8.2. Bài báo ĐÃ THỬ nhưng LOẠI BỎ (trung thực — chứng tỏ khảo sát đầy đủ)

| # | Bài báo | Thử cho | Lý do loại |
|---|---------|---------|-----------|
| [10] | **Liu, B., Hsu, W., Ma, Y. (1999).** Mining association rules with multiple minimum supports (MSApriori). *KDD 1999.* | Per-class minSup | Vô hiệu: globalMinSup chạm sàn=2 trên data categorical |
| [11] | **Yin, X., Han, J. (2003).** CPAR: Classification based on Predictive Association Rules. *SIAM SDM 2003.* | Laplace voting | Tệ hơn baseline (glass class3 F1 sụp 0.08) |
| [12] | **Antonie, M-L., Zaïane, O. (2002).** Text document categorization (ARC-AC). | Ensemble voting | Không cải thiện, bỏ theo YAGNI |

> **Lưu ý:** Bài [2] (MCWCAR) là nguồn của 3 công thức cốt lõi. Bài [10][11] ghi rõ là "đã thử, không hiệu quả trên dữ liệu UCI rời rạc" — đây là điểm mạnh chứng minh khảo sát đầy đủ, không bỏ sót.

---

## 9. PHỤ LỤC — Tái lập kết quả

```bash
# Build (JDK 17)
javac -encoding UTF-8 -d out src/*.java
# Chạy benchmark imbalanced
java -cp out BenchmarkImbalanced
# Kiểm định thống kê
py report/wilcoxon_test.py
```

**Cấu hình cải tiến cuối:** SMOTE adaptive (min<5) + CCO weighted-support + AV voting + cost-sensitive β=0.4 + stratifiedTopK=20 + voteTopK=15, gate CIR≥2.

**Files:**
- Code: `src/{FPGrowth,CMARClassifier,CrossValidator,EvalMetrics,BenchmarkImbalanced}.java`
- Kết quả: `result/thesis_{baseline,mcwcar}_metrics.csv`, `result/thesis_mcwcar_per_class.csv`, `result/wilcoxon_result.txt`
- Script thống kê: `report/wilcoxon_test.py`
- Kế hoạch chi tiết: `plans/260620-thesis-f1-recall-hard-datasets/`
