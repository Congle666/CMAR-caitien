# Báo cáo final: Cải tiến thuật toán CMAR

> **Cập nhật 30/05/2026:** Code chính đã chuyển sang pipeline **No-H2** để bỏ minSup động theo từng class/dataset. File này là báo cáo lịch sử cho bản có H2; khi nộp theo code hiện tại, ưu tiên dùng `report/BAO_CAO_HUONG_NO_H2.md`.

**Đề tài:** Cải tiến thuật toán CMAR cho bài toán phân lớp, tập trung vào dữ liệu mất cân bằng  
**Phiên bản nên dùng để báo cáo:** Final Pipeline = v13 + H4 + H5b Adaptive  
**Nguồn số liệu:** `result/final_baseline_metrics.csv`, `result/final_full_metrics.csv`, `result/final_*_per_class.csv`, `result/v13_*`  
**Giao thức đánh giá:** 10-fold stratified cross-validation, seed = 42  
**Ghi chú:** File này tổng hợp lại kết quả theo source code và CSV hiện tại để tránh nhầm giữa các báo cáo v13, v14, v15 cũ.

---

## 1. Kết luận nhanh

Baseline trong project là CMAR gốc theo Li, Han và Pei (2001). Pipeline baseline dùng FP-Growth/CR-tree để sinh Class Association Rules, cắt tỉa bằng chi-square và database coverage, sau đó phân lớp bằng weighted chi-square voting.

Pipeline cải tiến hiện tại thêm 3 lớp:

| Lớp | Kỹ thuật | Mục tiêu |
|---|---|---|
| Layer 1 | H2 class-specific minSup + Adaptive Borderline-SMOTE-N | Tăng F1 và Recall cho class thiểu số |
| Layer 2 | H4 Stratified Coverage | Bảo vệ rule của từng class khi cắt tỉa |
| Layer 3 | H5b Adaptive confidence x Lift voting | Cải thiện quyết định khi nhiều rule match nhiều class |

Kết quả final trung bình trên 19 dataset UCI:

| Metric | Trước cải tiến | Sau cải tiến | Tăng tuyệt đối | Tăng tương đối |
|---|---:|---:|---:|---:|
| Accuracy | 0.8330 | 0.8401 | +0.0070 | +0.85% |
| Macro-F1 | 0.8041 | 0.8231 | +0.0190 | +2.36% |
| Macro-Recall | 0.7973 | 0.8309 | +0.0336 | +4.21% |

Kết quả nổi bật nhất nằm ở dataset `lymph`, một trường hợp mất cân bằng rất mạnh:

| Metric | Baseline | Final Pipeline | Tăng |
|---|---:|---:|---:|
| Accuracy | 0.8058 | 0.8806 | +0.0748 |
| Macro-F1 | 0.6864 | 0.8726 | +0.1862 |
| Macro-Recall | 0.4209 | 0.8770 | +0.4560 |

Ý nghĩa chính: baseline có Accuracy khá cao nhưng bỏ sót class hiếm. Pipeline final cải thiện mạnh Macro-Recall và Macro-F1, tức là xử lý đúng vấn đề mất cân bằng dữ liệu.

---

## 2. Phân biệt v13, v14, v15 trong repo

Trong thư mục `report/` hiện có nhiều báo cáo lịch sử. Không nên nộp lẫn lộn các file này nếu chưa giải thích rõ.

| File | Ý nghĩa |
|---|---|
| `v13_imbalanced_results.md` | Báo cáo v13 cũ, chỉ tập trung 7 dataset mất cân bằng, dùng H2 + Adaptive Borderline-SMOTE |
| `v14_three_layer_final_results.md` | Báo cáo khi đã thêm pipeline 3 tầng: H2 + SMOTE + H4 + H5b |
| `v15_post_bugfix_final_results.md` | Báo cáo sau khi sửa bug về metric, Borderline threshold, stddev |
| `KET_QUA_TRUOC_VA_SAU.md` | Báo cáo tổng hợp sau hơn, có Adaptive H5b |
| **`BAO_CAO_FINAL_CMAR_CAI_TIEN.md`** | File nên dùng để báo cáo vì đã tổng hợp lại theo source code và CSV hiện tại |

Tên gọi khuyến nghị khi trình bày:

```text
CMAR Final Pipeline: v13 + H4 + H5b Adaptive sau bugfix
```

Không nên gọi đơn giản là v13, vì source hiện tại đã thêm H4, H5b và các bugfix sau v13.

---

## 3. Source code hiện tại đang làm gì

### 3.1. Đọc và tiền xử lý dữ liệu

File liên quan:

- `preprocess_datasets.py`
- `src/DatasetLoader.java`
- `src/Transaction.java`

Pipeline tiền xử lý:

1. Đọc dataset UCI thô từ thư mục `datasets/`.
2. Xử lý delimiter, cột ID, cột class và missing value.
3. Rời rạc hóa thuộc tính liên tục bằng equal-frequency 5 bins.
4. Chuyển mỗi dòng dữ liệu thành transaction gồm các item dạng `attribute=value`.
5. Ghi dữ liệu sạch vào `data_clean/`.

Lý do cần tiền xử lý: CMAR là thuật toán dựa trên itemset và association rule, vì vậy dữ liệu phải ở dạng categorical/item trước khi mining.

### 3.2. Baseline CMAR gốc

File liên quan:

- `src/FPGrowth.java`
- `src/FPTree.java`
- `src/FPNode.java`
- `src/AssociationRule.java`
- `src/CMARClassifier.java`
- `src/CRTree.java`

Baseline hiện tại cài đặt các thành phần chính của CMAR:

1. Sinh frequent pattern bằng FP-Growth mở rộng có lưu class distribution.
2. Sinh Class Association Rules dạng `condset => class`.
3. Sắp xếp rule theo confidence, support, độ dài condset.
4. Cắt tỉa rule tổng quát.
5. Cắt tỉa bằng chi-square với ngưỡng 3.841, tương ứng p = 0.05 và df = 1.
6. Cắt tỉa bằng database coverage với `delta = 4`.
7. Lưu rule bằng CR-tree để truy vấn rule match nhanh.
8. Phân lớp bằng weighted chi-square voting theo CMAR gốc.

Baseline không dùng SMOTE, không dùng H2, không dùng H4 và không dùng H5b.

---

## 4. Các cải tiến đã thêm

### 4.1. H2: Class-specific minimum support

File liên quan:

- `src/CrossValidator.java`
- `src/FPGrowth.java`

CMAR gốc dùng một ngưỡng minSupport toàn cục. Điểm yếu của cách này là class thiểu số có quá ít bản ghi, nên itemset/rule của class đó thường không đạt ngưỡng support và không sinh được rule.

H2 trong source hiện tại tính ngưỡng theo từng class:

```text
minSup(c) = max(2, round(supPct * freq(c)))
```

Trong đó:

- `supPct` là cấu hình support theo dataset.
- `freq(c)` là số bản ghi của class `c` trong training fold.
- `max(2, ...)` tránh trường hợp ngưỡng quá thấp sinh quá nhiều rule nhiễu.

Ý nghĩa: class lớn có ngưỡng cao hơn, class nhỏ có ngưỡng thấp hơn, từ đó class thiểu số có cơ hội sinh rule.

Lưu ý: H2 là phần có thể xử lý sau nếu muốn đơn giản hóa báo cáo. File này ghi đúng theo source hiện tại. Nếu sau này bỏ H2, cần chạy lại benchmark và cập nhật số liệu.

Paper liên quan:

- Liu, B., Ma, Y., Wong, C.K. (2000). *Improving an Association Rule Based Classifier*. PKDD 2000.

### 4.2. Borderline-SMOTE-N cho dữ liệu categorical

File liên quan:

- `src/SMOTE.java`
- `src/BorderlineSMOTE.java`
- `src/CrossValidator.java`
- `src/BenchmarkImbalanced.java`
- `src/BenchmarkFinal.java`

SMOTE-N tạo thêm bản ghi minority cho dữ liệu categorical bằng cách:

1. Tìm k nearest neighbors theo Hamming distance.
2. Với mỗi thuộc tính, chọn mode value từ record gốc và neighbors.
3. Tạo synthetic transaction mới cùng class.

Borderline-SMOTE-N chỉ oversample các minority record nằm gần ranh giới quyết định:

```text
SAFE   : majority_neighbors < ceil(k/2)
DANGER : ceil(k/2) <= majority_neighbors < k
NOISE  : majority_neighbors = k
```

Trong source, `k = 5`, `targetRatio = 1.0`. Sau bugfix, code dùng `ceil(k/2)` bằng `(kEff + 1) / 2`, khớp hơn với logic của Borderline-SMOTE.

Adaptive trigger hiện tại:

```text
if min_class_freq < 5:
    bật Borderline-SMOTE-N
else:
    không oversampling
```

Ý nghĩa: chỉ bật SMOTE khi class nhỏ nhất cực ít mẫu, tránh tạo synthetic noise trên dataset không quá cực đoan.

Paper liên quan:

- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR.
- Han, Wang, Mao (2005). *Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning*. ICIC.

### 4.3. H4: Stratified Coverage

File liên quan:

- `src/CMARClassifier.java`

Database coverage của CMAR có thể làm rule của minority class bị loại quá nhiều, vì rule majority thường có support cao và được xử lý trước.

H4 thêm cơ chế bảo vệ top-K rule theo từng class trước khi chạy coverage pruning:

```text
for each class c:
    reserve top-K rules of class c
then:
    run database coverage pruning
```

Trong source, `TOP_K = 10`. Mục tiêu là mỗi class vẫn có rule đại diện sau cắt tỉa, đặc biệt với dataset có class thiểu số.

### 4.4. H5b: Adaptive confidence x Lift voting

File liên quan:

- `src/CMARClassifier.java`
- `src/BenchmarkFinal.java`

CMAR gốc khi có nhiều rule match nhiều class sẽ dùng weighted chi-square voting. H5b thêm cách vote mới:

```text
Lift(r, c) = confidence(r) * N / classFreq(c)
weight(r) = confidence(r) * Lift(r, c)
score(c) = sum weight(r) trên top-K rule của class c
```

Ý nghĩa:

- Confidence đo độ tin cậy của rule.
- Lift đo mức tương quan dương giữa condset và class.
- Tích confidence x Lift ưu tiên rule vừa có độ tin cậy cao, vừa có tương quan mạnh với class.

Source hiện tại có thêm adaptive threshold:

```text
chỉ dùng confidence x Lift nếu originalImbalanceRatio >= 3.0
nếu ratio < 3.0 thì fallback về weighted chi-square gốc
```

Lý do: Lift có bias theo kích thước class. Trên data gần cân bằng, Lift có thể đẩy model về class nhỏ không cần thiết. Adaptive H5b giúp tránh regression trên các dataset balanced như tic-tac-toe, diabetes, horse.

Paper liên quan:

- Brin et al. (1997). *Dynamic Itemset Counting and Implication Rules for Market Basket Data*. SIGMOD.
- Bahri et al. (2020). *WEviRC: Weighted Evidential Rule Combination*. Knowledge-Based Systems.

---

## 5. Kết quả trước và sau trên 19 UCI dataset

Bảng dưới lấy từ:

- Trước: `result/final_baseline_metrics.csv` và `result/final_baseline_per_class.csv`
- Sau: `result/final_full_metrics.csv` và `result/final_full_per_class.csv`

| Dataset | Accuracy trước | Accuracy sau | Delta Acc | F1 trước | F1 sau | Delta F1 | Recall trước | Recall sau | Delta Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast-w | 0.9298 | 0.9298 | +0.0000 | 0.9185 | 0.9185 | +0.0000 | 0.9052 | 0.9052 | +0.0000 |
| cleve | 0.8051 | 0.8051 | +0.0000 | 0.8015 | 0.8015 | +0.0000 | 0.8015 | 0.8015 | +0.0000 |
| crx | 0.8670 | 0.8655 | -0.0015 | 0.8659 | 0.8644 | -0.0015 | 0.8680 | 0.8670 | -0.0010 |
| diabetes | 0.7631 | 0.7605 | -0.0026 | 0.7291 | 0.7348 | +0.0057 | 0.7245 | 0.7364 | +0.0119 |
| german | 0.7320 | 0.7500 | +0.0180 | 0.6440 | 0.6916 | +0.0476 | 0.6372 | 0.6890 | +0.0519 |
| glass | 0.6563 | 0.6915 | +0.0352 | 0.5752 | 0.6544 | +0.0792 | 0.6503 | 0.7270 | +0.0767 |
| heart | 0.8296 | 0.8296 | +0.0000 | 0.8273 | 0.8273 | +0.0000 | 0.8292 | 0.8292 | +0.0000 |
| hepatitis | 0.8186 | 0.8064 | -0.0122 | 0.7282 | 0.7331 | +0.0049 | 0.7591 | 0.7741 | +0.0150 |
| horse | 0.8397 | 0.8370 | -0.0027 | 0.8302 | 0.8277 | -0.0025 | 0.8363 | 0.8342 | -0.0022 |
| iris | 0.9267 | 0.9267 | +0.0000 | 0.9223 | 0.9223 | +0.0000 | 0.9267 | 0.9267 | +0.0000 |
| labor | 0.9467 | 0.9467 | +0.0000 | 0.9457 | 0.9457 | +0.0000 | 0.9595 | 0.9595 | +0.0000 |
| led7 | 0.7259 | 0.7265 | +0.0006 | 0.7170 | 0.7172 | +0.0002 | 0.7238 | 0.7243 | +0.0004 |
| lymph | 0.8058 | 0.8806 | +0.0748 | 0.6864 | 0.8726 | +0.1862 | 0.4209 | 0.8770 | +0.4560 |
| sonar | 0.8071 | 0.8071 | +0.0000 | 0.8041 | 0.8041 | +0.0000 | 0.8061 | 0.8061 | +0.0000 |
| tic-tac-toe | 0.9729 | 0.9729 | +0.0000 | 0.9701 | 0.9701 | +0.0000 | 0.9700 | 0.9700 | +0.0000 |
| vehicle | 0.6465 | 0.6572 | +0.0107 | 0.6042 | 0.6227 | +0.0185 | 0.6497 | 0.6603 | +0.0106 |
| waveform | 0.8390 | 0.8366 | -0.0024 | 0.8383 | 0.8356 | -0.0027 | 0.8387 | 0.8363 | -0.0024 |
| wine | 0.9663 | 0.9663 | +0.0000 | 0.9685 | 0.9685 | +0.0000 | 0.9709 | 0.9709 | +0.0000 |
| zoo | 0.9490 | 0.9650 | +0.0160 | 0.9019 | 0.9273 | +0.0254 | 0.8715 | 0.8929 | +0.0213 |

Trung bình 19 dataset:

| Metric | Trước | Sau | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8330 | 0.8401 | +0.0070 |
| Macro-F1 | 0.8041 | 0.8231 | +0.0190 |
| Macro-Recall | 0.7973 | 0.8309 | +0.0336 |

Nhận xét:

- Macro-Recall tăng mạnh nhất, phù hợp mục tiêu giảm bỏ sót class thiểu số.
- Macro-F1 tăng rõ, cho thấy cải tiến không chỉ tăng recall mà vẫn giữ được cân bằng precision/recall.
- Accuracy tăng nhẹ nhưng ổn định, hợp lý vì baseline CMAR đã khá mạnh trên nhiều dataset cân bằng.
- Một số dataset giảm nhẹ accuracy như `hepatitis`, `crx`, `horse`, `waveform`; đây là trade-off cần trình bày rõ vì mục tiêu chính là cải thiện F1/Recall trên dữ liệu mất cân bằng.

---

## 6. Kết quả focused trên 7 dataset mất cân bằng

Bảng dưới dùng scope v13 focused:

- Trước: `result/v13_baseline_metrics.csv`, `result/v13_baseline_per_class.csv`
- Sau Layer 1: `result/v13_adaptive_metrics.csv`, `result/v13_adaptive_per_class.csv`

| Dataset | Accuracy trước | Accuracy sau | Delta Acc | F1 trước | F1 sau | Delta F1 | Recall trước | Recall sau | Delta Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lymph | 0.8058 | 0.8244 | +0.0186 | 0.6864 | 0.7958 | +0.1094 | 0.4209 | 0.7908 | +0.3699 |
| zoo | 0.9490 | 0.9573 | +0.0083 | 0.9019 | 0.9210 | +0.0191 | 0.8715 | 0.9001 | +0.0286 |
| glass | 0.6563 | 0.6563 | +0.0000 | 0.5752 | 0.5752 | +0.0000 | 0.6503 | 0.6503 | +0.0000 |
| hepatitis | 0.8186 | 0.8186 | +0.0000 | 0.7282 | 0.7235 | -0.0047 | 0.7591 | 0.7475 | -0.0116 |
| german | 0.7320 | 0.7510 | +0.0190 | 0.6440 | 0.6921 | +0.0481 | 0.6372 | 0.6888 | +0.0517 |
| vehicle | 0.6465 | 0.6572 | +0.0107 | 0.6042 | 0.6242 | +0.0200 | 0.6497 | 0.6604 | +0.0106 |
| breast-w | 0.9298 | 0.9298 | +0.0000 | 0.9185 | 0.9185 | +0.0000 | 0.9052 | 0.9052 | +0.0000 |

Trung bình 7 dataset:

| Metric | Trước | Sau Layer 1 | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7911 | 0.7992 | +0.0081 |
| Macro-F1 | 0.7226 | 0.7500 | +0.0274 |
| Macro-Recall | 0.6991 | 0.7633 | +0.0642 |

Nhận xét: Layer 1 chủ yếu đẩy Recall lên, đặc biệt trên Lymph. Khi thêm H4 và H5b vào final pipeline, Lymph tiếp tục tăng lên 0.8806 Accuracy và 0.8726 F1.

---

## 7. Phân tích chi tiết dataset Lymph

`lymph` là ví dụ tốt nhất để giải thích với giáo viên, vì baseline bỏ sót hoàn toàn class cực hiếm.

### 7.1. Tổng quan

| Metric | Baseline | Final Pipeline | Delta |
|---|---:|---:|---:|
| Accuracy | 0.8058 | 0.8806 | +0.0748 |
| Macro-F1 | 0.6864 | 0.8726 | +0.1862 |
| Macro-Recall | 0.4209 | 0.8770 | +0.4560 |

### 7.2. Per-class Lymph

| Class | Support | F1 trước | Recall trước | F1 sau | Recall sau | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| 2 | 81 | 0.8354 | 0.8148 | 0.8889 | 0.8889 | Majority cũng tăng |
| 3 | 61 | 0.8030 | 0.8689 | 0.8618 | 0.8689 | Giữ recall, tăng precision/F1 |
| 4 | 4 | 0.0000 | 0.0000 | 0.8571 | 0.7500 | Class cực hiếm được nhận diện |
| 1 | 2 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | Class cực hiếm được nhận diện hoàn toàn |

Điểm nên nói khi bảo vệ: baseline có Accuracy khá cao nhưng Macro-Recall rất thấp vì không bắt được class hiếm. Final pipeline làm Macro-Recall tăng từ 0.4209 lên 0.8770, tức là giải quyết đúng vấn đề mất cân bằng.

---

## 8. So sánh với paper CMAR 2001

Paper CMAR 2001 báo cáo accuracy trên 19 UCI dataset. Pipeline final hiện tại có trung bình:

```text
Accuracy final trung bình: 84.01%
Accuracy paper CMAR trung bình: 83.47%
Chênh lệch trung bình: +0.54%
Số dataset vượt paper: 12/19
Số dataset nằm trong khoảng +/-5% paper: 17/19
```

Một số dataset vượt paper đáng chú ý:

| Dataset | Accuracy final | Accuracy CMAR paper | Chênh lệch |
|---|---:|---:|---:|
| lymph | 88.06% | 82.43% | +5.63% |
| labor | 94.67% | 89.47% | +5.20% |
| waveform | 83.66% | 80.17% | +3.49% |
| sonar | 80.71% | 79.33% | +1.38% |
| german | 75.00% | 73.40% | +1.60% |
| zoo | 96.50% | 96.04% | +0.46% |

Lưu ý: so sánh với paper chỉ nên dùng như tham chiếu, vì tiền xử lý, discretization và môi trường chạy có thể khác. Giá trị quan trọng nhất trong project là so sánh baseline và final trên cùng source code, cùng `data_clean`, cùng 10-fold seed 42.

---

## 9. Các bug đã sửa trước khi lấy kết quả final

File liên quan:

- `src/EvalMetrics.java`
- `src/BorderlineSMOTE.java`
- `src/CMARClassifier.java`
- `src/FPGrowth.java`

| Bug | Trạng thái | Ý nghĩa |
|---|---|---|
| MacroF1 mean và stddev khác methodology | Đã sửa | MacroF1 và MacroF1Std cùng tính theo per-fold |
| Ghost class làm sai MacroF1 | Đã sửa | Class chỉ xuất hiện trong prediction nhưng support = 0 không tính vào denominator |
| Borderline DANGER threshold dùng floor thay vì ceil | Đã sửa | Dùng `ceil(k/2)` để xác định DANGER |
| Population stddev thay vì sample stddev | Đã sửa | Phù hợp báo cáo học thuật với k-fold |
| H2 condFreq per-class full filter gây OOM | Không giữ fix | Ghi nhận là trade-off; vẫn dùng global filter ở conditional tree để tránh tree explosion |

Ý nghĩa khi báo cáo: số liệu final đang dùng implementation sau bugfix, đáng tin cậy hơn các báo cáo v13/v14 cũ.

---

## 10. Tài liệu tham khảo và liên hệ với cải tiến

| Kỹ thuật trong project | Paper liên quan | Vai trò trong project |
|---|---|---|
| CMAR baseline | Li, W., Han, J., Pei, J. (2001). *CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules*. ICDM. | Thuật toán gốc: FP-Growth/CR-tree, rule pruning, weighted chi-square voting |
| H2 class-specific minSup | Liu, B., Ma, Y., Wong, C.K. (2000). *Improving an Association Rule Based Classifier*. PKDD. | Giảm support threshold cho class ít mẫu |
| SMOTE-N | Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR. | Cơ sở tạo synthetic sample cho dữ liệu categorical bằng mode voting |
| Borderline-SMOTE | Han, Wang, Mao (2005). *Borderline-SMOTE*. ICIC. | Chỉ oversample minority record ở vùng DANGER |
| Lift | Brin et al. (1997). *Dynamic Itemset Counting and Implication Rules*. SIGMOD. | Cơ sở cho measure Lift trong voting |
| Weighted rule combination/Lift usage | Bahri et al. (2020). *WEviRC*. Knowledge-Based Systems. | Liên quan đến ý tưởng kết hợp trọng số rule |
| Top-K/stratified rule idea | Ong et al. (2020), hướng iTCAR/top-k CAR mining | Cơ sở tham khảo cho H4/H5b top-K |

---

## 11. Câu trả lời ngắn gọn cho giáo viên

Nếu giáo viên hỏi: "Em cải tiến gì so với CMAR gốc?"

```text
Em giữ baseline CMAR gốc của Li, Han, Pei 2001, sau đó thêm pipeline 3 tầng:

1. Tăng sinh rule cho class thiểu số:
   - H2 class-specific minSup theo Liu 2000.
   - Adaptive Borderline-SMOTE-N theo Chawla 2002 và Han 2005.

2. Bảo vệ rule minority khi cắt tỉa:
   - H4 Stratified Coverage giữ top-K rule cho mỗi class.

3. Cải tiến cách vote khi nhiều rule match:
   - H5b confidence x Lift voting, có adaptive threshold theo imbalance ratio.

Kết quả trên 19 UCI dataset:
Accuracy 0.8330 -> 0.8401, Macro-F1 0.8041 -> 0.8231,
Macro-Recall 0.7973 -> 0.8309.

Trên Lymph, dataset mất cân bằng mạnh:
Accuracy 0.8058 -> 0.8806, Macro-F1 0.6864 -> 0.8726,
Macro-Recall 0.4209 -> 0.8770. Hai class cực hiếm trước đó F1 = 0,
sau cải tiến đã được nhận diện với F1 = 0.8571 và 1.0000.
```

Nếu giáo viên hỏi: "Dựa vào paper nào?"

```text
Baseline dựa trên CMAR 2001 của Li, Han, Pei.
H2 dựa trên Liu, Ma, Wong 2000 về class-specific minimum support.
SMOTE-N dựa trên Chawla et al. 2002.
Borderline-SMOTE dựa trên Han, Wang, Mao 2005.
Lift voting dựa trên measure Lift của Brin et al. 1997 và các hướng weighted rule combination.
Phần adaptive trigger và adaptive H5b threshold là đóng góp thực nghiệm trong project.
```

---

## 12. Giới hạn và việc cần làm tiếp

1. H2 hiện làm báo cáo phức tạp hơn vì minSup thay đổi theo class. Nếu muốn đơn giản hóa, cần chạy thêm benchmark no-H2 để xem Borderline-SMOTE + H4 + H5b có đủ giữ kết quả không.
2. Một số dataset giảm nhẹ accuracy, vì cải tiến ưu tiên Macro-F1 và Macro-Recall cho dữ liệu mất cân bằng.
3. Kết quả so sánh với paper CMAR 2001 chỉ là tham chiếu; so sánh chính nên là baseline vs final trong cùng repo.
4. Chưa nên xóa các report v13/v14/v15 cũ, nhưng khi nộp nên dùng file final này để tránh nhầm version.

---

## 13. Lệnh tái lập kết quả

```bash
javac -d out -encoding UTF-8 src/*.java
python preprocess_datasets.py
java -Xmx2g -cp out BenchmarkImbalanced
java -Xmx4g -cp out BenchmarkFinal
```

File kết quả chính:

```text
result/v13_baseline_metrics.csv
result/v13_adaptive_metrics.csv
result/final_baseline_metrics.csv
result/final_full_metrics.csv
result/final_baseline_per_class.csv
result/final_full_per_class.csv
```
