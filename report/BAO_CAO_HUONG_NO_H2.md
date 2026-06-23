# Báo cáo thử nghiệm hướng không dùng H2

**Mục tiêu:** Kiểm tra có thể bỏ H2 class-specific minSup hay không.  
**Lý do:** H2 làm pipeline khó giải thích hơn vì minSup thay đổi theo class/training fold, đồng thời có thể làm mining phức tạp và khó test.  
**Code chính đã cập nhật:** `src/BenchmarkFinal.java`, `src/BenchmarkImbalanced.java`  
**Benchmark kiểm chứng phụ:** `src/BenchmarkNoH2.java`  
**Kết quả sinh ra:** `result/final_*`, `result/main_noh2_*`, `result/noh2_*`

---

## 0. Cập nhật sau khi đưa No-H2 vào code chính

Ngày 30/05/2026, pipeline chính đã được chuyển sang hướng **không dùng H2**:

- `BenchmarkFinal` đặt `classMinSupFraction = 0.0`, tức là không còn minSup động theo class.
- `BenchmarkImbalanced` gọi các biến thể chính với `useH2 = false`.
- `CrossValidator` vẫn giữ tham số H2 để chạy ablation nếu cần, nhưng code chính không bật H2.

Kết quả verify code chính:

| Benchmark | Baseline Acc | Full Acc | Baseline F1 | Full F1 | Baseline Recall | Full Recall | Delta chính |
|---|---:|---:|---:|---:|---:|---:|---|
| 19 UCI (`BenchmarkFinal`) | 0.8330 | 0.8367 | 0.8041 | 0.8171 | 0.7973 | 0.8255 | F1 +0.0130, Recall +0.0282 |
| 7 imbalanced (`BenchmarkImbalanced`) | 0.7911 | 0.8024 | 0.7226 | 0.7592 | 0.6991 | 0.7767 | F1 +0.0366, Recall +0.0775 |

File verify mới:

- `result/final_baseline_metrics.csv`
- `result/final_full_metrics.csv`
- `result/final_baseline_per_class.csv`
- `result/final_full_per_class.csv`
- `result/final_noh2_benchmark.log`
- `result/main_noh2_*`

Kết luận cập nhật: **No-H2 là hướng đang dùng trong code chính**. H2 chỉ nên trình bày như một thử nghiệm ablation/optional, không phải đóng góp chính.

---

## 1. Kết luận nhanh

Có thể bỏ H2 nếu mục tiêu là làm pipeline dễ bảo vệ hơn:

```text
CMAR global minSup gốc
+ Adaptive Borderline-SMOTE-N
+ H4 Stratified Coverage
+ H5b Adaptive confidence x Lift voting
```

Kết quả no-H2 vẫn cải thiện rõ so với baseline trên 7 dataset mất cân bằng:

| Metric | Baseline | No-H2 Full | Tăng |
|---|---:|---:|---:|
| Accuracy | 0.7911 | 0.8024 | +0.0113 |
| Macro-F1 | 0.7226 | 0.7592 | +0.0366 |
| Macro-Recall | 0.6991 | 0.7767 | +0.0775 |

So với bản có H2 đã thử trước đó, no-H2 có thể thấp hơn trên một số dataset như `german`, `vehicle`, `hepatitis`. Tuy nhiên code chính hiện chọn no-H2 vì đơn giản hơn và vẫn tăng rõ Macro-F1/Macro-Recall.

- Hướng báo cáo chính: **No-H2 + Adaptive Borderline-SMOTE-N + H4 + H5b**.
- Hướng H2: chỉ đưa vào phần ablation hoặc thảo luận hạn chế.

---

## 2. Pipeline no-H2 đã chạy

Khác biệt chính:

| Thành phần | Pipeline final cũ | Hướng no-H2 |
|---|---|---|
| minSupport | H2 theo từng class | Global minSup gốc của CMAR |
| SMOTE | Adaptive Borderline-SMOTE khi `min_class_freq < 5` | Giữ nguyên |
| H4 | Có | Giữ nguyên |
| H5b | Có adaptive threshold | Giữ nguyên |

Trong code, no-H2 được đảm bảo bằng cách truyền:

```java
classMinSupFraction = 0.0
```

Khi đó `CrossValidator` không tạo `classMinSupMap`, và `FPGrowth` dùng global minSupport như CMAR gốc.

---

## 3. Kết quả chi tiết no-H2 trên 7 dataset mất cân bằng

Nguồn số liệu:

- `result/noh2_baseline_metrics.csv`
- `result/noh2_full_metrics.csv`
- `result/noh2_baseline_per_class.csv`
- `result/noh2_full_per_class.csv`

| Dataset | Acc baseline | Acc no-H2 | F1 baseline | F1 no-H2 | Recall baseline | Recall no-H2 | Nhận xét |
|---|---:|---:|---:|---:|---:|---:|---|
| lymph | 0.8058 | 0.8607 | 0.6864 | 0.8549 | 0.4209 | 0.8667 | Tăng rất mạnh |
| zoo | 0.9490 | 0.9650 | 0.9019 | 0.9273 | 0.8715 | 0.8929 | Tăng |
| glass | 0.6563 | 0.6915 | 0.5752 | 0.6544 | 0.6503 | 0.7270 | Tăng mạnh |
| hepatitis | 0.8186 | 0.7931 | 0.7282 | 0.7102 | 0.7591 | 0.7543 | Giảm |
| german | 0.7320 | 0.7290 | 0.6440 | 0.6453 | 0.6372 | 0.6398 | Gần như ngang |
| vehicle | 0.6465 | 0.6477 | 0.6042 | 0.6038 | 0.6497 | 0.6508 | Gần như ngang |
| breast-w | 0.9298 | 0.9298 | 0.9185 | 0.9185 | 0.9052 | 0.9052 | Không đổi |

Trung bình:

| Metric | Baseline | No-H2 Full | Delta |
|---|---:|---:|---:|
| Accuracy | 0.7911 | 0.8024 | +0.0113 |
| Macro-F1 | 0.7226 | 0.7592 | +0.0366 |
| Macro-Recall | 0.6991 | 0.7767 | +0.0775 |

---

## 4. So sánh no-H2 với pipeline final có H2

**Lưu ý quan trọng:** Phần này là snapshot lịch sử trước khi chuyển code chính sang No-H2. Hiện tại `result/final_full_metrics.csv` đã được ghi lại bằng pipeline No-H2, không còn đại diện cho bản có H2.

Nguồn so sánh:

- No-H2: `result/noh2_full_metrics.csv`
- Final có H2: kết quả cũ trong báo cáo/nhật ký trước khi ghi lại `result/final_*`

| Dataset | Acc no-H2 | Acc final có H2 | Chênh Acc | F1 no-H2 | F1 final có H2 | Chênh F1 |
|---|---:|---:|---:|---:|---:|---:|
| lymph | 0.8607 | 0.8806 | -0.0199 | 0.8549 | 0.8726 | -0.0177 |
| zoo | 0.9650 | 0.9650 | +0.0000 | 0.9273 | 0.9273 | +0.0000 |
| glass | 0.6915 | 0.6915 | +0.0000 | 0.6544 | 0.6544 | +0.0000 |
| hepatitis | 0.7931 | 0.8064 | -0.0133 | 0.7102 | 0.7331 | -0.0229 |
| german | 0.7290 | 0.7500 | -0.0210 | 0.6453 | 0.6916 | -0.0463 |
| vehicle | 0.6477 | 0.6572 | -0.0095 | 0.6038 | 0.6227 | -0.0189 |
| breast-w | 0.9298 | 0.9298 | +0.0000 | 0.9185 | 0.9185 | +0.0000 |

Nhận xét:

- H2 giúp rõ nhất trên `german`, `vehicle`, `hepatitis`.
- No-H2 vẫn giữ được gain rất lớn trên `lymph`, vì Borderline-SMOTE + H4/H5b đã đủ mạnh.
- Trên `zoo`, `glass`, `breast-w`, no-H2 gần như bằng pipeline final.

---

## 5. Phân tích Lymph khi bỏ H2

No-H2 vẫn xử lý tốt class hiếm trong `lymph`:

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| 2 | 81 | 0.8750 | 0.8642 | 0.8696 |
| 3 | 61 | 0.8254 | 0.8525 | 0.8387 |
| 4 | 4 | 1.0000 | 0.7500 | 0.8571 |
| 1 | 2 | 1.0000 | 1.0000 | 1.0000 |

Điểm quan trọng: hai class cực hiếm vẫn được nhận diện tốt dù không dùng H2.

---

## 6. Khuyến nghị

Tôi đề xuất có 2 hướng:

### Hướng A: Giữ H2 nếu muốn số liệu tốt nhất

Dùng báo cáo `BAO_CAO_FINAL_CMAR_CAI_TIEN.md`.

Ưu điểm:

- Kết quả trung bình tốt hơn.
- German/Vehicle/Hepatitis tốt hơn.
- Có đầy đủ paper backing từ Liu 2000.

Nhược điểm:

- Khó giải thích vì minSup thay đổi theo class/fold.
- Dễ bị hỏi sâu về tại sao chọn công thức và tại sao không quá khớp.

### Hướng B: Bỏ H2 nếu muốn pipeline dễ bảo vệ hơn

Dùng hướng no-H2:

```text
CMAR global minSup
+ Adaptive Borderline-SMOTE-N
+ H4 Stratified Coverage
+ H5b Adaptive confidence x Lift voting
```

Ưu điểm:

- Giữ minSupport toàn cục giống CMAR gốc.
- Câu chuyện đơn giản hơn: không đổi rule mining threshold, chỉ xử lý mất cân bằng bằng data-level oversampling và cải tiến pruning/voting.
- Vẫn cải thiện: Accuracy +1.13%, Macro-F1 +3.66%, Macro-Recall +7.75% trên 7 dataset mất cân bằng.

Nhược điểm:

- Kém pipeline final có H2 trên một số dataset moderate imbalance.

### Kết luận thực dụng

Sau khi verify lại code chính, nên chọn **Hướng B: bỏ H2 khỏi câu chuyện chính**. H2 có thể nhắc như một ablation/optional technique đã thử nhưng không chọn vì làm pipeline phức tạp, khó test và khó giải thích do minSup thay đổi theo class/fold.

Khi đó đóng góp chính sẽ là:

1. Adaptive Borderline-SMOTE-N cho categorical data.
2. H4 Stratified Coverage để bảo vệ rule minority.
3. H5b Adaptive confidence x Lift voting.

---

## 7. Lệnh chạy lại

```bash
javac -d out -encoding UTF-8 src/*.java
java -Xmx4g -cp out BenchmarkFinal
java -Xmx3g -cp out BenchmarkImbalanced
java -Xmx3g -cp out BenchmarkNoH2
```

Output:

```text
result/final_baseline_metrics.csv
result/final_baseline_per_class.csv
result/final_full_metrics.csv
result/final_full_per_class.csv
result/main_noh2_baseline_metrics.csv
result/main_noh2_full_metrics.csv
result/noh2_baseline_metrics.csv
result/noh2_baseline_per_class.csv
result/noh2_smote_metrics.csv
result/noh2_smote_per_class.csv
result/noh2_full_metrics.csv
result/noh2_full_per_class.csv
result/noh2_benchmark.log
```
