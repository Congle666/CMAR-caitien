# Báo cáo kiểm chứng lại source code và kết quả CMAR No-H2

Ngày kiểm chứng: 30/05/2026  
Phạm vi: đọc source Java, biên dịch lại, chạy lại benchmark và lấy số từ CSV mới sinh.  
Lưu ý quan trọng: báo cáo này không dùng nội dung từ các file `.md` cũ. Nguồn dữ liệu chỉ gồm source code, log chạy mới và CSV trong thư mục `result`.

## 1. Kết luận ngắn

Sau khi kiểm tra lại, pipeline chính đã bỏ H2: không còn dùng `minSup` động theo từng lớp trong benchmark chính. Code chính dùng global `minSup` theo dataset, dễ kiểm thử hơn và chạy ổn định hơn.

Không phát hiện lỗi làm giả số liệu Accuracy, Macro-F1 hoặc Recall. Tuy nhiên có một số điểm dễ gây hiểu nhầm hoặc làm báo cáo sai hình thức, đã sửa trước khi chạy lại:

- Nhãn benchmark tổng từng ghi "4 variants", trong khi code chính chỉ chạy 2 biến thể. Đã sửa thành "2 variants".
- Một số `catch` trước đó chỉ in lỗi rồi chạy tiếp, có thể che mất dataset lỗi. Đã đổi sang fail-fast bằng `RuntimeException`.
- Cột `records` trong CSV từng có nguy cơ bị trống. Đã sửa để ghi đúng số dòng dataset.
- Hàm tính macro recall trong benchmark đã lọc class có `support = 0`, tránh class chỉ xuất hiện ở prediction làm nhiễu recall.
- Nhãn per-class còn sót chữ `v13`; đã đổi thành `No-H2`.

## 2. Đường code đã kiểm chứng

Các điểm source quan trọng:

- `src/BenchmarkFinal.java:233`: `double h2 = 0.0;` nên benchmark chính tắt H2.
- `src/CrossValidator.java:122-134`: chỉ tạo và gán `classMinSupMap` khi `classMinSupFraction > 0`.
- `src/FPGrowth.java:69-70`: nếu `classMinSupMap == null` thì dùng `minSupport` global.
- `src/BenchmarkFinal.java:198-207`: bật H4/H5b trong classifier khi chạy full pipeline.
- `src/BenchmarkFinal.java:226-231`: chỉ bật Borderline-SMOTE khi class nhỏ nhất có tần suất dưới ngưỡng trigger.
- `src/CrossValidator.java:88-103`: chia train/test trước, sau đó mới áp dụng SMOTE trên `trainData`, không áp dụng lên test fold.
- `src/BenchmarkFinal.java:245` và `src/BenchmarkImbalanced.java:285`: nếu dataset lỗi thì dừng chương trình, không âm thầm bỏ qua.

Kết luận từ source: H2 vẫn còn trong code như khả năng legacy/ablation, nhưng code chính hiện tại không bật H2. Vì vậy hướng hiện tại là No-H2, dùng global minSup.

## 3. Các cải tiến đang có trong pipeline sau cải tiến

Baseline là CMAR cơ bản: khai phá luật kết hợp phân lớp bằng FP-Growth, sinh CAR, cắt tỉa bằng luật tổng quát, kiểm định chi-square, database coverage và phân loại bằng tập luật còn lại.

Pipeline sau cải tiến là No-H2 + 3 lớp:

1. Adaptive Borderline-SMOTE  
   Chỉ oversampling khi dataset/fold có lớp quá nhỏ. SMOTE được áp dụng sau khi đã chia fold, chỉ trên training data, nên không thấy rò rỉ dữ liệu test.

2. Stratified top-k coverage  
   Classifier giữ top-k luật theo hướng cân bằng lớp, giảm việc lớp lớn áp đảo lớp nhỏ khi bỏ phiếu.

3. Adaptive confidence x lift voting  
   Khi tỷ lệ mất cân bằng ban đầu đủ cao, classifier dùng thêm tín hiệu `confidence x lift` và giới hạn top-k vote để tăng chất lượng bỏ phiếu. Đây là cải tiến cục bộ trong code, không phải tái hiện nguyên văn một paper riêng.

## 4. Cơ sở bài báo liên quan

Các phần nền tảng có thể trình bày với giáo viên:

- CMAR gốc: Wenmin Li, Jiawei Han, Jian Pei, "CMAR: Accurate and Efficient Classification Based on Multiple Class-Association Rules", ICDM 2001.  
  Link: https://www.cs.sfu.ca/~jpei/publications/cmar.pdf

- FP-Growth: Jiawei Han, Jian Pei, Yiwen Yin, "Mining Frequent Patterns without Candidate Generation", SIGMOD 2000.  
  Link: https://experts.illinois.edu/en/publications/mining-frequent-patterns-without-candidate-generation/

- SMOTE: Nitesh V. Chawla, Kevin W. Bowyer, Lawrence O. Hall, W. Philip Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique", JAIR 2002.  
  Link: https://cse.usf.edu/~lohall/papers/smote.pdf

- Borderline-SMOTE: Hui Han, Wen-Yuan Wang, Bing-Huan Mao, "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning", LNCS 2005.  
  Link: https://cir.nii.ac.jp/crid/1360011145234473600?lang=en

## 5. Cách chạy kiểm chứng

Đã biên dịch lại toàn bộ source:

```powershell
javac -d out -encoding UTF-8 src\*.java
```

Kết quả: compile thành công.

Đã chạy lại benchmark tập mất cân bằng:

```powershell
java -Xmx4g -cp out BenchmarkImbalanced
```

Log mới:

- `result/verify_noh2_imbalanced_rerun.log`
- `result/verify_noh2_imbalanced_rerun.err`

Kết quả stderr: rỗng.

Đã chạy lại benchmark tổng 19 dataset:

```powershell
java -Xmx4g -cp out BenchmarkFinal
```

Log mới:

- `result/verify_noh2_final_rerun.log`
- `result/verify_noh2_final_rerun.err`

Kết quả stderr: rỗng.

## 6. Cách hiểu metric

Trong file metrics CSV:

- Accuracy lấy từ trung bình 10-fold.
- Macro-F1 lấy từ `EvalMetrics.average()`, tức trung bình Macro-F1 theo fold.
- Macro-Recall trong báo cáo này được tính lại từ file per-class CSV bằng cách lấy trung bình recall của các class có `support > 0`.

Điểm này cần nói rõ khi báo cáo, vì Macro-F1 và Macro-Recall không lấy từ cùng một dòng công thức trong CSV metrics.

## 7. Kết quả tổng 19 dataset

Trước cải tiến = baseline CMAR.  
Sau cải tiến = full No-H2 pipeline.

| Dataset | Acc trước | F1 trước | Recall trước | Acc sau | F1 sau | Recall sau | ΔAcc | ΔF1 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast-w | 0.9298 | 0.9185 | 0.9052 | 0.9298 | 0.9185 | 0.9052 | 0.0000 | 0.0000 | 0.0000 |
| cleve | 0.8051 | 0.8015 | 0.8015 | 0.8051 | 0.8015 | 0.8015 | 0.0000 | 0.0000 | 0.0000 |
| crx | 0.8670 | 0.8659 | 0.8680 | 0.8655 | 0.8646 | 0.8673 | -0.0015 | -0.0013 | -0.0007 |
| diabetes | 0.7631 | 0.7291 | 0.7245 | 0.7631 | 0.7291 | 0.7245 | 0.0000 | 0.0000 | 0.0000 |
| german | 0.7320 | 0.6440 | 0.6372 | 0.7290 | 0.6453 | 0.6398 | -0.0030 | +0.0013 | +0.0027 |
| glass | 0.6563 | 0.5752 | 0.6503 | 0.6915 | 0.6544 | 0.7270 | +0.0352 | +0.0792 | +0.0767 |
| heart | 0.8296 | 0.8273 | 0.8292 | 0.8296 | 0.8273 | 0.8292 | 0.0000 | 0.0000 | 0.0000 |
| hepatitis | 0.8186 | 0.7282 | 0.7591 | 0.7931 | 0.7102 | 0.7543 | -0.0255 | -0.0180 | -0.0048 |
| horse | 0.8397 | 0.8302 | 0.8363 | 0.8397 | 0.8302 | 0.8363 | 0.0000 | 0.0000 | 0.0000 |
| iris | 0.9267 | 0.9223 | 0.9267 | 0.9267 | 0.9223 | 0.9267 | 0.0000 | 0.0000 | 0.0000 |
| labor | 0.9467 | 0.9457 | 0.9595 | 0.9467 | 0.9457 | 0.9595 | 0.0000 | 0.0000 | 0.0000 |
| led7 | 0.7259 | 0.7170 | 0.7238 | 0.7265 | 0.7172 | 0.7243 | +0.0006 | +0.0002 | +0.0004 |
| lymph | 0.8058 | 0.6864 | 0.4209 | 0.8607 | 0.8549 | 0.8667 | +0.0549 | +0.1685 | +0.4458 |
| sonar | 0.8071 | 0.8041 | 0.8062 | 0.8071 | 0.8041 | 0.8062 | 0.0000 | 0.0000 | 0.0000 |
| tic-tac-toe | 0.9729 | 0.9701 | 0.9700 | 0.9729 | 0.9701 | 0.9700 | 0.0000 | 0.0000 | 0.0000 |
| vehicle | 0.6465 | 0.6042 | 0.6497 | 0.6477 | 0.6038 | 0.6508 | +0.0012 | -0.0004 | +0.0011 |
| waveform | 0.8390 | 0.8383 | 0.8387 | 0.8316 | 0.8306 | 0.8312 | -0.0074 | -0.0077 | -0.0075 |
| wine | 0.9663 | 0.9685 | 0.9709 | 0.9663 | 0.9685 | 0.9709 | 0.0000 | 0.0000 | 0.0000 |
| zoo | 0.9490 | 0.9019 | 0.8715 | 0.9650 | 0.9273 | 0.8929 | +0.0160 | +0.0254 | +0.0213 |

Trung bình 19 dataset:

| Metric | Trước cải tiến | Sau cải tiến | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.8330 | 0.8367 | +0.0037 |
| Macro-F1 | 0.8041 | 0.8171 | +0.0130 |
| Macro-Recall | 0.7973 | 0.8255 | +0.0282 |

Số dataset cải thiện trên 19 dataset:

| Metric | Tăng | Giảm |
|---|---:|---:|
| Accuracy | 5/19 | 4/19 |
| Macro-F1 | 5/19 | 4/19 |
| Macro-Recall | 6/19 | 3/19 |

Nhận xét: cải tiến không làm mọi dataset tăng. Tác động mạnh nhất nằm ở dataset mất cân bằng như `lymph`, `glass`, `zoo`. Một số dataset cân bằng hoặc không kích hoạt SMOTE gần như giữ nguyên.

## 8. Kết quả riêng 7 dataset mất cân bằng

| Dataset | Acc trước | F1 trước | Recall trước | Acc sau | F1 sau | Recall sau | ΔAcc | ΔF1 | ΔRecall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| breast-w | 0.9298 | 0.9185 | 0.9052 | 0.9298 | 0.9185 | 0.9052 | 0.0000 | 0.0000 | 0.0000 |
| german | 0.7320 | 0.6440 | 0.6372 | 0.7290 | 0.6453 | 0.6398 | -0.0030 | +0.0013 | +0.0027 |
| glass | 0.6563 | 0.5752 | 0.6503 | 0.6915 | 0.6544 | 0.7270 | +0.0352 | +0.0792 | +0.0767 |
| hepatitis | 0.8186 | 0.7282 | 0.7591 | 0.7931 | 0.7102 | 0.7543 | -0.0255 | -0.0180 | -0.0048 |
| lymph | 0.8058 | 0.6864 | 0.4209 | 0.8607 | 0.8549 | 0.8667 | +0.0549 | +0.1685 | +0.4458 |
| vehicle | 0.6465 | 0.6042 | 0.6497 | 0.6477 | 0.6038 | 0.6508 | +0.0012 | -0.0004 | +0.0011 |
| zoo | 0.9490 | 0.9019 | 0.8715 | 0.9650 | 0.9273 | 0.8929 | +0.0160 | +0.0254 | +0.0213 |

Trung bình 7 dataset mất cân bằng:

| Metric | Trước cải tiến | Sau cải tiến | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 0.7911 | 0.8024 | +0.0113 |
| Macro-F1 | 0.7226 | 0.7592 | +0.0366 |
| Macro-Recall | 0.6991 | 0.7767 | +0.0775 |

Số dataset cải thiện trên 7 dataset mất cân bằng:

| Metric | Số dataset tăng |
|---|---:|
| Accuracy | 4/7 |
| Macro-F1 | 4/7 |
| Macro-Recall | 5/7 |

Nhận xét: nhóm mất cân bằng là nơi cải tiến có ý nghĩa nhất. Macro-Recall tăng mạnh vì các lớp nhỏ trong `lymph`, `glass`, `zoo` được dự đoán tốt hơn sau khi dùng adaptive Borderline-SMOTE và voting có kiểm soát.

## 9. Điểm nên trình bày với giáo viên

Nội dung có thể nói gọn:

- Bài gốc CMAR dùng nhiều luật kết hợp phân lớp và cắt tỉa bằng chi-square/database coverage.
- Em giữ nền CMAR nhưng bỏ H2 vì H2 dùng minSup riêng theo từng class, khó kiểm thử, chạy lâu và khó giải thích ổn định.
- Thay H2 bằng hướng No-H2:
  - vẫn dùng global minSup theo dataset,
  - thêm Borderline-SMOTE thích ứng cho tập mất cân bằng,
  - thêm chọn top-k luật theo lớp,
  - thêm bỏ phiếu dựa trên confidence và lift khi dataset mất cân bằng.
- SMOTE chỉ áp dụng trong training fold nên không rò rỉ test data.
- Kết quả tổng 19 dataset tăng nhẹ: Accuracy +0.0037, Macro-F1 +0.0130, Macro-Recall +0.0282.
- Kết quả trên 7 dataset mất cân bằng tăng rõ hơn: Accuracy +0.0113, Macro-F1 +0.0366, Macro-Recall +0.0775.

## 10. Hạn chế còn lại

- Đây là benchmark 10-fold với seed cố định, chưa phải repeated cross-validation nhiều seed.
- Chưa có unit test riêng cho FP-Growth, CR-tree pruning và từng luật voting.
- H2 vẫn còn trong source để phục vụ ablation/legacy, nhưng không được bật ở code chính.
- Một số dataset giảm nhẹ sau cải tiến, ví dụ `hepatitis`, `waveform`, `crx`. Vì vậy nên trình bày rằng cải tiến tập trung xử lý mất cân bằng, không phải cải thiện đồng đều mọi dataset.

## 11. Kết luận cuối

Sau khi kiểm tra lại bằng source và chạy lại benchmark, số liệu hợp lệ để báo cáo là:

- 19 dataset: Accuracy 0.8330 -> 0.8367, Macro-F1 0.8041 -> 0.8171, Macro-Recall 0.7973 -> 0.8255.
- 7 dataset mất cân bằng: Accuracy 0.7911 -> 0.8024, Macro-F1 0.7226 -> 0.7592, Macro-Recall 0.6991 -> 0.7767.

Kết luận kỹ thuật: nên giữ hướng No-H2 trong code chính vì dễ chạy, dễ test và vẫn cải thiện rõ trên nhóm dataset mất cân bằng.
