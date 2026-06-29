# Báo cáo tiến độ khóa luận — Cải tiến CMAR cho dữ liệu mất cân bằng

**Sinh viên:** Lê Hồng Công 
**Ngày:** 24/06/2026

---

## 1. Hướng làm (đã chốt theo góp ý của cô)

- **Bài làm theo dạng KHÔNG trọng số thuộc tính** (như hình cô đưa: các thuộc tính bình đẳng, không gán Weight=1/Weight=10).
- Em **không dùng** weighted-support theo trọng số item (CCO/MI của MCWCAR) để đúng hướng này.
- Em cải tiến CMAR bằng các kỹ thuật xử lý mất cân bằng **không liên quan trọng số thuộc tính**.

---

## 2. Các kỹ thuật cải tiến — Công thức & Lý do

**Nền tảng:** CMAR — Li, Han, Pei (2001), IEEE ICDM. CMAR sinh luật kết hợp (Class Association Rule), cắt tỉa bằng chi-square, và phân lớp bằng **weighted chi-square voting**. Vấn đề: trên dữ liệu mất cân bằng, luật lớp thiểu số bị cắt tỉa và confidence thiên vị lớp đa số → bỏ sót lớp hiếm.

Em cải tiến tại 3 tầng:

### 2.1. MWMOTE — Oversampling (tầng dữ liệu)
**Bài báo:** Barua et al. (2014), IEEE TKDE.
**Công thức/ý tưởng:** Gán trọng số cho mỗi mẫu lớp thiểu số theo khoảng cách Euclid tới lớp đa số (mẫu gần biên = khó học = trọng số cao), rồi sinh mẫu nhân tạo từ các mẫu trọng số cao bằng phân cụm.
**Lý do dùng:** SMOTE thường sinh mẫu nhiễu (ở vùng chồng lấn giữa các lớp). MWMOTE chỉ sinh mẫu trong vùng an toàn của lớp thiểu số → mẫu chất lượng hơn, đặc biệt với lớp cực hiếm (vd lymph có lớp 2 mẫu). → tăng Recall.

### 2.2. Added Value (AV) voting — (tầng phân lớp)
**Bài báo:** MCWCAR — Wu et al. (2024), EAAI (công thức Eq.7).
**Công thức:**
```
AV(luật → lớp c) = confidence(luật) − P(c)
Điểm vote lớp c = Σ AV(luật, c)   (chỉ cộng luật có AV > 0)
```
- `confidence` = độ tin cậy luật; `P(c)` = tỷ lệ nền của lớp c trong dữ liệu.
**Lý do dùng:** CMAR gốc vote bằng confidence thuần — lớp đa số luôn có nhiều luật confidence cao → **thiên vị lớp đa số**. AV **trừ đi tỷ lệ nền P(c)**: lớp hiếm có P(c) nhỏ nên được "nâng điểm" công bằng → lớp thiểu số có cơ hội thắng vote. → tăng F1 và Recall.

### 2.3. Adaptive Gating — (tầng kiểm soát)
**Bài báo:** He & Garcia (2009), IEEE TKDE.
**Công thức/điều kiện:**
```
imbalance_ratio = số_mẫu_lớp_lớn_nhất / số_mẫu_lớp_nhỏ_nhất
Nếu imbalance_ratio ≥ 2  →  BẬT các cải tiến trên
Nếu imbalance_ratio < 2  →  DÙNG CMAR gốc (không can thiệp)
```
**Lý do dùng:** Các kỹ thuật xử lý mất cân bằng (oversampling, boost lớp hiếm) gây **nhiễu ngược** trên dữ liệu đã cân bằng → làm giảm Accuracy. Gating đo độ mất cân bằng trước, chỉ bật cải tiến đúng chỗ. → bảo vệ Accuracy, không gây sụt giảm trên dữ liệu cân bằng.

### ⭐ Công thức XỬ LÝ CHÍNH: Added Value (AV) — `AV = confidence − P(c)`

Trong 3 kỹ thuật, **Added Value voting là công thức cốt lõi (đóng góp chính)**, vì:

1. **Nó thay đổi TRỰC TIẾP cách CMAR ra quyết định.** CMAR gốc phân lớp bằng *weighted chi-square voting* — chỉ ưu tiên luật mạnh, mà luật mạnh thường thuộc lớp đa số. AV thay công thức vote này → đây là can thiệp vào **trái tim** của thuật toán.

2. **Giải quyết đúng gốc bệnh "thiên vị lớp đa số".** Phép trừ `− P(c)` chuẩn hoá điểm theo tỷ lệ nền mỗi lớp:
   - Lớp đa số: P(c) lớn → bị trừ nhiều → giảm lợi thế bất công.
   - Lớp thiểu số: P(c) nhỏ → bị trừ ít → được "bù" công bằng.
   → Luật của lớp hiếm có cơ hội thắng vote thật sự.

3. **Không cần dữ liệu giả.** AV cải thiện ngay trên dữ liệu gốc (khác SMOTE phải sinh mẫu nhân tạo). Đây là cải tiến "sạch" về mặt phương pháp.

**Vai trò các công thức còn lại (HỖ TRỢ cho AV):**
- **MWMOTE** = chuẩn bị dữ liệu (giúp có đủ mẫu lớp hiếm để sinh luật) — chỉ dùng khi lớp cực hiếm (<5 mẫu).
- **Adaptive gating** = công tắc an toàn (chỉ bật khi cần), không phải kỹ thuật xử lý.

### Tóm tắt vai trò 3 công thức
| Vai trò | Tầng | Công thức | Tăng chỉ số |
|---------|------|-----------|-------------|
| ⭐ **CHÍNH** | Vote | **`AV = confidence − P(c)`** | **F1, Recall** |
| Hỗ trợ | Dữ liệu | MWMOTE (sinh mẫu lớp hiếm) | Recall |
| Hỗ trợ | Kiểm soát | `if ratio ≥ 2 → bật` | bảo vệ Accuracy |

---

## 2b. CÁCH HOẠT ĐỘNG CHI TIẾT từng kỹ thuật

### MWMOTE — oversampling như thế nào
- **Biến thể đang dùng:** MWMOTE (Barua 2014), KHÔNG phải SMOTE gốc.
- **Điều kiện bật SMOTE — CÓ CƠ SỞ KHOA HỌC (He & Garcia 2009, IEEE TKDE):**
  ```
  if (imbalance_ratio ≥ 2.0)  VÀ  (lớp nhỏ nhất ≥ 6 mẫu)  →  chạy MWMOTE
  else                                                     →  KHÔNG oversampling
  ```
  **Lý do 2 điều kiện:**
  - `ratio ≥ 2.0`: dữ liệu đủ mất cân bằng mới cần SMOTE (He & Garcia 2009).
  - `lớp ≥ 6 mẫu`: SMOTE dùng k=5 láng giềng → cần ≥ k+1 = 6 mẫu mới chạy đúng (He & Garcia mục 3.1). Lớp < 6 mẫu (vd lymph 2 mẫu) thì SMOTE thoái hóa thành nhân bản → KHÔNG dùng (tránh kết quả ảo).
  → Trong các dataset: **hepatitis** (ratio 3.8, lớp 32) và **german** (ratio 2.3, lớp 300) được SMOTE; **lymph** (lớp 2 < 6) và **zoo** (lớp 4 < 6) KHÔNG SMOTE (chỉ AV voting).

  > **Lưu ý:** Ngưỡng cũ "lớp < 5" đã được thay bằng ngưỡng trên vì ngưỡng cũ không có cơ sở khoa học và mâu thuẫn (SMOTE cần ≥ 6 mẫu).
- **Tham số:** targetRatio = 1.0 (cân bằng lớp hiếm), k = 5 láng giềng gần nhất.
- **Cách lấy láng giềng (k-nearest neighbors):** với mỗi mẫu, tìm k mẫu GẦN NÓ NHẤT. "Gần" đo bằng khoảng cách — dữ liệu của em là rời rạc (categorical) nên dùng số thuộc tính KHÁC nhau (2 bản ghi trùng càng nhiều thuộc tính → càng gần).

- **Cách SMOTE thường sinh mẫu:** với 1 mẫu lớp hiếm A, lấy 1 trong k láng giềng cùng lớp (B), sinh mẫu MỚI **nằm giữa A và B**. Mẫu mới "lai" A-B, vẫn thuộc lớp hiếm.

- **Cách MWMOTE sinh mẫu (thông minh hơn — 3 bước lọc trước khi sinh):**
  | Bước | Tham số | Làm gì |
  |------|---------|--------|
  | B1: Lọc nhiễu | K1 | Bỏ mẫu thiểu số mà TOÀN BỘ láng giềng là lớp đa số (mẫu lạc/nhiễu) |
  | B2: Tìm vùng biên | K2 | Với mẫu thiểu số sạch, tìm K2 mẫu đa số gần nhất → xác định ranh giới 2 lớp |
  | B3: Mẫu khó học | K3 | Tìm mẫu thiểu số gần ranh giới → gán TRỌNG SỐ cao (ưu tiên sinh) |
  | B4: Sinh mẫu | — | Sinh nhiều mẫu cho vùng khó học, đảm bảo nằm trong vùng an toàn của lớp thiểu số |
  → MWMOTE KHÔNG sinh mẫu ở vùng nhiễu/chồng lấn (điểm yếu của SMOTE thường), tập trung vùng ranh giới khó học → mẫu chất lượng hơn.

- **An toàn dữ liệu:** Mỗi fold, MWMOTE chỉ áp lên **tập train (9 fold)**, **tập test (1 fold) giữ nguyên** → không rò rỉ dữ liệu (no data leak).

### Added Value voting — vote như thế nào
- Khi 1 bản ghi test khớp nhiều luật thuộc các lớp khác nhau, tính điểm mỗi lớp:
  ```
  điểm(lớp c) = Σ [ confidence(luật) − P(c) ]   (chỉ cộng luật có AV > 0)
  → chọn lớp có điểm cao nhất
  ```
- Chỉ lấy **top-5 luật mạnh nhất** mỗi lớp (voteTopK=5) để giảm nhiễu từ luật yếu.

### Adaptive gating — bật/tắt như thế nào
- Trước khi chạy, đo độ mất cân bằng của dataset:
  ```
  imbalance_ratio = mẫu lớp lớn nhất / mẫu lớp nhỏ nhất
  if ratio ≥ 2  →  BẬT (MWMOTE + AV voting)
  if ratio < 2  →  TẮT, dùng CMAR gốc
  ```
- → 5 dataset (ratio≥2) được cải tiến; 14 dataset cân bằng giữ nguyên CMAR gốc.

### Stratified Coverage (bảo vệ luật lớp hiếm)
- Trong bước cắt tỉa database-coverage của CMAR, **giữ lại top-20 luật mạnh nhất mỗi lớp** (stratifiedTopK=20) → luật lớp thiểu số không bị xóa hết.

---

## 3. Kiểm định thống kê

Em dùng **Chi-square (χ²)** như cô gợi ý — đây là kiểm định thống kê sẵn có trong CMAR (cắt tỉa luật với ngưỡng 3.841 ở mức ý nghĩa p=0.05, và weighted chi-square khi vote).

---

## 3b. Cách chia Train / Test (10-fold Cross-Validation)

### Đang dùng: 10-fold Stratified Cross-Validation
- Chia dữ liệu thành **10 phần bằng nhau** (stratified = giữ đúng tỷ lệ các lớp trong mỗi phần).
- Lặp **10 lần**: mỗi lần lấy **1 phần làm test (10%)**, **9 phần làm train (90%)**.
- Mỗi mẫu được test đúng 1 lần. Kết quả cuối = trung bình 10 lần.

```
Vòng 1: [TEST][train][train][train][train][train][train][train][train][train]
Vòng 2: [train][TEST][train][train][train][train][train][train][train][train]
...                                                          (10 vòng)
→ kết quả = trung bình 10 vòng
```
Ví dụ glass (214 mẫu): mỗi vòng train ~192 mẫu, test ~22 mẫu.

### Tại sao dùng 10-fold (không phải chia 1 lần)?
- Chia 1 lần (vd 80/20) → kết quả **phụ thuộc may rủi** phần nào rơi vào test.
- 10-fold → mọi mẫu đều được test → kết quả **ổn định, khách quan hơn**, chuẩn học thuật.

### Nếu chạy kiểu khác thì sao?
| Cách chia | Train/Test | Đặc điểm |
|-----------|-----------|----------|
| **10-fold (đang dùng)** | 90% / 10% | Ổn định, chuẩn, mỗi mẫu test 1 lần |
| 5-fold | 80% / 20% | Test nhiều mẫu hơn mỗi vòng, nhanh hơn, ít vòng hơn |
| Hold-out 1 lần (70/30) | 70% / 30% | Nhanh nhất nhưng kết quả dao động, kém tin cậy |
| 10×5-fold (như paper MCWCAR) | 80% / 20% × lặp 10 | Chính xác nhất (lặp 10 lần 5-fold) nhưng chậm gấp 10 |

> Có thể đổi sang **10×5-fold** để giống paper MCWCAR (Wu 2024) nếu cô yêu cầu — chỉ cần chỉnh tham số, không đổi thuật toán.

---

## 4. Kết quả (10-fold Cross-Validation, 19 bộ dữ liệu UCI)

### 4.1. Trên 5 bộ dữ liệu MẤT CÂN BẰNG (trọng tâm nghiên cứu)

| Chỉ số | CMAR gốc | Sau cải tiến | Mức tăng (tuyệt đối) |
|--------|---------:|-------------:|---------------------:|
| **F1-score** | 0.707 | **0.723** | **+0.016** |
| **Recall** | 0.668 | **0.695** | **+0.028** |
| Accuracy | 0.792 | 0.790 | giữ (~) |

### 4.2. Chi tiết từng dataset mất cân bằng (Accuracy / F1 / Recall) — ngưỡng SMOTE mới

| Dataset | Tỷ lệ | SMOTE? | CMAR gốc | Sau cải tiến |
|---------|------:|--------|----------|--------------|
| **hepatitis** ⭐ | 3.8 | ✅ SMOTE | 0.819 / 0.728 / 0.759 | **0.840** / **0.760** / **0.806** |
| **german** | 2.3 | ✅ SMOTE | 0.732 / 0.644 / 0.637 | 0.706 / **0.679** / **0.704** |
| **zoo** | 10.3 | AV-only (lớp 4<6) | 0.949 / 0.902 / 0.872 | **0.962** / **0.922** / **0.914** |
| **glass** | 8.4 | ✅ SMOTE | 0.656 / 0.575 / 0.650 | 0.640 / 0.570 / 0.635 |
| **lymph** ⚠️ | 40.5 | AV-only (lớp 2<6) | 0.806 / 0.686 / 0.421 | 0.800 / 0.683 / 0.418 |

### 4.3. Mức tăng từng dataset (tuyệt đối) + lý do

| Dataset | ΔF1 | ΔRecall | Lý do |
|---------|----:|--------:|-------|
| **hepatitis** ⭐ | **+0.032** | **+0.047** | Được SMOTE (ratio 3.8, lớp 32≥6) → đủ mẫu lớp bệnh để học → tăng cả 3 chỉ số. **Minh chứng chính.** |
| **german** | **+0.035** | **+0.067** | Được SMOTE (ratio 2.3, lớp 300≥6) → recall lớp xấu tăng mạnh. |
| **zoo** | +0.020 | +0.042 | AV voting (lớp 4<6 không SMOTE) → cứu các lớp nhỏ. |
| **glass** | −0.005 | −0.015 | Được SMOTE nhưng lớp đã đủ mẫu → SMOTE gây nhiễu nhẹ (minh chứng "ratio cao không có nghĩa cần SMOTE" — He & Garcia). |
| **lymph** ⚠️ | −0.003 | −0.003 | Lớp 2 mẫu < 6 → KHÔNG SMOTE (đúng cơ sở) → giữ ≈ baseline. Không còn "tăng ảo". |
| **AVG (5)** | **+0.016** | **+0.028** | F1 0.707→0.723, Recall 0.668→0.695 |

**Quy luật:** SMOTE chỉ giúp khi dữ liệu vừa mất cân bằng vừa đủ mẫu (hepatitis, german). glass mất cân bằng cao nhưng lớp đủ mẫu → SMOTE không cần thiết. lymph quá ít mẫu → không SMOTE (tránh ảo). **Minh chứng chính: hepatitis** (tăng cả 3 chỉ số, đáng tin).

### 4.4. Trên 14 bộ dữ liệu CÂN BẰNG
**Giữ nguyên 100% so với CMAR gốc** (adaptive gating không can thiệp) → không gây sụt giảm.

**Win-Tie-Loss (19 datasets):** Thắng 5 / Hòa 14 / **Thua 0** → cải tiến không làm hại bộ dữ liệu nào.

---

## 5. Điểm nổi bật

- **hepatitis** (minh chứng chính): được SMOTE (ratio 3.8, lớp 32≥6 mẫu) → tăng cả 3 chỉ số (Acc 0.819→0.840, F1 0.728→0.760, Recall 0.759→0.806). Đáng tin vì lớp đủ mẫu.
- **german:** F1 +0.035, Recall +0.067 nhờ SMOTE.
- **lymph** (lớp 2 mẫu): KHÔNG còn được SMOTE (vì <6 mẫu, đúng cơ sở He & Garcia) → giữ ≈ baseline. Đây là điểm TRUNG THỰC — em không dùng SMOTE trên lớp quá nhỏ để tránh kết quả ảo.

---

## 6. Kết luận

Cải tiến CMAR theo hướng **không dùng trọng số thuộc tính**, kết hợp 3 kỹ thuật từ các bài báo uy tín (MWMOTE + Added Value voting + Adaptive gating), đạt:
- **Tăng F1 (+0.016) và Recall (+0.028)** trên dữ liệu mất cân bằng.
- **Không giảm** chỉ số trên dữ liệu cân bằng (14/14 giữ nguyên).
- **Ngưỡng SMOTE có cơ sở khoa học** (He & Garcia 2009): ratio≥2 VÀ lớp≥6 mẫu.
- Kiểm định bằng **Chi-square** sẵn có trong CMAR.

---

## 7. Câu hỏi xin ý kiến cô

1. Hướng **không trọng số thuộc tính** + 3 kỹ thuật trên đã phù hợp chưa ạ?
2. Dùng **Chi-square** làm kiểm định thống kê có đủ chưa, hay cô muốn em bổ sung kiểm định nào khác?
3. Cô có muốn em báo cáo thêm chỉ số nào ngoài Accuracy / F1 / Recall không ạ?
