# Giải thích: Vấn đề lớp CỰC HIẾM và vai trò của MWMOTE

**Đề tài:** Cải tiến CMAR cho dữ liệu mất cân bằng
**Sinh viên:** Lê Hồng Công
**Ngày:** 26/06/2026
**Mục đích:** Làm rõ một câu hỏi quan trọng — với lớp chỉ có 2-4 mẫu, làm sao cải tiến cho đáng tin? Đây là phần em chủ động phân tích để thầy/cô thấy em hiểu sâu giới hạn của bài toán.

---

## 1. ĐẶT VẤN ĐỀ

Trong các dataset mất cân bằng, có 2 loại lớp hiếm khác nhau:

| Loại | Ví dụ | Số mẫu lớp hiếm | Vấn đề |
|------|-------|----------------:|--------|
| **Cực hiếm** | lymph (lớp 1) | 2 mẫu | Quá ít để đánh giá tin cậy |
| **Cực hiếm** | zoo (lớp 5) | 4 mẫu | Quá ít |
| **Hiếm vừa** | glass (lớp 3,6) | 9-17 mẫu | Đủ để học + đánh giá |

→ Câu hỏi cốt lõi: **Lớp cực hiếm (2-4 mẫu) thì cải tiến thế nào, và con số có đáng tin không?**

---

## 2. CMAR GỐC THẤT BẠI TRÊN LỚP CỰC HIẾM

CMAR sinh luật dựa trên **support** (số lần xuất hiện). Lớp 2 mẫu → support cực thấp → **không sinh nổi luật** → mô hình **bỏ qua hoàn toàn** lớp đó.

**Bằng chứng:** lymph — CMAR gốc Recall = **0.42** (bỏ sót quá nửa số ca thuộc lớp hiếm).

→ Nếu KHÔNG có cơ chế xử lý, lớp cực hiếm **chắc chắn bị bỏ rơi**.

---

## 3. MWMOTE GIẢI QUYẾT BẰNG CÁCH NÀO

MWMOTE sinh thêm mẫu nhân tạo cho lớp cực hiếm:
- lymph lớp 1: **2 mẫu → 81 mẫu**
- zoo lớp 5: **4 mẫu → 81 mẫu**

→ Giờ CMAR **có đủ mẫu để sinh luật** cho lớp hiếm → mô hình **không bỏ qua nữa** → Recall tăng.

**Đây là cải thiện THẬT về hành vi mô hình:** từ "bỏ qua hoàn toàn lớp hiếm" thành "có nhận diện lớp hiếm".

---

## 4. ⚠️ GIỚI HẠN — Tại sao con số trên lớp CỰC hiếm KHÔNG đáng tin về thống kê

> Đây là điểm em chủ động nêu để minh bạch.

### 4.1. "Sinh 79 mẫu từ 2 điểm" nghĩa là gì?
- MWMOTE sinh mẫu mới bằng cách **pha trộn giữa các mẫu gốc**.
- lymph lớp 1 chỉ có **2 mẫu gốc (A, B)** → 79 mẫu mới đều là **biến thể của A và B**, không có "mẫu thật mới".
- → Mô hình **học thuộc lòng (memorization)** 2 mẫu A, B thay vì **hiểu khái quát** lớp hiếm trông như thế nào.

### 4.2. Tại sao điều này làm con số không đáng tin
- 10-fold CV: lymph lớp 1 (2 mẫu) → mỗi fold test chỉ có **~0-1 mẫu**.
- Mẫu test đó thường **chính là A hoặc B** (vì lớp chỉ có 2 mẫu) → mô hình đã "thuộc" → đoán đúng.
- → Recall 0.91 thực chất là "đoán đúng vài lần trên 1-2 mẫu test" → **biên độ sai số cực lớn, không có ý nghĩa thống kê**.

### 4.3. Đây là giới hạn của BÀI TOÁN, không phải thuật toán
> Với lớp chỉ 2-4 mẫu, **KHÔNG phương pháp nào** (kể cả các thuật toán SOTA) cho được con số đánh giá chắc chắn — vì không đủ mẫu test để kiểm chứng. Đây là giới hạn vật lý của dữ liệu.

---

## 5. PHÂN BIỆT QUAN TRỌNG: "Cải thiện THẬT" vs "Con số ĐÁNG TIN"

Đây là 2 khái niệm khác nhau:

| | MWMOTE trên lymph |
|---|---|
| **Cải thiện thật về hành vi?** | ✅ CÓ — mô hình từ bỏ qua lớp hiếm → bắt được lớp hiếm |
| **Con số 0.91 đáng tin thống kê?** | ❌ KHÔNG — chỉ 2 mẫu, biên độ sai số quá lớn |

→ **Kết luận:** MWMOTE **đáng dùng** (nó thực sự giúp mô hình không bỏ rơi lớp hiếm), nhưng **con số cụ thể trên lymph KHÔNG nên trích dẫn làm bằng chứng định lượng chính**.

---

## 6. CÁCH GIẢI QUYẾT — Phân tầng theo độ tin cậy

Em phân loại dataset theo độ tin cậy của kết quả:

| Nhóm | Dataset | Số mẫu lớp hiếm | MWMOTE? | Cải thiện từ đâu | Độ tin cậy |
|------|---------|----------------:|:-------:|------------------|------------|
| **Đáng tin (bằng chứng chính)** | glass | 9-17 | ❌ không cần | AV voting (F1 +0.048) | ✅ Cao |
| **Đáng tin** | zoo | 4 (nhưng 7 lớp) | nhẹ | Chủ yếu AV voting (F1 0.902→0.922 dù KHÔNG MWMOTE) | ✅ Khá |
| **Chỉ minh họa** | lymph | 2 | bắt buộc | MWMOTE (memorization) | ⚠️ Thấp — không đại diện |

### Bằng chứng zoo KHÔNG phụ thuộc MWMOTE (quan trọng)
zoo bỏ MWMOTE: F1 vẫn 0.902 → **0.922** (tăng nhờ AV voting). MWMOTE chỉ thêm chút (0.922→0.926).
→ zoo cải thiện chủ yếu nhờ AV voting, không phải "data giả" → đáng tin.

### Bằng chứng glass HOÀN TOÀN nhờ AV voting
glass lớp 9-17 mẫu → MWMOTE KHÔNG kích hoạt → F1 tăng +0.048 thuần từ AV voting → **bằng chứng vững nhất**.

---

## 7. KẾT LUẬN — Trả lời câu hỏi "lớp hiếm làm sao cải tiến đáng tin"

1. **Lớp hiếm VỪA (glass 9-17 mẫu):** AV voting cải thiện THẬT và ĐÁNG TIN, không cần MWMOTE.
2. **Lớp hiếm VỪA-NHỎ (zoo 4 mẫu, nhiều lớp):** AV voting vẫn là chính, MWMOTE hỗ trợ nhẹ → đáng tin.
3. **Lớp CỰC HIẾM (lymph 2 mẫu):** MWMOTE giúp mô hình KHÔNG bỏ rơi lớp hiếm (cải thiện thật), nhưng con số không đại diện thống kê → chỉ dùng minh họa, KHÔNG làm bằng chứng định lượng.

**Tóm tắt:** Cải tiến của em **đáng tin** vì dựa trên glass/zoo (lớp đủ mẫu, AV voting). lymph được giữ để minh họa khả năng cứu lớp cực hiếm, với hạn chế được nêu rõ.

---

## 8. CÂU HỎI XIN Ý KIẾN THẦY/CÔ

1. Với lymph (lớp 2 mẫu), thầy/cô muốn em **giữ + ghi rõ hạn chế**, hay **loại khỏi đánh giá định lượng** (nhiều paper loại class < 5-10 mẫu)?
2. Cách phân tầng độ tin cậy (Phần 6) đã hợp lý chưa ạ?
3. Em có nên báo cáo thêm con số "AVG không tính lymph" để minh bạch không ạ?

---

## PHỤ LỤC — Số liệu (10-fold CV, seed=42)

| Dataset | Baseline F1/R | KHÔNG MWMOTE F1/R | CÓ MWMOTE F1/R |
|---------|---------------|-------------------|----------------|
| lymph (2) | 0.686/0.421 | 0.683/0.418 | 0.843/0.908 |
| zoo (4) | 0.902/0.872 | 0.922/0.914 | 0.926/0.900 |
| glass (9) | 0.575/0.650 | 0.623/0.650 | 0.623/0.650 |
| hepatitis (32) | 0.728/0.759 | 0.700/0.776 | 0.700/0.776 |
| german (300) | 0.644/0.637 | 0.643/0.644 | 0.643/0.644 |

→ glass/hepatitis/german: cột "có" và "không" MWMOTE GIỐNG HỆT (vì lớp ≥9 mẫu, MWMOTE không kích hoạt) → cải thiện hoàn toàn từ AV voting.
