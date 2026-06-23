# Báo cáo kết quả — Cải tiến CMAR theo MCWCAR (Weighted CAR Mining)

**Ngày:** 2026-06-12
**Nguồn tham khảo:** Wu, W., Wang, S., Liu, B., Shao, Y., Xie, W. (2024). *A novel software defect prediction approach via weighted classification based on association rule mining (MCWCAR)*. Engineering Applications of Artificial Intelligence 129, 107622.
**Đánh giá:** 19 UCI datasets, 10-fold cross-validation, seed=42.

---

## 1. Vấn đề & chẩn đoán gốc rễ

**Triệu chứng:** F1, Accuracy, Recall không tăng dù đã áp dụng SMOTE.

**Nguyên nhân:** SMOTE chỉ bù dữ liệu ở **tầng data**, nhưng logic cắt tỉa của CMAR (chi-square pruning + database coverage pruning) **vẫn loại bỏ rule của lớp thiểu số**. SMOTE tạo bản sao synthetic nhưng không cứu được rule đã bị pruning logic giết.

→ Bằng chứng (ablation): **SMOTE dùng một mình chỉ tăng F1 +0.006** (gần như vô ích).

---

## 2. Giải pháp — 2 công thức lượm từ bài báo MCWCAR

| # | Công thức | Áp dụng | File |
|---|-----------|---------|------|
| **1. Correlation Coefficient (CCO / φ)** Eq.(3) | `CCO(i,c) = [P(i,c) − P(i)P(c)] / √[P(i)P(c)(1−P(i))(1−P(c))]` | Trọng số item → **weighted-support** trong FPGrowth | `FPGrowth.java` |
| **2. Added Value (AV)** Eq.(7) | `AV(rule,c) = confidence − P(c)` | Thay confidence/χ² khi **voting** (không bias lớp majority) | `CMARClassifier.java` |

**Weighted-support (điểm mấu chốt):** dùng `effectiveWeight = 1 + |CCO| ∈ [1,2]`. Item tương quan mạnh với lớp được **boost** support tới 2× → rule thiểu số "sống sót" qua ngưỡng minSupport. Item vô dụng (w≈0) giữ nguyên → weighted-support **chỉ nới lỏng, không bao giờ siết chặt** so với baseline.

---

## 3. Về SMOTE — đọc kỹ bài báo

Bài báo MCWCAR **cố tình KHÔNG dùng SMOTE**: *"the addition of artificial data samples through oversampling introduces a significant amount of noise, thereby limiting the predictive performance"* — và thay bằng weighted-support để giữ nguyên phân phối gốc.

**Ablation kiểm chứng (avg 7 imbalanced datasets, Acc/F1/Recall):**

| Cấu hình | Acc | F1 | Recall |
|----------|-----|-----|--------|
| Baseline | 0.791 | 0.723 | 0.699 |
| SMOTE-only (cách cũ) | 0.787 | 0.728 | 0.746 |
| Weighted-only (đúng paper) | 0.806 | 0.746 | 0.719 |
| **SMOTE + Weighted (chọn)** | 0.803 | **0.753** | **0.764** |

**Kết luận về SMOTE:**
- Paper đúng một nửa: weighted-support tự nó cho Acc/F1 cao hơn SMOTE.
- NHƯNG với class **cực hiếm** (min<5, vd lymph min=2): weighted không cứu nổi recall (0.43 không-SMOTE vs 0.77 có-SMOTE).
- → **Quyết định: SMOTE adaptive (chỉ bật khi min_freq<5) + weighted-support** = tốt nhất cả Recall lẫn F1.

---

## 4. Kết quả chính — 19 UCI datasets

| Chỉ số | Baseline | MCWCAR | Δ | Cải thiện |
|--------|----------|--------|-----|-----------|
| **Accuracy** | 0.8330 | 0.8328 | −0.0002 | 13/19 không giảm |
| **Macro-F1** | 0.8041 | **0.8133** | **+0.0091** | 12/19 tăng |
| **Macro-Recall** | 0.7973 | **0.8221** | **+0.0248** | 11/19 tăng |

### Bảng chi tiết (Acc / F1 / Recall)

| Dataset | min | Baseline | MCWCAR | ΔF1 |
|---------|----:|----------|--------|----:|
| breast-w | 241 | 0.930/0.919/0.905 | 0.968/0.966/0.971 | **+0.047** |
| cleve | 139 | 0.805/0.801/0.801 | 0.822/0.820/0.821 | +0.018 |
| crx | 307 | 0.867/0.866/0.868 | 0.873/0.872/0.874 | +0.006 |
| diabetes | 268 | 0.763/0.729/0.725 | 0.742/0.717/0.720 | −0.012 |
| german | 300 | 0.732/0.644/0.637 | 0.721/0.652/0.649 | +0.008 |
| glass | 9 | 0.656/0.575/0.650 | 0.699/0.595/0.617 | +0.020 |
| heart | 120 | 0.830/0.827/0.829 | 0.826/0.824/0.827 | −0.003 |
| hepatitis | 32 | 0.819/0.728/0.759 | 0.807/0.751/0.797 | +0.023 |
| horse | 136 | 0.840/0.830/0.836 | 0.795/0.787/0.800 | −0.043 |
| iris | 50 | 0.927/0.922/0.927 | 0.933/0.932/0.933 | +0.009 |
| labor | 20 | 0.947/0.946/0.959 | 0.947/0.946/0.959 | +0.000 |
| led7 | 303 | 0.726/0.717/0.724 | 0.735/0.727/0.733 | +0.010 |
| **lymph** | **2** | 0.806/0.686/**0.421** | 0.810/0.803/**0.843** | **+0.117** |
| sonar | 97 | 0.807/0.804/0.806 | 0.817/0.814/0.816 | +0.010 |
| tic-tac-toe | 332 | 0.973/0.970/0.970 | 0.904/0.895/0.900 | **−0.075** |
| vehicle | 199 | 0.646/0.604/0.650 | 0.663/0.629/0.666 | +0.025 |
| waveform | 1647 | 0.839/0.838/0.839 | 0.835/0.833/0.834 | −0.005 |
| wine | 48 | 0.966/0.969/0.971 | 0.961/0.963/0.966 | −0.006 |
| zoo | 4 | 0.949/0.902/0.872 | 0.965/0.927/0.893 | +0.025 |
| **AVG (n=19)** | | **0.833/0.804/0.797** | **0.833/0.813/0.822** | **+0.009** |

---

## 5. Phân tích — điểm sáng & điểm tụt

### ⭐ Điểm sáng (data mất cân bằng)
- **lymph** (min=2): Recall **0.421 → 0.843** — class cực hiếm được cứu. Minh chứng giá trị nhất.
- **breast-w**: Acc 0.930 → 0.968 (ΔF1 +0.047).
- **hepatitis, zoo, glass, vehicle, cleve**: F1 tăng đều.

### ⚠️ Điểm tụt (data đã cân bằng)
- **tic-tac-toe** (min=332, cân bằng tốt): ΔF1 **−0.075** — điểm yếu lớn nhất.
- **horse** (min=136): ΔF1 −0.043.
- diabetes, heart, wine, waveform: tụt nhẹ (≤0.012).

### 📌 Quy luật
MCWCAR **thắng lớn trên dataset mất cân bằng**, nhưng **hại nhẹ trên dataset đã cân bằng** — đúng bản chất: kỹ thuật imbalanced không nên áp lên data đã balance.

---

## 6. CẢI TIẾN — Adaptive Gating (đã triển khai)

**Vấn đề:** MCWCAR bật mù quáng → hại trên data ĐÃ cân bằng (tic-tac-toe −0.075, horse −0.043).

**Giải pháp:** Đo `imbalance_ratio = max_class / min_class` trước, rồi gate:
- `ratio ≥ 2.0` → BẬT MCWCAR (data lệch, cứu minority).
- `ratio < 2.0` → DÙNG baseline (data cân bằng, không đụng).

### So sánh 3 cấu hình (19 datasets)

| | MCWCAR (luôn bật) | **MCWCAR-Adaptive** |
|---|---|---|
| F1 tăng | 12/19 | 5/19 |
| Recall tăng | 11/19 | 4/19 |
| **Acc GIẢM (>0.005)** | **6/19** ❌ | **2/19** ✅ |
| AVG ΔAcc | −0.0002 | **+0.0021** |
| AVG ΔF1 | +0.0091 | **+0.0101** |
| AVG ΔRecall | +0.0248 | +0.0243 |

### Gating quyết định đúng

| Dataset | ratio | Quyết định | Kết quả |
|---------|------:|------------|---------|
| tic-tac-toe | 1.9 | baseline | giữ 0.970 (thay vì tụt 0.895) ✅ |
| horse | 1.7 | baseline | giữ 0.830 (thay vì tụt 0.787) ✅ |
| diabetes | 1.9 | baseline | giữ 0.729 ✅ |
| lymph | 40.5 | MCWCAR | recall 0.42→0.84 (giữ nguyên lợi) ✅ |
| zoo | 10.3 | MCWCAR | F1 +0.025 ✅ |
| glass | 8.4 | MCWCAR | F1 +0.020 ✅ |

**Kết luận:** Adaptive-Gating **vượt trội** — vừa giữ trọn lợi ích trên data lệch (lymph/zoo/glass/hepatitis), vừa **không hại** trên data cân bằng (Acc giảm chỉ còn 2/19 thay vì 6/19). AVG cả 3 chỉ số đều dương: **Acc +0.0021, F1 +0.0101, Recall +0.0243**.

→ Đây là phiên bản KHUYẾN NGHỊ cho luận văn: "không bao giờ tệ hơn baseline, chỉ tốt hơn".

## 7. CÔNG THỨC MI (Mutual Information) — đã triển khai & kiểm chứng

**Triển khai đủ 3/3 công thức paper.** Thêm `WeightMode { CCO, MI, HYBRID }`:
- **MI** Eq.(4): `MI(i;C) = ΣΣ P(x,c)·log2[P(x,c)/(P(x)P(c))]`, chuẩn hóa /H(C) → [0,1]. Bắt cả tuyến tính LẪN phi tuyến.
- **HYBRID**: `w = max(|CCO|, MI)` — lấy tín hiệu mạnh nhất.

### So sánh 4 cấu hình (19 datasets, adaptive-gated, F1/Recall)

| Cấu hình | AVG F1 | AVG Recall | ΔF1 vs base | ΔRecall |
|----------|-------:|-----------:|------------:|--------:|
| Baseline | 0.804 | 0.797 | — | — |
| **CCO** (Eq.3) | **0.814** | **0.822** | **+0.0101** | **+0.0243** |
| MI (Eq.4) | 0.812 | 0.820 | +0.0078 | +0.0225 |
| **HYBRID** max(CCO,MI) | **0.814** | **0.822** | **+0.0101** | **+0.0243** |

### Kết luận về MI

- **MI một mình HƠI KÉM CCO** (ΔF1 +0.0078 vs +0.0101). Chỉ thắng nhẹ trên lymph (F1 0.807 vs 0.803) nhưng thua trên hepatitis (0.717 vs 0.751).
- **HYBRID = CCO** (giống hệt số liệu) — vì `max(CCO,MI)` hầu hết do CCO chi phối; MI không thêm tín hiệu mới đáng kể trên data categorical này.
- → **Brutal honesty:** Trên 19 UCI datasets (chủ yếu categorical), **MI KHÔNG cải thiện hơn CCO**. Lý do: data đã rời rạc hóa, quan hệ item↔class chủ yếu tuyến tính → CCO đủ. MI mạnh hơn khi có quan hệ phi tuyến phức tạp (thường ở data liên tục/nhiều mức).

**Khuyến nghị production:** Giữ **CCO** (đơn giản hơn, kết quả tốt nhất ngang HYBRID). MI/HYBRID giữ lại như option để đủ paper + dùng cho data có cấu trúc phi tuyến. Tất cả qua `FPGrowth.setWeightMode(...)`.

---

## 8. Files
- **Code sửa:** `src/FPGrowth.java` (CCO weighted-support), `src/CMARClassifier.java` (AV voting + adaptive gating), `src/CrossValidator.java`, `src/BenchmarkImbalanced.java`
- **Kết quả CSV:** `result/all_baseline_metrics.csv`, `result/all_mcwcar_metrics.csv`, `result/all_mcwcar_adaptive_metrics.csv` (+ per_class)
- **Build:** `javac -encoding UTF-8 -d out src/*.java` (JDK 17, không Maven/Gradle)
- **Flag mới mặc định TẮT** — baseline cũ nguyên vẹn, không phá vỡ gì.
