# Verify SMOTE — Phân Tích Đầy Đủ Code CMAR

> **Mục đích**: Verify toàn bộ implementation SMOTE trong project. Trả lời 4 câu hỏi cốt lõi: **KHI NÀO** bật SMOTE, **VÌ SAO** dùng, **HOW** sinh synthetic, và **K/N parameter** thực chất là gì.

---

## Mục Lục

1. [TL;DR — Tóm tắt](#tldr)
2. [Khi nào bật SMOTE / Khi nào không](#1-khi-nao)
3. [Vì sao phải dùng SMOTE](#2-vi-sao)
4. [How — Synthetic sinh ra như nào](#3-how)
5. [K và N parameter](#4-k-n)
6. [Vấn đề lớn của SMOTE](#5-van-de)
7. [Bảng verification tổng kết](#6-verify)
8. [So sánh với paper](#7-paper)

---

<a name="tldr"></a>
## 1. TL;DR — Tóm tắt 30 giây

| Câu hỏi | Trả lời |
|---|---|
| Code đang dùng SMOTE-N hay SMOTE numeric? | **SMOTE-N** (Nominal/Categorical) theo Chawla 2002 §6.2 |
| Có dùng K? | ✓ CÓ, `k = 5` |
| Có dùng N? | ✗ KHÔNG dùng N% — thay bằng `targetRatio` (cùng ý tưởng, parameter hóa khác) |
| Distance dùng gì? | **Hamming** (đếm attribute khác value) — KHÔNG phải Euclidean 2D |
| Synthesis dùng gì? | **Mode voting** per-attribute — KHÔNG phải interpolation `A + λ(B-A)` |
| Trigger logic? | **Adaptive**: chỉ bật khi `min_class_freq < TRIGGER` (5 hoặc 10) |
| Có Borderline variant? | ✓ CÓ — Han 2005, phân loại SAFE/DANGER/NOISE |
| Có data leakage? | ✗ KHÔNG — SMOTE chỉ áp dụng trên `trainData` bên trong CV loop |

---

<a name="1-khi-nao"></a>
## 2. KHI NÀO bật SMOTE / KHI NÀO không?

### 2.1 Adaptive Trigger — Logic cốt lõi

Code KHÔNG bật SMOTE mù trên mọi dataset. Chỉ bật khi **minority class quá ít records**.

| Phiên bản | File | Trigger | Hành vi |
|---|---|---|---|
| v11 (Full UCI) | [BenchmarkSMOTEFull.java:49](../src/BenchmarkSMOTEFull.java#L49) | `min_freq < 10` | Vanilla SMOTE-N |
| v12 (Borderline) | [BenchmarkBorderline.java:43](../src/BenchmarkBorderline.java#L43) | `min_freq < 10` | So sánh 2 variants |
| **v13 (Adaptive)** | [BenchmarkImbalanced.java:40](../src/BenchmarkImbalanced.java#L40) | **`min_freq < 5`** | Borderline-SMOTE chỉ khi cực ít |

### 2.2 Logic Adaptive (v13)

```java
// File: BenchmarkImbalanced.java:177-184
double smoteRatio = 0.0;       // mặc định TẮT
boolean useBorderline = false;
if (useAdaptiveSmote) {
    int minFreq = computeMinClassFreq(data);
    if (minFreq < SMOTE_TRIGGER) {     // < 5
        smoteRatio = SMOTE_RATIO;       // 1.0 → fully balanced
        useBorderline = true;
    }
    // else: KHÔNG SMOTE — H2 alone đã đủ
}
```

### 2.3 Áp dụng trong Cross-Validation Loop

```java
// File: CrossValidator.java:119-132
if (smoteTargetRatio > 0) {
    if (useBorderlineSMOTE) {
        trainData = BorderlineSMOTE.apply(trainData, 5, smoteTargetRatio, seed+fold);
    } else {
        trainData = SMOTE.apply(trainData, 5, smoteTargetRatio, seed+fold);
    }
    // ⚠️ CHỈ áp dụng trên trainData — testData GIỮ NGUYÊN (tránh data leakage)
}
```

### 2.4 Ví dụ cụ thể — 7 imbalanced datasets v13

| Dataset | min_freq | SMOTE? | Strategy |
|---|---|---|---|
| Lymph | **2** | ✓ ON | Borderline-SMOTE |
| Zoo | **4** | ✓ ON | Borderline-SMOTE |
| Glass | 9 | ✗ off | H2 only |
| Hepatitis | ~32 | ✗ off | H2 only |
| German | ~300 | ✗ off | H2 only |
| Vehicle | ~199 | ✗ off | H2 only |
| Breast-w | ~241 | ✗ off | H2 only |

**Quan sát**: Chỉ 2/7 datasets thực sự cần SMOTE. H2 (class-specific minSup) đã xử lý đủ tốt 5 datasets còn lại.

---

<a name="2-vi-sao"></a>
## 3. VÌ SAO phải dùng SMOTE?

### 3.1 Vấn đề gốc trong CMAR Pipeline

```
trainData → FP-Growth (minSup=X%) → frequent patterns → CARs → classifier
```

Bottleneck: **support threshold loại bỏ minority patterns**.

### 3.2 Ví dụ Lymph dataset

```
Lymph: 148 records, 2 minority records
minSup = 5% = 7.4 → cần ≥ 7 records cho 1 pattern
→ minority chỉ có 2 records → KHÔNG BAO GIỜ tạo được CAR
→ Classifier không có rule predict minority
→ Default class fallback → Recall = 0, F1 = 0
```

### 3.3 SMOTE giải quyết

```
Trước SMOTE: minority 2 records → support = 2/148 = 1.35% < 5%  ✗
Sau SMOTE (targetRatio=1.0, majority=146):
   minority sinh thêm 144 synthetic → tổng 146 records
   → support = 146/292 = 50% >> 5%  ✓
   → CAR cho minority được sinh ra
   → Classifier có rule predict minority
   → F1 tăng từ 0.42 → 0.74
```

**Xác nhận thực nghiệm**: [report/v11_smote_full_uci.md](v11_smote_full_uci.md) — Lymph F1 cải thiện từ 0.42 → 0.74.

---

<a name="3-how"></a>
## 4. HOW — Synthetic sinh ra như nào?

### 4.1 Phân biệt: SMOTE GỐC vs SMOTE-N

**⚠️ Đây là chỗ DỄ NHẦM nhất khi thuyết trình.**

#### SMOTE GỐC (Chawla 2002 §6.1) — cho dữ liệu **số học**

```
Cho minority point A = (x1, x2) trong không gian 2D
Bước 1: Tìm K nearest neighbors bằng Euclidean distance
Bước 2: Chọn ngẫu nhiên 1 neighbor B = (y1, y2)
Bước 3: Sinh synthetic = A + λ·(B - A), với λ ∈ [0, 1]
```

→ Synthetic nằm **trên đoạn thẳng AB**.  
→ Đây là hình minh họa cổ điển trong sách giáo trình (đường nét đứt đỏ).

#### SMOTE-N (Chawla 2002 §6.2) — cho dữ liệu **categorical** ← code của bạn

```
Cho minority record A = {color=red, shape=square, size=big}
Bước 1: Tìm K nearest neighbors bằng Hamming distance
Bước 2: Lấy TẤT CẢ K neighbors + A
Bước 3: Với mỗi attribute, vote mode (giá trị xuất hiện nhiều nhất)
Bước 4: Tạo synthetic = các mode values
```

→ KHÔNG có không gian 2D.  
→ KHÔNG có interpolation `λ`.  
→ Synthetic là một record categorical mới.

### 4.2 Chi tiết từng bước trong code

#### Bước 1 — Hamming distance ([SMOTE.java:129-146](../src/SMOTE.java#L129-L146))

```java
private static int hammingDistance(Set<String> baseItems, List<String> other) {
    Map<String, String> baseMap = itemsToMap(baseItems);
    Map<String, String> otherMap = itemsToMap(other);
    Set<String> allAttrs = new HashSet<>(baseMap.keySet());
    allAttrs.addAll(otherMap.keySet());
    int diff = 0;
    for (String attr : allAttrs) {
        String bv = baseMap.get(attr);
        String ov = otherMap.get(attr);
        if (bv == null || ov == null || !bv.equals(ov)) {
            diff++;
        }
    }
    return diff;
}
```

**Ví dụ**:
```
A = {color=red,  shape=square, size=big}
B = {color=red,  shape=circle, size=big}
C = {color=blue, shape=square, size=small}

d(A, B) = 1  (chỉ khác shape)
d(A, C) = 2  (khác color + size)
→ B gần A hơn
```

#### Bước 2 — K-Nearest Neighbors ([SMOTE.java:104-123](../src/SMOTE.java#L104-L123))

```java
private static List<Transaction> kNearestNeighbors(
        Transaction base, List<Transaction> pool, int k) {
    List<Map.Entry<Transaction, Integer>> distances = new ArrayList<>();
    for (Transaction t : pool) {
        if (t == base) continue;
        int d = hammingDistance(...);
        distances.add(...);
    }
    distances.sort(Comparator.comparingInt(Map.Entry::getValue));
    // Lấy K records có distance nhỏ nhất
    return distances.subList(0, min(k, distances.size()));
}
```

#### Bước 3 — Mode Voting ([SMOTE.java:165-202](../src/SMOTE.java#L165-L202))

```java
private static Transaction createSynthetic(Transaction base,
                                            List<Transaction> neighbors,
                                            String classLabel,
                                            Random rng) {
    // Đếm count cho mỗi attr=value trong base + neighbors
    Map<String, Map<String, Integer>> attrValueCounts = new HashMap<>();
    addItemsToCount(base.getItems(), attrValueCounts);
    for (Transaction n : neighbors) {
        addItemsToCount(n.getItems(), attrValueCounts);
    }

    // Pick mode cho mỗi attribute
    List<String> syntheticItems = new ArrayList<>();
    for (Map.Entry<String, Map<String, Integer>> e : attrValueCounts.entrySet()) {
        String attr = e.getKey();
        int maxCount = max values counts;
        // Tie-breaking: random pick
        String chosen = modes.get(rng.nextInt(modes.size()));
        syntheticItems.add(attr + "=" + chosen);
    }
    return new Transaction(syntheticItems, classLabel);
}
```

**Ví dụ end-to-end** (base=A, k=2 neighbors=B,C):

| Attribute | Values trong {A, B, C} | Count | Mode | Chọn |
|---|---|---|---|---|
| color | red, red, blue | red:2, blue:1 | red | `color=red` |
| shape | square, circle, square | square:2, circle:1 | square | `shape=square` |
| size | big, big, small | big:2, small:1 | big | `size=big` |

→ Synthetic = `{color=red, shape=square, size=big}` (trùng A vì A là mode trong 3 records này — thực tế nhiều neighbor hơn → khác).

### 4.3 Sơ đồ trực quan so sánh

```
═══════════════════════════════════════════════════════════════════
SMOTE NUMERIC (Chawla 2002 §6.1)        SMOTE-N của bạn (§6.2)
═══════════════════════════════════════════════════════════════════
Dataset 2D:                              Dataset categorical:

  • • • • majority                       A {red,  sq, big}
   • •                                   B {red,  ci, big}    ← k-NN
  A ─λ─ B  minority interpolate          C {blue, sq, sma}    ← k-NN
   •                                     ─────────────
                                         Mode vote per attr:
synthetic = A + λ(B-A)                     color: red(2) > blue(1)
         nằm trên đoạn AB                  shape: sq(2)  > ci(1)
                                           size:  big(2) > sma(1)
                                         ─────────────
                                         synth {red, sq, big}
═══════════════════════════════════════════════════════════════════
```

---

<a name="4-k-n"></a>
## 5. K và N — Parameter Thực Chất

### 5.1 Bảng đối chiếu

| Param | Ý nghĩa | Paper Chawla 2002 | Code của bạn | Vị trí |
|---|---|---|---|---|
| **K** | Số nearest neighbors | k = 5 | `k = 5` (default) | [SMOTE.java:41](../src/SMOTE.java#L41), [:96](../src/SMOTE.java#L96) |
| **N** | Lượng oversample | N% (vd. 200% = ×2) | thay bằng `targetRatio` | [SMOTE.java:42](../src/SMOTE.java#L42) |

### 5.2 N% (Chawla) ↔ targetRatio — Tương đương

**Paper Chawla 2002**:
```
N% = 300%, minority có 10 records
→ Sinh 3 × 10 = 30 synthetic
→ Tổng minority sau = 10 + 30 = 40
```

**Code của bạn** ([SMOTE.java:57](../src/SMOTE.java#L57)):
```java
int target = (int) Math.round(maxFreq * targetRatio);
```

```
targetRatio = 1.0, minority = 10, majority = 100
→ target = 100 × 1.0 = 100
→ Sinh 100 - 10 = 90 synthetic
→ Tổng minority = 100 (= majority → balanced)
```

### 5.3 Vì sao đổi từ N% sang targetRatio?

| Cách | Ưu | Nhược |
|---|---|---|
| N% (paper) | Theo chuẩn paper | Phải tính lại N cho mỗi dataset để đạt balance |
| targetRatio (code) | `1.0` → balanced tự động cho mọi dataset | Lệch nomenclature với paper |

→ **Adaptive hơn cho benchmark đa dataset**. Khi report cho thầy cô, nên giải thích rõ.

---

<a name="5-van-de"></a>
## 6. VẤN ĐỀ LỚN của SMOTE (Quan trọng cho defense)

### ⚠️ Vấn đề 1 — Synthetic overlap với majority class

**Mô tả**: SMOTE gốc interpolate mù → synthetic có thể nằm **trong vùng majority** → classifier confused → Accuracy giảm.

**Giải pháp trong code**: **Borderline-SMOTE (Han 2005)** ở [BorderlineSMOTE.java:82-106](../src/BorderlineSMOTE.java#L82-L106).

Phân loại minority records thành 3 nhóm:
```
SAFE   : majorityCount < kEff/2     → an toàn, không cần oversample
DANGER : kEff/2 ≤ majorityCount < kEff → gần biên, OVERSAMPLE
NOISE  : majorityCount == kEff       → cô lập, có thể là noise → skip
```

→ Chỉ oversample DANGER set → tránh overlap.

### ⚠️ Vấn đề 2 — Amplify noise

**Mô tả**: Nếu minority có outlier → SMOTE sinh thêm synthetic CỦA outlier → noise lan rộng.

**Giải pháp**: NOISE filter trong Borderline ([BorderlineSMOTE.java:100-101](../src/BorderlineSMOTE.java#L100-L101)).

### ⚠️ Vấn đề 3 — Mode voting tạo synthetic trùng base

**Mô tả**: Với SMOTE-N, nếu đa số neighbors giống base → synthetic = base → duplicate, không tăng diversity.

**Giải pháp**: Tie-breaking random pick ở [SMOTE.java:197](../src/SMOTE.java#L197).

**Đây là giới hạn cố hữu của SMOTE-N** — phải nêu thẳng khi defense.

### ⚠️ Vấn đề 4 — Data leakage nếu sai pipeline

**Mô tả**: Nếu SMOTE chạy TRƯỚC train/test split → synthetic dựa trên test → cheat → metric ảo cao.

**Code của bạn ĐÚNG**: SMOTE chỉ áp dụng trên `trainData` BÊN TRONG CV loop ở [CrossValidator.java:119](../src/CrossValidator.java#L119):
```java
for (int fold = 0; fold < k; fold++) {
    List<Transaction> testData = folds[fold];          // GIỮ NGUYÊN
    List<Transaction> trainData = new ArrayList<>();
    for (int j = 0; j < k; j++) if (j != fold) trainData.addAll(folds[j]);

    if (smoteTargetRatio > 0) {
        trainData = SMOTE.apply(trainData, ...);       // CHỈ train
    }
    // ... train + evaluate
}
```

✓ **Không data leakage**.

---

<a name="6-verify"></a>
## 7. Bảng Verification Tổng Kết

| # | Khía cạnh | Trong paper | Trong code | Verify |
|---|---|---|---|---|
| 1 | K = 5 default | ✓ | ✓ ([SMOTE.java:96](../src/SMOTE.java#L96)) | ✅ ĐÚNG |
| 2 | N% (oversample %) | ✓ | ⚠️ `targetRatio` (tương đương) | ⚠️ Khác parameter hóa |
| 3 | Hamming distance (categorical) | ✓ Chawla §6.2 | ✓ ([SMOTE.java:129](../src/SMOTE.java#L129)) | ✅ ĐÚNG |
| 4 | Mode voting (categorical) | ✓ Chawla §6.2 | ✓ ([SMOTE.java:165](../src/SMOTE.java#L165)) | ✅ ĐÚNG |
| 5 | Borderline SAFE/DANGER/NOISE | ✓ Han 2005 | ✓ ([BorderlineSMOTE.java:82](../src/BorderlineSMOTE.java#L82)) | ✅ ĐÚNG |
| 6 | k-NN trong CÙNG class cho synthesis | ✓ | ✓ ([SMOTE.java:104](../src/SMOTE.java#L104)) | ✅ ĐÚNG |
| 7 | k-NN TOÀN BỘ data cho DANGER detection | ✓ Han 2005 | ✓ ([BorderlineSMOTE.java:88](../src/BorderlineSMOTE.java#L88)) | ✅ ĐÚNG |
| 8 | Áp dụng SMOTE chỉ trên trainData | (best practice) | ✓ ([CrossValidator.java:119](../src/CrossValidator.java#L119)) | ✅ ĐÚNG |
| 9 | Edge case: class < 2 records | (paper không đề cập) | ✓ Duplicate fallback ([SMOTE.java:71](../src/SMOTE.java#L71)) | ✅ ĐÚNG |
| 10 | Adaptive trigger | KHÔNG có trong paper | ✓ Đóng góp gốc | ⭐ ĐÓNG GÓP MỚI |
| 11 | Reproducibility (seed) | (best practice) | ✓ `seed=42` | ✅ ĐÚNG |

---

<a name="7-paper"></a>
## 8. Đối Chiếu Paper Chính

### 8.1 Chawla 2002 — SMOTE Original

> Chawla N. V., Bowyer K. W., Hall L. O., Kegelmeyer W. P. (2002), "SMOTE: Synthetic Minority Over-sampling Technique", **JAIR vol 16, 321-357**.

**§6.1 — SMOTE Continuous** (numeric, interpolation): ❌ KHÔNG implement.  
**§6.2 — SMOTE-N (Nominal)** (Hamming + mode voting): ✓ IMPLEMENT chuẩn trong [SMOTE.java](../src/SMOTE.java).

### 8.2 Han 2005 — Borderline-SMOTE

> Han, H., Wang, W.Y., & Mao, B.H. (2005). "Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning". **ICIC 2005, LNCS 3644, pp. 878-887**.

**Borderline-SMOTE1**: SAFE/DANGER/NOISE classification + oversample chỉ DANGER. ✓ IMPLEMENT trong [BorderlineSMOTE.java](../src/BorderlineSMOTE.java).

**Lưu ý**: Code chuyển sang **SMOTE-N variant của Borderline** (Hamming thay Euclidean) — đây là **đóng góp gốc**, paper Han 2005 dùng Euclidean cho numeric.

### 8.3 So sánh implementation

| File | Paper | Section | Distance | Synthesis |
|---|---|---|---|---|
| [SMOTE.java](../src/SMOTE.java) | Chawla 2002 | §6.2 SMOTE-N | Hamming | Mode voting |
| [BorderlineSMOTE.java](../src/BorderlineSMOTE.java) | Han 2005 | Algo Borderline-SMOTE1 + Chawla §6.2 | Hamming | Mode voting |

---

## 9. Kết Luận Verification

### ✅ Điểm mạnh

1. **Đúng paper**: SMOTE-N + Borderline-SMOTE-N implement đúng chuẩn Chawla 2002 §6.2 + Han 2005.
2. **Không data leakage**: SMOTE chỉ áp dụng trên trainData trong CV loop.
3. **Adaptive trigger**: Đóng góp gốc, chỉ bật khi cần → tránh over-oversampling.
4. **Edge case**: Có fallback cho class < 2 records (duplicate).
5. **Reproducible**: Seed cố định.

### ⚠️ Điểm cần lưu ý khi thuyết trình

1. **Phân biệt SMOTE numeric vs SMOTE-N** — nếu thầy cô hỏi "tại sao không có interpolation 2D?" → trả lời: "dữ liệu CMAR là categorical, dùng SMOTE-N §6.2 của Chawla, không phải SMOTE gốc §6.1".
2. **`targetRatio` khác N%** — phải giải thích tương đương về mặt toán học.
3. **Borderline-SMOTE-N là tổ hợp** — Borderline (Han 2005) gốc dùng cho numeric; code này merge với SMOTE-N (Chawla §6.2) → đóng góp gốc.

### 📊 Kết quả thực nghiệm xác nhận

- **Lymph**: F1 0.42 → 0.74 sau SMOTE (v11 → v13)
- **20 UCI datasets**: SMOTE không làm Accuracy giảm trên hầu hết datasets (xác nhận trong [v12_borderline_smote_results.md](v12_borderline_smote_results.md))

---

**File này tổng hợp toàn bộ phân tích để verify lại độc lập. Mọi reference đều có line number — click để jump đến code gốc.**
