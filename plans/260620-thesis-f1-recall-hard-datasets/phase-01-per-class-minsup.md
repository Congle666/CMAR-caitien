# Phase 1 — Per-Class Minimum Support (MSApriori-lite) trong FPGrowth

## Context links
- Plan: [plan.md](plan.md)
- Research: [researcher-01-techniques.md](research/researcher-01-techniques.md) (Technique 1 MSApriori, Technique 2 class-specific threshold)
- Scout: [scout-01-codebase.md](scout/scout-01-codebase.md)

## Overview
- **Date:** 2026-06-20 · **Priority:** CAO NHẤT · **Status:** TODO
- Sinh thêm CAR cho class cực hiếm bằng cách hạ ngưỡng support RIÊNG cho từng class
  (class hiếm → minSup thấp; class đông → minSup cao). Đây là bottleneck thật: hiện
  CAR cho glass class3 bị cắt mất ở ngưỡng global → classifier không có luật để vote.

## Key Insights
1. CMAR sinh CAR per-class ngay trong đệ quy (FPGrowth.java dòng 188-210), KHÔNG phải
   FP-Growth thuần. Có ĐÚNG 1 chỗ quyết định "giữ/cắt rule theo support":
   - Site A — pattern chính: dòng ~194/196 `classSup * (1+w) < minSupport` / `classSup < minSupport`.
   - Site B — conditional base: dòng ~244 `condFreq < minSupport` (cắt ITEM trong nhánh,
     KHÔNG per-class → KHÔNG đụng ở Phase 1 để tránh bùng nổ; xem Risk).
2. `minSupport` hiện là 1 `int` global (constructor). MSApriori đầy đủ (Liu 1999) dùng
   MIS per-ITEM. KISS: ta làm **per-CLASS minSup** (đủ giải quyết bottleneck, ít code,
   ít rủi ro bùng nổ hơn per-item). Đây là "MSApriori-lite" lai class-specific threshold
   (researcher-01 Technique 2).
3. Đã có cơ chế weighted-support `(1+w)` cùng vị trí → per-class minSup phải KẾT HỢP
   nhân hoà với nó, không phá. Thứ tự: tính `effSup = classSup*(1+w)`, so với `minSup(cls)`.
4. minSup tính lại sau SMOTE trong CrossValidator → per-class minSup phải tính trên
   distribution SAU SMOTE (distribution mà tree thực sự xây). Cần truyền class-count map.

## Requirements
- R1: FPGrowth nhận map `minSupByClass: Map<String,Integer>` (nullable). Null → hành vi cũ.
- R2: Khi map != null, tại Site A so sánh `effSup < minSupByClass.get(cls)` thay vì global.
- R3: Cờ bật/tắt ở tầng gọi (CMARClassifier/CrossValidator); default OFF = baseline.
- R4: Công thức minSup per-class derive từ class size, có sàn (floor) để chống bùng nổ.

## Architecture
```
CrossValidator.runWithMetrics(... boolean usePerClassMinSup ...)
   │  (sau SMOTE) tính classCount của trainData
   ▼
computeMinSupByClass(classCount, globalMinSup)  ← hàm tĩnh mới (KISS, testable)
   │  Map<String,Integer>
   ▼
FPGrowth.setMinSupByClass(map)   ← setter mới, field nullable
   │
   ▼
mineTree() Site A: dùng minSupByClass.getOrDefault(cls, globalMinSup)
```

### Công thức minSup per-class (heuristic log-based, researcher-01 Eq.)
```
weight_c   = log(|class_c| + 1) / log(N + 1)        // ∈ (0,1], lớp đông → gần 1
rawMinSup_c = round(globalMinSup * weight_c)
minSup_c   = clamp(rawMinSup_c, FLOOR, globalMinSup) // không vượt global, không dưới FLOOR
FLOOR      = max(2, ceil(0.5 * |class_c| * trainFrac))  // chống rule trên 1 mẫu
```
- Lớp đông: weight_c≈1 → minSup_c≈global (giữ tight, không bùng nổ, không giảm precision).
- Lớp hiếm: weight_c nhỏ → minSup_c giảm tới FLOOR → CAR cho class hiếm sống sót.
- FLOOR đảm bảo rule phải cover ≥2 mẫu (tránh overfit 1 instance → precision crash).

## Related code files
- `src/FPGrowth.java` — thêm field `minSupByClass`, setter, sửa Site A (dòng ~188-210).
- `src/CMARClassifier.java` — thêm cờ `usePerClassMinSup`, truyền xuống FPGrowth khi train.
- `src/CrossValidator.java` — overload mới truyền cờ; tính classCount sau SMOTE; gọi
  `computeMinSupByClass`. Đặt hàm tĩnh `computeMinSupByClass` ở đây (hoặc FPGrowth).
- `src/BenchmarkImbalanced.java` — factory bật cờ để verify.

## Implementation Steps
1. **FPGrowth: field + setter.**
   ```java
   private Map<String,Integer> minSupByClass = null; // null = dùng minSupport global
   public void setMinSupByClass(Map<String,Integer> m){ this.minSupByClass = m; }
   ```
2. **FPGrowth Site A (dòng ~193-197): so sánh theo class.**
   ```java
   int thr = (minSupByClass != null)
             ? minSupByClass.getOrDefault(cls, minSupport)
             : minSupport;
   double eff = useCcoWeighting ? classSup*(1.0+patternWeight(patternSet)) : classSup;
   if (eff < thr) continue;
   ```
   GIỮ NGUYÊN confidence check phía sau. KHÔNG đụng Site B (dòng 244).
3. **Hàm tĩnh `computeMinSupByClass`.**
   ```java
   static Map<String,Integer> computeMinSupByClass(
            Map<String,Integer> classCount, int globalMinSup){
     int N = classCount.values().stream().mapToInt(i->i).sum();
     Map<String,Integer> out = new HashMap<>();
     double lnN = Math.log(N+1);
     for (var e : classCount.entrySet()){
       double w = Math.log(e.getValue()+1)/lnN;        // ∈(0,1]
       int raw = (int)Math.round(globalMinSup * w);
       int floor = Math.max(2, (int)Math.ceil(0.5 * e.getValue() * /*trainFrac≈1, đã là train*/ 1.0));
       floor = Math.min(floor, globalMinSup);           // floor không vượt global
       out.put(e.getKey(), Math.max(floor, Math.min(raw, globalMinSup)));
     }
     return out;
   }
   ```
   (classCount đã là của trainData sau SMOTE → trainFrac=1.)
4. **CrossValidator: overload + wiring.** Sau khi xây trainData (sau SMOTE), tính
   classCount, gọi computeMinSupByClass, gọi `fp.setMinSupByClass(map)` chỉ khi cờ ON.
5. **CMARClassifier: cờ `usePerClassMinSup` + setter**, truyền vào FPGrowth ở train().
6. **BenchmarkImbalanced:** bật cờ trong factory; chạy; lưu CSV mới (vd `result/p1_*`).
7. **Verify glass riêng**: in per-class F1 glass class3 trước/sau.

## Todo list
- [ ] FPGrowth: field + setter + sửa Site A
- [ ] computeMinSupByClass + sanity check (lớp đông→global, lớp hiếm→floor)
- [ ] CrossValidator overload + tính classCount sau SMOTE
- [ ] CMARClassifier cờ + wiring train()
- [ ] BenchmarkImbalanced factory bật cờ + lưu CSV
- [ ] Build + chạy + ghi bảng trước-sau (glass class3 + 5 imbalanced + per-class)
- [ ] Kiểm precision class hiếm KHÔNG sụp (xem Risk)

## Success Criteria
- glass class3 F1: 0.14 → **≥ 0.30** (P và R đều > 0).
- 5 imbalanced macroF1: **≥ 0.751** (không tụt), recall ≥ 0.779.
- Số CAR sinh ra tăng có kiểm soát (log tổng #rules; tăng < 3× là chấp nhận).
- precision class hiếm (glass class3, zoo class nhỏ) không < 0.10 sau cải tiến.

## Risk Assessment
- **Bùng nổ rule** nếu hạ minSup quá sâu / đụng Site B → giữ Site B nguyên, có FLOOR≥2,
  cap minSup_c ≤ global. Log #rules mỗi fold; nếu >3× baseline → nâng FLOOR.
- **Giảm precision class hiếm** (nhiều luật yếu → false positive). Mitigation: FLOOR cover
  ≥2 mẫu; confidence check giữ nguyên; nếu precision sụp, tăng minConf riêng class hiếm
  (đẩy sang Phase 4 F1-tuning, KHÔNG làm trong Phase 1 để giữ KISS).
- **Tương tác với weighted-support (1+w)**: đã xử lý ở Step 2 (nhân eff trước khi so thr).
- **SMOTE đổi distribution**: classCount lấy SAU SMOTE → đúng tree thực tế.

## Security Considerations
N/A (offline batch ML, không I/O ngoài, không untrusted input). Chỉ lưu ý: tránh chia
cho 0 khi N hoặc class size = 0 (đã guard bằng +1 trong log).

## Next steps
→ Phase 2 (đảm bảo cờ này CHỈ bật trên dataset imbalanced qua CIR gating, dataset cân
bằng giữ baseline). Per-class minSup nên nằm DƯỚI cùng một gate với AV/cost-sensitive.
