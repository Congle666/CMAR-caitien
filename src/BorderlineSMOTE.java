import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Borderline-SMOTE-N — biến thể categorical của Borderline-SMOTE (Han et al. 2005).
 *
 * <p>Khác với SMOTE-N (Chawla 2002) tạo synthetic cho TẤT CẢ minority records,
 * Borderline-SMOTE chỉ tạo synthetic cho các minority records nằm ở
 * <b>borderline</b> (gần ranh giới quyết định) — vùng dễ bị classifier nhầm.</p>
 *
 * <p>Phân loại minority records (Han et al. 2005):</p>
 * <ul>
 *   <li><b>SAFE</b>: most k-NN (≥ k/2) cùng class → đã trong vùng an toàn → skip</li>
 *   <li><b>DANGER</b>: k/2 ≤ majority neighbors &lt; k → gần biên → OVERSAMPLE</li>
 *   <li><b>NOISE</b>: tất cả k-NN là majority → record cô lập → skip (có thể là noise)</li>
 * </ul>
 *
 * <p>Lợi ích so với SMOTE chuẩn:</p>
 * <ul>
 *   <li>Không tạo synthetic ở vùng "safe" → giảm noise → Accuracy không giảm.</li>
 *   <li>Focus oversampling vào vùng DANGER → tăng F1/Recall mạnh hơn.</li>
 *   <li>Filter NOISE records → tránh học pattern sai từ outliers.</li>
 * </ul>
 *
 * <p>Reference: Han, H., Wang, W.Y., &amp; Mao, B.H. (2005). "Borderline-SMOTE:
 * A New Over-Sampling Method in Imbalanced Data Sets Learning". ICIC 2005,
 * LNCS 3644, pp. 878-887. DOI: 10.1007/11538059_91.</p>
 */
public final class BorderlineSMOTE {

    private BorderlineSMOTE() {}  // static-only

    /**
     * Áp dụng Borderline-SMOTE-N: oversample DANGER minority records lên target size.
     *
     * @param data         training transactions (gồm tất cả classes)
     * @param k            số nearest neighbors (mặc định 5)
     * @param targetRatio  mỗi class hướng tới có ít nhất {@code targetRatio × maxFreq} records
     * @param seed         random seed cho reproducibility
     * @return augmented training set
     */
    public static List<Transaction> apply(List<Transaction> data, int k,
                                            double targetRatio, long seed) {
        if (data.isEmpty()) return new ArrayList<>(data);

        // Group records by class
        Map<String, List<Transaction>> byClass = new HashMap<>();
        for (Transaction t : data) {
            byClass.computeIfAbsent(t.getClassLabel(), c -> new ArrayList<>()).add(t);
        }

        int maxFreq = 0;
        for (List<Transaction> g : byClass.values()) {
            if (g.size() > maxFreq) maxFreq = g.size();
        }
        int target = (int) Math.round(maxFreq * targetRatio);
        Random rng = new Random(seed);

        List<Transaction> augmented = new ArrayList<>(data);

        for (Map.Entry<String, List<Transaction>> e : byClass.entrySet()) {
            String cls = e.getKey();
            List<Transaction> records = e.getValue();

            if (records.size() >= target) continue;
            int needed = target - records.size();

            // Edge case: class quá ít records (< 2) → duplicate
            if (records.size() < 2) {
                for (int i = 0; i < needed; i++) {
                    augmented.add(records.get(i % records.size()));
                }
                continue;
            }

            // ----- BƯỚC 1: Phân loại minority records thành SAFE/DANGER/NOISE -----
            List<Transaction> dangerSet = new ArrayList<>();
            int kEff = Math.min(k, data.size() - 1);

            for (Transaction r : records) {
                // Find k-NN trong TOÀN BỘ data (gồm cả majority)
                List<Transaction> neighbors = kNearestNeighborsGlobal(r, data, kEff);
                int majorityCount = 0;
                for (Transaction n : neighbors) {
                    if (!n.getClassLabel().equals(cls)) {
                        majorityCount++;
                    }
                }
                // Phân loại theo paper Han 2005 (Section 3):
                //   NOISE  : majorityCount == kEff (tất cả là majority)
                //   DANGER : ⌈k/2⌉ ≤ majorityCount < kEff   ← BUG #3 fix: ceiling
                //   SAFE   : majorityCount < ⌈k/2⌉
                // Trước đây dùng kEff/2 (integer division = floor) → ngưỡng quá lỏng:
                // k=5 → floor(5/2)=2 nhưng paper muốn ceil(5/2)=3.
                int halfK = (kEff + 1) / 2;  // ceiling division
                if (majorityCount == kEff) {
                    // NOISE — skip
                } else if (majorityCount >= halfK) {
                    dangerSet.add(r);  // DANGER
                }
                // SAFE — skip
            }

            // ----- BƯỚC 2: Apply SMOTE chỉ trên DANGER set -----
            List<Transaction> seedSet;
            if (dangerSet.isEmpty()) {
                // Edge case: không có DANGER → fall back to SMOTE vanilla (toàn bộ minority)
                seedSet = records;
            } else {
                seedSet = dangerSet;
            }

            int kSyn = Math.min(k, records.size() - 1);
            for (int i = 0; i < needed; i++) {
                Transaction base = seedSet.get(rng.nextInt(seedSet.size()));
                // k-NN cho synthetic — trong CÙNG class (giống SMOTE-N)
                List<Transaction> neighbors = kNearestNeighborsInClass(base, records, kSyn);
                Transaction synthetic = createSynthetic(base, neighbors, cls, rng);
                augmented.add(synthetic);
            }
        }

        return augmented;
    }

    /** Convenience: targetRatio = 1.0 (fully balanced), k = 5, seed = 42. */
    public static List<Transaction> apply(List<Transaction> data) {
        return apply(data, 5, 1.0, 42);
    }

    // -----------------------------------------------------------------------

    /**
     * k-NN trong TOÀN BỘ data (gồm majority + minority). Dùng cho DANGER detection.
     */
    private static List<Transaction> kNearestNeighborsGlobal(
            Transaction base, List<Transaction> pool, int k) {
        List<Map.Entry<Transaction, Integer>> distances = new ArrayList<>();
        Set<String> baseItems = new HashSet<>(base.getItems());

        for (Transaction t : pool) {
            if (t == base) continue;
            int d = hammingDistance(baseItems, t.getItems());
            distances.add(new java.util.AbstractMap.SimpleEntry<>(t, d));
        }
        distances.sort(Comparator.comparingInt(Map.Entry::getValue));

        List<Transaction> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, distances.size()); i++) {
            result.add(distances.get(i).getKey());
        }
        return result;
    }

    /**
     * k-NN chỉ trong CÙNG class. Dùng cho synthetic generation (giống SMOTE-N).
     */
    private static List<Transaction> kNearestNeighborsInClass(
            Transaction base, List<Transaction> pool, int k) {
        List<Map.Entry<Transaction, Integer>> distances = new ArrayList<>();
        Set<String> baseItems = new HashSet<>(base.getItems());

        for (Transaction t : pool) {
            if (t == base) continue;
            int d = hammingDistance(baseItems, t.getItems());
            distances.add(new java.util.AbstractMap.SimpleEntry<>(t, d));
        }
        distances.sort(Comparator.comparingInt(Map.Entry::getValue));

        List<Transaction> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, distances.size()); i++) {
            result.add(distances.get(i).getKey());
        }
        return result;
    }

    private static int hammingDistance(Set<String> baseItems, List<String> other) {
        Map<String, String> baseMap = itemsToMap(baseItems);
        Map<String, String> otherMap = itemsToMap(other);

        Set<String> allAttrs = new HashSet<>(baseMap.keySet());
        allAttrs.addAll(otherMap.keySet());

        int diff = 0;
        for (String attr : allAttrs) {
            String bv = baseMap.get(attr);
            String ov = otherMap.get(attr);
            if (bv == null || ov == null || !bv.equals(ov)) diff++;
        }
        return diff;
    }

    private static Map<String, String> itemsToMap(Iterable<String> items) {
        Map<String, String> m = new HashMap<>();
        for (String item : items) {
            int idx = item.indexOf('=');
            if (idx > 0) m.put(item.substring(0, idx), item.substring(idx + 1));
        }
        return m;
    }

    /** Mode voting để tạo synthetic — giống SMOTE-N (Chawla 2002 §6.2). */
    private static Transaction createSynthetic(Transaction base,
                                                 List<Transaction> neighbors,
                                                 String classLabel,
                                                 Random rng) {
        Map<String, Map<String, Integer>> attrValueCounts = new HashMap<>();

        addItemsToCount(base.getItems(), attrValueCounts);
        for (Transaction n : neighbors) {
            addItemsToCount(n.getItems(), attrValueCounts);
        }

        List<String> syntheticItems = new ArrayList<>();
        for (Map.Entry<String, Map<String, Integer>> e : attrValueCounts.entrySet()) {
            String attr = e.getKey();
            Map<String, Integer> valueCounts = e.getValue();

            int maxCount = 0;
            for (int c : valueCounts.values()) if (c > maxCount) maxCount = c;

            List<String> modes = new ArrayList<>();
            for (Map.Entry<String, Integer> ve : valueCounts.entrySet()) {
                if (ve.getValue() == maxCount) modes.add(ve.getKey());
            }

            String chosen = modes.get(rng.nextInt(modes.size()));
            syntheticItems.add(attr + "=" + chosen);
        }

        return new Transaction(syntheticItems, classLabel);
    }

    private static void addItemsToCount(List<String> items,
                                          Map<String, Map<String, Integer>> counts) {
        for (String item : items) {
            int idx = item.indexOf('=');
            if (idx <= 0) continue;
            String attr = item.substring(0, idx);
            String value = item.substring(idx + 1);
            counts.computeIfAbsent(attr, a -> new HashMap<>())
                  .merge(value, 1, Integer::sum);
        }
    }
}
