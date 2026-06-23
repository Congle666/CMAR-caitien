import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Thành phần khai thác luật của CMAR — Li, Han & Pei (2001) §3.2.
 *
 * Khác với FP-Growth thuần (coi item lớp như item bình thường và cần
 * một pass hậu kỳ để trích luật), bộ miner này làm việc trên CR-tree
 * có nhận thức về phân phối lớp: mỗi nút lưu số đếm theo từng lớp,
 * và các Class Association Rule được sinh trực tiếp trong lúc đệ quy
 * mà không cần đến thủ thuật chèn item "class=...".
 *
 * Pipeline:
 *   1. Đếm tần suất item và tần suất lớp; loại bỏ item dưới ngưỡng minSupport.
 *   2. Xây CR-tree ban đầu (mỗi path mang theo lớp của transaction sinh ra nó).
 *   3. Mining đệ quy: với mỗi frequent pattern P, lấy trực tiếp support theo
 *      lớp của P từ classCount tích luỹ dọc chuỗi header-table, rồi sinh một
 *      CAR cho mỗi lớp c có supCount(P, c) ≥ minSupport và confidence ≥ minConfidence.
 */
public class FPGrowth {

    private final int minSupport;
    private int maxPatternLength = Integer.MAX_VALUE;

    // --- MCWCAR (Wu et al. 2024 EAAI): CCO weighted-support ---
    /**
     * Bật weighted-support dựa trên correlation coefficient (CCO / φ-coefficient).
     * Mỗi item nhận trọng số w(i) = max_c |CCO(i, c)| ∈ [0,1]. Item tương quan
     * mạnh với MỘT lớp (kể cả minority count thấp) được "nâng" support → sống sót
     * qua ngưỡng minSupport. Mặc định TẮT để giữ baseline CMAR gốc.
     *
     * <p>CCO(i,c) = [P(i,c) − P(i)P(c)] / √[P(i)P(c)(1−P(i))(1−P(c))]  — Eq.(3).</p>
     */
    private boolean useCcoWeighting = false;
    /** Trọng số item, tính 1 lần đầu mine(). w ∈ [0,1]. */
    private Map<String, Double> itemWeight = new HashMap<>();

    /**
     * Chế độ tính trọng số item (MCWCAR Wu et al. 2024):
     * <ul>
     *   <li><b>CCO</b> — φ-coefficient Eq.(3): chỉ tương quan TUYẾN TÍNH item↔class.</li>
     *   <li><b>MI</b>  — Mutual Information Eq.(4): bắt cả tuyến tính LẪN phi tuyến.</li>
     *   <li><b>HYBRID</b> — max(|CCO|, MI): lấy tín hiệu mạnh nhất từ cả hai (đúng
     *       tinh thần paper dùng MI cho feature + CCO cho class).</li>
     * </ul>
     */
    public enum WeightMode { CCO, MI, HYBRID }
    private WeightMode weightMode = WeightMode.CCO;

    /** Bật/tắt weighted-support (MCWCAR). Mặc định false = CMAR gốc. */
    public void setUseCcoWeighting(boolean use) { this.useCcoWeighting = use; }
    /** Chọn chế độ trọng số: CCO (mặc định) / MI / HYBRID. */
    public void setWeightMode(WeightMode mode)  { this.weightMode = mode; }

    /**
     * Per-class minimum support (MSApriori-lite — Liu, Hsu, Ma 1999 KDD).
     * Map class → ngưỡng support riêng. Null = dùng minSupport global (hành vi cũ).
     * Class hiếm → ngưỡng thấp → CAR cho class hiếm sống sót qua cắt tỉa.
     */
    private Map<String, Integer> minSupByClass = null;
    /** Set per-class minSup. Null = tắt (global minSupport). */
    public void setMinSupByClass(Map<String, Integer> m) { this.minSupByClass = m; }

    /**
     * Tính minSup per-class theo heuristic log-based (Liu 1999 + class-specific threshold).
     * weight_c = log(|c|+1)/log(N+1) ∈ (0,1]; minSup_c = clamp(global×weight_c, FLOOR, global).
     * Lớp đông → ~global (giữ tight); lớp hiếm → FLOOR (cứu rule). FLOOR≥2 chống overfit 1 mẫu.
     */
    public static Map<String, Integer> computeMinSupByClass(
            Map<String, Integer> classCount, int globalMinSup) {
        int N = classCount.values().stream().mapToInt(Integer::intValue).sum();
        Map<String, Integer> out = new HashMap<>();
        double lnN = Math.log(N + 1);
        for (Map.Entry<String, Integer> e : classCount.entrySet()) {
            double w = (lnN > 0) ? Math.log(e.getValue() + 1) / lnN : 1.0;  // ∈(0,1]
            int raw = (int) Math.round(globalMinSup * w);
            int floor = Math.max(2, (int) Math.ceil(0.5 * e.getValue()));
            floor = Math.min(floor, globalMinSup);                 // floor không vượt global
            out.put(e.getKey(), Math.max(floor, Math.min(raw, globalMinSup)));
        }
        return out;
    }

    private final List<FrequentPattern> patterns = new ArrayList<>();
    private final List<AssociationRule> rules    = new ArrayList<>();

    private FPTree initialTree;

    // Cache trong mine() để mineTree() không phải nhận tham số dài dòng.
    private double minConfidence;
    private int    totalTransactions;

    public FPGrowth(int minSupport) {
        this.minSupport = minSupport;
    }

    /** Đặt độ dài tối đa của pattern, giới hạn bộ nhớ trên dữ liệu nhiều chiều. */
    public void setMaxPatternLength(int maxLen) {
        this.maxPatternLength = maxLen;
    }

    /** CR-tree ban đầu được xây trong lần mine() gần nhất. */
    public FPTree getInitialTree() {
        return initialTree;
    }

    /** Các frequent pattern tìm được trong lần mine() gần nhất (dùng cho báo cáo). */
    public List<FrequentPattern> getPatterns() {
        return patterns;
    }

    /** Các CAR sinh trực tiếp trong lần mine() gần nhất. */
    public List<AssociationRule> getRules() {
        return rules;
    }

    /**
     * Khai thác Class Association Rule từ các transaction huấn luyện.
     *
     * @param trainData     các transaction huấn luyện (item + nhãn lớp)
     * @param minConfidence ngưỡng confidence tối thiểu của luật
     * @return danh sách CAR (đã sắp theo thứ tự ưu tiên)
     */
    public List<AssociationRule> mine(List<Transaction> trainData,
                                      double minConfidence) {
        patterns.clear();
        rules.clear();
        this.minConfidence     = minConfidence;
        this.totalTransactions = trainData.size();

        // --- Bước 1: tần suất item toàn cục (chỉ thuộc tính, không tính class) ---
        Map<String, Integer> freq = new HashMap<>();
        for (Transaction t : trainData) {
            for (String item : t.getItems()) {
                freq.merge(item, 1, Integer::sum);
            }
        }
        // --- MCWCAR: tính trọng số CCO cho mỗi item TRƯỚC khi cắt theo minSupport ---
        // (phải tính trên TẤT cả item, vì weighted-support có thể cứu item count thấp)
        if (useCcoWeighting) {
            computeCcoWeights(trainData, freq);
            // Weighted-support: effectiveWeight = 1 + |CCO| ∈ [1,2] (WARM, Wang 2000).
            // Item tương quan mạnh được BOOST support tới 2× → sống sót dù count thấp.
            // Item vô dụng (w≈0) giữ nguyên → weighted-support CHỈ nới lỏng, không siết.
            freq.entrySet().removeIf(e -> {
                double w = itemWeight.getOrDefault(e.getKey(), 0.0);
                return e.getValue() * (1.0 + w) < minSupport;
            });
        } else {
            freq.entrySet().removeIf(e -> e.getValue() < minSupport);
        }
        if (freq.isEmpty()) {
            this.initialTree = new FPTree(minSupport);
            return rules;
        }

        // --- Bước 2: xây CR-tree ban đầu ---
        FPTree tree = new FPTree(minSupport);
        tree.headerFreq.putAll(freq);

        for (Transaction t : trainData) {
            List<String> path = t.getItems().stream()
                .filter(freq::containsKey)
                .sorted((a, b) -> {
                    int c = freq.get(b) - freq.get(a);
                    return c != 0 ? c : a.compareTo(b);
                })
                .collect(Collectors.toList());

            if (!path.isEmpty()) {
                Map<String, Integer> classDist = new HashMap<>();
                classDist.put(t.getClassLabel(), 1);
                tree.insertPath(path, classDist, 1);
            }
        }
        this.initialTree = tree;

        // --- Bước 3: khai thác đệ quy ---
        mineTree(tree, new ArrayList<>());

        Collections.sort(rules);  // thứ tự ưu tiên luật theo CMAR
        return rules;
    }

    // -----------------------------------------------------------------------
    // Khai thác đệ quy — có nhận thức về lớp
    // -----------------------------------------------------------------------

    private void mineTree(FPTree tree, List<String> prefix) {
        if (prefix.size() >= maxPatternLength) return;

        for (String item : tree.getItemsSortedByFreqAsc()) {
            int itemSupport = tree.headerFreq.get(item);

            // Pattern P = prefix ∪ {item}
            List<String> newPattern = new ArrayList<>(prefix);
            newPattern.add(item);
            HashSet<String> patternSet = new HashSet<>(newPattern);
            patterns.add(new FrequentPattern(patternSet, itemSupport));

            // --- Gom phân phối lớp của P qua chuỗi header ---
            Map<String, Integer> classDistForP = new HashMap<>();
            FPNode node = tree.headerFirst.get(item);
            while (node != null) {
                for (Map.Entry<String, Integer> e : node.classCount.entrySet()) {
                    classDistForP.merge(e.getKey(), e.getValue(), Integer::sum);
                }
                node = node.nodeLink;
            }

            // --- Sinh một CAR cho mỗi lớp đạt ngưỡng và đạt minConfidence ---
            for (Map.Entry<String, Integer> e : classDistForP.entrySet()) {
                String cls       = e.getKey();
                int    classSup  = e.getValue();
                // Ngưỡng support cho class này: per-class (MSApriori-lite) nếu bật, else global.
                int thr = (minSupByClass != null)
                        ? minSupByClass.getOrDefault(cls, minSupport)
                        : minSupport;
                // MCWCAR weighted-support: BOOST (1 + avg|CCO|) ∈ [1,2] → cứu rule
                // có pattern tương quan mạnh với lớp (kể cả minority count thấp).
                double effSup = useCcoWeighting
                        ? classSup * (1.0 + patternWeight(patternSet))
                        : classSup;
                if (effSup < thr) continue;

                double confidence = (double) classSup / itemSupport;
                if (confidence < minConfidence) continue;

                double support = (double) classSup / totalTransactions;

                rules.add(new AssociationRule(
                    new HashSet<>(patternSet),
                    cls,
                    support, confidence,
                    classSup, itemSupport
                ));
            }

            // --- Xây conditional pattern base kèm phân phối lớp ---
            List<List<String>>            condBase       = new ArrayList<>();
            List<Integer>                 condCounts     = new ArrayList<>();
            List<Map<String, Integer>>    condClassDists = new ArrayList<>();

            node = tree.headerFirst.get(item);
            while (node != null) {
                List<String> prefixPath = new ArrayList<>();
                FPNode ancestor = node.parent;
                while (!ancestor.isRoot()) {
                    prefixPath.add(0, ancestor.item);
                    ancestor = ancestor.parent;
                }
                if (!prefixPath.isEmpty()) {
                    condBase.add(prefixPath);
                    condCounts.add(node.count);
                    // Copy classCount để các thao tác sau này không rò rỉ ngược về nút gốc
                    condClassDists.add(new HashMap<>(node.classCount));
                }
                node = node.nodeLink;
            }

            if (condBase.isEmpty()) continue;

            // --- Tần suất bên trong conditional base ---
            Map<String, Integer> condFreq = new HashMap<>();
            for (int i = 0; i < condBase.size(); i++) {
                int cnt = condCounts.get(i);
                for (String condItem : condBase.get(i)) {
                    condFreq.merge(condItem, cnt, Integer::sum);
                }
            }
            condFreq.entrySet().removeIf(e -> e.getValue() < minSupport);
            if (condFreq.isEmpty()) continue;

            // --- Xây CR-tree điều kiện, lan truyền phân phối lớp ---
            FPTree condTree = new FPTree(minSupport);
            condTree.headerFreq.putAll(condFreq);

            for (int i = 0; i < condBase.size(); i++) {
                List<String> sortedPath = condBase.get(i).stream()
                    .filter(condFreq::containsKey)
                    .sorted((a, b) -> {
                        int c = condFreq.get(b) - condFreq.get(a);
                        return c != 0 ? c : a.compareTo(b);
                    })
                    .collect(Collectors.toList());

                if (!sortedPath.isEmpty()) {
                    condTree.insertPath(
                        sortedPath,
                        condClassDists.get(i),
                        condCounts.get(i));
                }
            }

            // --- Đệ quy với prefix đã mở rộng ---
            mineTree(condTree, newPattern);
        }
    }

    // -----------------------------------------------------------------------
    // MCWCAR — Item weighting: CCO (φ-coefficient) + MI (Mutual Information)
    // -----------------------------------------------------------------------

    /**
     * Tính trọng số item theo {@link #weightMode} (Wu et al. 2024 EAAI):
     * <ul>
     *   <li>CCO Eq.(3): w = max_c |φ(i,c)| — tương quan tuyến tính.</li>
     *   <li>MI  Eq.(4): w = MI(i;C) normalized — tuyến tính + phi tuyến.</li>
     *   <li>HYBRID: w = max(|CCO|, MI).</li>
     * </ul>
     * Tất cả ∈ [0,1].
     */
    private void computeCcoWeights(List<Transaction> trainData, Map<String, Integer> itemFreq) {
        itemWeight.clear();
        int n = trainData.size();
        if (n == 0) return;

        // Tần suất lớp + đồng xuất hiện (item, class)
        Map<String, Integer> classFreq = new HashMap<>();
        Map<String, Map<String, Integer>> jointFreq = new HashMap<>();  // item -> class -> count
        for (Transaction t : trainData) {
            String c = t.getClassLabel();
            classFreq.merge(c, 1, Integer::sum);
            for (String item : t.getItems()) {
                if (!itemFreq.containsKey(item)) continue;
                jointFreq.computeIfAbsent(item, k -> new HashMap<>()).merge(c, 1, Integer::sum);
            }
        }

        for (Map.Entry<String, Integer> ie : itemFreq.entrySet()) {
            String item = ie.getKey();
            int itemCount = ie.getValue();
            double pi = (double) itemCount / n;              // P(i)
            if (pi <= 0.0 || pi >= 1.0) { itemWeight.put(item, 0.0); continue; }

            Map<String, Integer> joints = jointFreq.getOrDefault(item, java.util.Collections.emptyMap());

            double cco = (weightMode == WeightMode.CCO || weightMode == WeightMode.HYBRID)
                ? maxAbsCco(pi, joints, classFreq, n) : 0.0;
            double mi  = (weightMode == WeightMode.MI  || weightMode == WeightMode.HYBRID)
                ? mutualInfo(itemCount, joints, classFreq, n) : 0.0;

            double w;
            switch (weightMode) {
                case MI:     w = mi; break;
                case HYBRID: w = Math.max(cco, mi); break;
                case CCO:
                default:     w = cco; break;
            }
            itemWeight.put(item, w);
        }
    }

    /** Trọng số CCO = max_c |φ-coefficient(i,c)| ∈ [0,1] (Eq.3). */
    private double maxAbsCco(double pi, Map<String, Integer> joints,
                             Map<String, Integer> classFreq, int n) {
        double maxAbs = 0.0;
        for (Map.Entry<String, Integer> ce : classFreq.entrySet()) {
            double pc = (double) ce.getValue() / n;          // P(c)
            if (pc <= 0.0 || pc >= 1.0) continue;
            double pic = (double) joints.getOrDefault(ce.getKey(), 0) / n;  // P(i,c)
            double denom = Math.sqrt(pi * pc * (1 - pi) * (1 - pc));
            if (denom <= 0.0) continue;
            maxAbs = Math.max(maxAbs, Math.abs((pic - pi * pc) / denom));
        }
        return maxAbs;
    }

    /**
     * Mutual Information MI(i; C) cho item nhị phân (có/vắng) vs class (Eq.4),
     * chuẩn hoá về [0,1] bằng cách chia entropy lớp H(C) (= normalized MI / uncertainty
     * coefficient). Bắt được cả tương quan tuyến tính LẪN phi tuyến giữa item và class.
     *
     * <p>MI(i;C) = Σ_{x∈{present,absent}} Σ_c P(x,c)·log2[ P(x,c) / (P(x)·P(c)) ]</p>
     */
    private double mutualInfo(int itemCount, Map<String, Integer> joints,
                              Map<String, Integer> classFreq, int n) {
        double pPresent = (double) itemCount / n;
        double pAbsent  = 1.0 - pPresent;
        double mi = 0.0, hC = 0.0;

        for (Map.Entry<String, Integer> ce : classFreq.entrySet()) {
            int classCount = ce.getValue();
            double pc = (double) classCount / n;
            if (pc <= 0.0) continue;
            hC -= pc * log2(pc);   // entropy lớp

            int jPresent = joints.getOrDefault(ce.getKey(), 0);
            int jAbsent  = classCount - jPresent;

            // ô (item present, class c)
            if (jPresent > 0 && pPresent > 0) {
                double pxc = (double) jPresent / n;
                mi += pxc * log2(pxc / (pPresent * pc));
            }
            // ô (item absent, class c)
            if (jAbsent > 0 && pAbsent > 0) {
                double pxc = (double) jAbsent / n;
                mi += pxc * log2(pxc / (pAbsent * pc));
            }
        }
        if (hC <= 0.0) return 0.0;
        double norm = mi / hC;                 // ∈ [0,1] (uncertainty coefficient)
        return norm < 0 ? 0.0 : (norm > 1 ? 1.0 : norm);
    }

    private static double log2(double x) { return Math.log(x) / Math.log(2.0); }

    /** Trọng số pattern = trung bình trọng số item (Wu et al. 2024, weighted support). */
    private double patternWeight(java.util.Collection<String> pattern) {
        if (pattern.isEmpty()) return 0.0;
        double sum = 0.0;
        for (String item : pattern) sum += itemWeight.getOrDefault(item, 0.0);
        return sum / pattern.size();
    }
}
