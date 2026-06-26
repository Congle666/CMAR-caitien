import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Bộ phân lớp CMAR — Li, Han & Pei (2001) ICDM.
 *
 * <p>Huấn luyện (train):</p>
 * <ul>
 *   <li><b>Cắt tỉa 1</b> — General rule pruning: loại bỏ luật cụ thể hơn đang
 *       bị thống trị bởi một luật tổng quát hơn có hạng cao hơn và cùng nhãn.</li>
 *   <li><b>Cắt tỉa 2</b> — Chi-square + tương quan dương: chỉ giữ luật có χ²
 *       vượt ngưỡng ý nghĩa (mặc định 3.841, p=0.05) VÀ tương quan dương.</li>
 *   <li><b>Cắt tỉa 3</b> — Database coverage (Thuật toán 1): lặp chọn luật phủ
 *       các bản ghi training; bản ghi được phủ đủ δ luật thì loại.</li>
 * </ul>
 *
 * <p>Cải tiến H4/H5/H6 (paper Geng et al. 2025 review):</p>
 * <ul>
 *   <li><b>H4 — Stratified Coverage</b>: Bảo vệ top-K luật cho mỗi lớp khỏi
 *       database coverage pruning. Đảm bảo class hiếm có đại diện trong rule set.
 *       Inspired by iTCAR (Ong et al. 2020) và CARGBA (Cong et al. 2005).</li>
 *   <li><b>H5b — Composite voting weight</b>: Vote weight = confidence × Lift
 *       với top-K rules per class. Defensible khi data đã balance (post-SMOTE).
 *       Inspired by Brin et al. 1997 + Bahri et al. 2020. Optional via
 *       {@link #setUseConfLiftVoting}. Active default trong v14 thesis pipeline.</li>
 *   <li><b>H6 — Top-K voting</b>: K luật mạnh nhất per class khi vote. Set qua
 *       {@link #setVoteTopK}. Default 0 (= dùng tất cả rules).</li>
 * </ul>
 *
 * <p>Phân lớp (classify):</p>
 * <ol>
 *   <li>Lấy tất cả luật trong CR-tree khớp với bản ghi test.</li>
 *   <li>Không có luật khớp → trả về lớp mặc định.</li>
 *   <li>Tất cả luật khớp cùng một lớp → trả về lớp đó.</li>
 *   <li>Ngược lại: score(c) = Σ [χ²(r)]² / maxχ²(r); lớp có score lớn nhất.</li>
 * </ol>
 */
public class CMARClassifier {

    private List<AssociationRule> rules;
    private CRTree crTree;
    private String defaultClass;
    private int totalTransactions;
    private Map<String, Integer> classFreq;
    private Map<String, Integer> itemFreq;   // dùng để sắp thứ tự path trong CR-tree

    // --- Tham số cắt tỉa ---
    private double chiSquareThreshold = 3.841;
    private int    coverageThreshold  = 4;

    // --- H4/H5/H6 config (Geng et al. 2025 review) ---
    /** H4: Bảo vệ top-K luật cho mỗi lớp khỏi coverage pruning. 0 = tắt. */
    private int stratifiedTopK = 0;
    /** H6: top-K luật mạnh nhất per class khi voting (giữ χ² scoring). 0 = dùng tất cả. */
    private int voteTopK = 0;
    /**
     * H5b: Bật composite voting weight = confidence × Lift (Brin 1997 + Bahri 2020).
     * Khi bật, classify() dùng score(c) = Σ_{top-K} conf(r) × lift(r,c) thay vì weighted χ².
     * Đáng dùng SAU SMOTE balancing (vì lift có inherent class-size bias).
     */
    private boolean useConfLiftVoting = false;
    /**
     * H5b Adaptive: ngưỡng imbalance ratio để enable conf×Lift. Default 0 (luôn enable).
     * Khi &gt; 0, H5b chỉ active nếu originalImbalanceRatio ≥ threshold.
     * Lý do: lift có class-size bias, hại trên mild balanced data (ratio &lt; 3:1).
     * Recommended: 3.0 cho thesis (tránh regression trên tic-tac-toe, diabetes, horse).
     */
    private double adaptiveConfLiftThreshold = 0.0;
    /**
     * H5b Adaptive: imbalance ratio ORIGINAL của dataset (trước SMOTE).
     * Set externally bởi pipeline trước khi train.
     * 0 = không adaptive, fallback to classFreq post-SMOTE.
     */
    private double originalImbalanceRatio = 0.0;
    /**
     * H5e: Bật Added Value (AV) voting — Ahmed et al. 2000, MCWCAR (Wu et al. 2024 EAAI).
     * AV(rule,c) = confidence − P(c). Bất đối xứng, bắt tương quan âm, không bias
     * majority như confidence thuần → tốt cho imbalanced. score(c) = Σ_{top-K, AV>0} AV.
     */
    private boolean useAvVoting = false;
    /**
     * H5e Adaptive: ngưỡng imbalance ratio để enable AV voting. Default 0 (luôn enable).
     * Khi &gt; 0, AV chỉ active nếu originalImbalanceRatio ≥ threshold; ngược lại
     * fallback weighted-χ² (baseline). Tránh hại trên data đã cân bằng (vd tic-tac-toe).
     */
    private double adaptiveAvThreshold = 0.0;
    /**
     * Cost-sensitive voting exponent β (King &amp; Zeng 2001 threshold-moving /
     * cost-sensitive learning). Score lớp c được nhân (N/classCount)^β → lớp hiếm
     * được boost, tăng recall/F1 minority. β=0 → tắt (không boost). β=0.5 nhẹ,
     * β=1.0 = inverse-frequency đầy đủ. Chỉ áp trong classifyByAV.
     */
    private double costSensitiveBeta = 0.0;

    // --- Thống kê ---
    private int candidateCount;
    private int afterGeneralPruneCount;
    private int afterChiPruneCount;
    private int afterCoveragePruneCount;

    public void setChiSquareThreshold(double threshold) { this.chiSquareThreshold = threshold; }
    public void setCoverageThreshold(int delta)         { this.coverageThreshold = delta; }
    /** H4: Set 0 để tắt (CMAR gốc). Set 10 để giữ top-10 luật/class trong coverage pruning. */
    public void setStratifiedTopK(int k)                { this.stratifiedTopK = k; }
    /** H6: top-K rules per class khi voting (giữ χ² scoring). 0 = dùng tất cả. */
    public void setVoteTopK(int k)                      { this.voteTopK = k; }
    /** H5b: Bật composite voting = conf × Lift (Brin 1997 + Bahri 2020). Đáng dùng SAU SMOTE. */
    public void setUseConfLiftVoting(boolean use)       { this.useConfLiftVoting = use; }
    /** H5b Adaptive: chỉ enable conf×Lift khi imbalance ratio ≥ threshold. 0 = luôn enable. */
    public void setAdaptiveConfLiftThreshold(double t)  { this.adaptiveConfLiftThreshold = t; }
    /** Set imbalance ratio ORIGINAL (trước SMOTE) — dùng cho adaptive H5b decision. */
    public void setOriginalImbalanceRatio(double r)     { this.originalImbalanceRatio = r; }
    /** H5e: Bật Added Value voting (MCWCAR — Wu et al. 2024). AV = confidence − P(c). */
    public void setUseAvVoting(boolean use)             { this.useAvVoting = use; }
    /** H5e Adaptive: chỉ enable AV khi imbalance ratio ≥ threshold. 0 = luôn enable. */
    public void setAdaptiveAvThreshold(double t)        { this.adaptiveAvThreshold = t; }
    /** Cost-sensitive voting: nhân score lớp c với (N/classCount)^β. β=0 tắt. */
    public void setCostSensitiveBeta(double beta)       { this.costSensitiveBeta = beta; }

    // -----------------------------------------------------------------------
    // Huấn luyện
    // -----------------------------------------------------------------------

    public void train(List<AssociationRule> candidateRules,
                      List<Transaction> trainData) {
        this.totalTransactions = trainData.size();
        this.classFreq = new HashMap<>();
        this.itemFreq  = new HashMap<>();

        for (Transaction t : trainData) {
            classFreq.merge(t.getClassLabel(), 1, Integer::sum);
            for (String item : t.getItems()) {
                itemFreq.merge(item, 1, Integer::sum);
            }
        }

        defaultClass = classFreq.entrySet().stream()
            .max(Map.Entry.comparingByValue())
            .map(Map.Entry::getKey)
            .orElse("");

        Collections.sort(candidateRules);
        this.candidateCount = candidateRules.size();

        // --- Cắt tỉa 1: luật tổng quát ---
        List<AssociationRule> afterPrune1 = pruneByGeneralRules(candidateRules);
        this.afterGeneralPruneCount = afterPrune1.size();
        System.out.println("    Cat tia 1 (luat tong quat):     "
            + candidateRules.size() + " -> " + afterPrune1.size());

        // --- Cắt tỉa 2: chi-square + tương quan dương ---
        List<AssociationRule> afterPrune2 = pruneByChiSquareSignificance(afterPrune1);
        this.afterChiPruneCount = afterPrune2.size();
        System.out.println("    Cat tia 2 (chi-square >= " + chiSquareThreshold + "): "
            + afterPrune1.size() + " -> " + afterPrune2.size());

        // --- Cắt tỉa 3: database coverage ---
        this.rules = pruneByDatabaseCoverage(afterPrune2, trainData);
        this.afterCoveragePruneCount = this.rules.size();
        System.out.println("    Cat tia 3 (db coverage, delta=" + coverageThreshold + "):  "
            + afterPrune2.size() + " -> " + this.rules.size());

        // --- Xây CR-tree để lưu luật nén gọn + truy vấn nhanh ---
        this.crTree = new CRTree(itemFreq);
        crTree.insertAll(this.rules);

        System.out.println("    Luat cuoi cung: " + this.rules.size()
            + " (tu " + candidateRules.size() + " ung vien; luu trong CR-tree)");
    }

    // -----------------------------------------------------------------------
    // Cắt tỉa 1: General Rule Pruning
    // -----------------------------------------------------------------------

    private List<AssociationRule> pruneByGeneralRules(List<AssociationRule> sorted) {
        List<AssociationRule> kept = new ArrayList<>();
        Map<String, List<AssociationRule>> keptByClass = new HashMap<>();

        for (AssociationRule r1 : sorted) {
            boolean dominated = false;
            List<AssociationRule> sameClass = keptByClass.get(r1.getClassLabel());
            if (sameClass != null) {
                for (AssociationRule r2 : sameClass) {
                    if (r1.getCondset().containsAll(r2.getCondset())) {
                        dominated = true;
                        break;
                    }
                }
            }
            if (!dominated) {
                kept.add(r1);
                keptByClass.computeIfAbsent(r1.getClassLabel(), k -> new ArrayList<>())
                           .add(r1);
            }
        }
        return kept;
    }

    // -----------------------------------------------------------------------
    // Cắt tỉa 2: chi-square + tương quan dương
    // -----------------------------------------------------------------------

    private List<AssociationRule> pruneByChiSquareSignificance(
            List<AssociationRule> rules) {
        List<AssociationRule> kept = new ArrayList<>();

        for (AssociationRule r : rules) {
            double chi2 = computeChiSquare(r, r.getClassLabel());
            double a = r.getSupportCount();
            double b = r.getCondsetSupportCount() - a;
            double c = classFreq.getOrDefault(r.getClassLabel(), 0) - a;
            double d = totalTransactions - a - b - c;
            boolean positivelyCorrelated = (a * d > b * c);

            if (chi2 < chiSquareThreshold || !positivelyCorrelated) continue;

            kept.add(r);
        }
        return kept;
    }

    // -----------------------------------------------------------------------
    // Cắt tỉa 3: Database Coverage (Thuật toán 1)
    // -----------------------------------------------------------------------

    private List<AssociationRule> pruneByDatabaseCoverage(
            List<AssociationRule> rules, List<Transaction> trainData) {

        int n = trainData.size();
        int[] coverCount = new int[n];
        boolean[] removed = new boolean[n];
        int remainingCount = n;

        List<AssociationRule> selected = new ArrayList<>();
        java.util.Set<AssociationRule> alreadyAdded = new java.util.HashSet<>();

        // --- H4: Stratified Coverage — bảo vệ top-K luật mỗi lớp trước khi prune ---
        // Lý do: minority class có thể bị "starved" — rules của nó bị prune sớm
        // do coverage threshold đạt từ majority records. Reserve top-K rules/class
        // để đảm bảo mỗi class có đại diện trong final rule set.
        if (stratifiedTopK > 0) {
            Map<String, List<AssociationRule>> byClass = new HashMap<>();
            for (AssociationRule r : rules) {
                byClass.computeIfAbsent(r.getClassLabel(), k -> new ArrayList<>()).add(r);
            }
            // rules đã được sort theo priority (confidence DESC, support DESC, |condset| ASC)
            // → các rule đầu trong list của mỗi class là TOP rules
            for (List<AssociationRule> classRules : byClass.values()) {
                int take = Math.min(stratifiedTopK, classRules.size());
                for (int i = 0; i < take; i++) {
                    AssociationRule r = classRules.get(i);
                    if (alreadyAdded.add(r)) {
                        selected.add(r);
                        // Update coverage cho các record được covered bởi rule này
                        for (int j = 0; j < n; j++) {
                            if (removed[j]) continue;
                            Transaction t = trainData.get(j);
                            if (r.matches(t) && r.getClassLabel().equals(t.getClassLabel())) {
                                coverCount[j]++;
                                if (coverCount[j] >= coverageThreshold) {
                                    removed[j] = true;
                                    remainingCount--;
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- Cắt tỉa 3 chuẩn: greedy coverage ---
        for (AssociationRule rule : rules) {
            if (alreadyAdded.contains(rule)) continue;
            if (remainingCount == 0) break;
            boolean coversAny = false;

            for (int i = 0; i < n; i++) {
                if (removed[i]) continue;
                Transaction t = trainData.get(i);
                if (rule.matches(t)
                        && rule.getClassLabel().equals(t.getClassLabel())) {
                    coversAny = true;
                    coverCount[i]++;
                    if (coverCount[i] >= coverageThreshold) {
                        removed[i] = true;
                        remainingCount--;
                    }
                }
            }

            if (coversAny) {
                selected.add(rule);
                alreadyAdded.add(rule);
            }
        }

        // Sort lại theo priority để giữ thứ tự khi predict
        Collections.sort(selected);
        return selected;
    }

    // -----------------------------------------------------------------------
    // Phân lớp — truy vấn qua CR-tree
    // -----------------------------------------------------------------------

    public String classify(Transaction record) {
        List<AssociationRule> matching = crTree.findMatching(record);
        if (matching.isEmpty()) return defaultClass;

        // LinkedHashMap (không HashMap) → thứ tự duyệt lớp ỔN ĐỊNH, đảm bảo
        // kết quả TÁI LẬP 100% giữa các lần chạy/máy (reproducibility).
        Map<String, List<AssociationRule>> byClass = new LinkedHashMap<>();
        for (AssociationRule rule : matching) {
            byClass.computeIfAbsent(rule.getClassLabel(), k -> new ArrayList<>())
                   .add(rule);
        }

        if (byClass.size() == 1) {
            return byClass.keySet().iterator().next();
        }

        // --- H5e: Added Value voting (MCWCAR — Wu et al. 2024 EAAI) ---
        // Adaptive gating: trên data đã cân bằng (ratio < threshold) → fallback
        // weighted-χ² (baseline) để tránh hại (vd tic-tac-toe, horse). Trên data
        // mất cân bằng → dùng AV (cứu minority).
        if (useAvVoting) {
            if (adaptiveAvThreshold > 0) {
                double ratio;
                if (originalImbalanceRatio > 0) {
                    ratio = originalImbalanceRatio;
                } else {
                    int maxFreq = 0, minFreq = Integer.MAX_VALUE;
                    for (int f : classFreq.values()) {
                        if (f > maxFreq) maxFreq = f;
                        if (f > 0 && f < minFreq) minFreq = f;
                    }
                    ratio = (minFreq > 0 && minFreq != Integer.MAX_VALUE)
                          ? (double) maxFreq / minFreq : Double.MAX_VALUE;
                }
                if (ratio < adaptiveAvThreshold) {
                    return classifyByWeightedChi2(byClass);
                }
            }
            return classifyByAV(byClass);
        }

        // --- H5b: Composite voting weight = confidence × Lift ---
        // Inspired by Brin et al. (1997) — Lift đo correlation thực
        // và Bahri et al. (2020) WEviRC — Lift filter cho AC.
        // Composite formula: w(r) = confidence(r) × Lift(r, c) = conf² × N / classFreq[c]
        //
        // H5b Adaptive: chỉ enable nếu imbalance ratio ≥ threshold.
        // Lý do: lift có class-size bias, hại trên mild balanced data (1.5-2:1).
        // Default threshold = 0 (luôn enable). Set 3.0 để tránh regression trên balanced.
        if (useConfLiftVoting) {
            if (adaptiveConfLiftThreshold > 0) {
                // Check imbalance ratio — ưu tiên originalImbalanceRatio (pre-SMOTE)
                double ratio;
                if (originalImbalanceRatio > 0) {
                    ratio = originalImbalanceRatio;
                } else {
                    int maxFreq = 0, minFreq = Integer.MAX_VALUE;
                    for (int f : classFreq.values()) {
                        if (f > maxFreq) maxFreq = f;
                        if (f > 0 && f < minFreq) minFreq = f;
                    }
                    ratio = (minFreq > 0) ? (double) maxFreq / minFreq : Double.MAX_VALUE;
                }
                if (ratio < adaptiveConfLiftThreshold) {
                    return classifyByWeightedChi2(byClass);
                }
            }
            // H5b: conf × Lift composite voting (Brin 1997 + Bahri 2020)
            // (AV MCWCAR 2024 cũng có sẵn — gọi classifyByAV nếu muốn alternative)
            return classifyByConfLift(byClass);
        }

        // --- Weighted chi-square (CMAR Li 2001 §4): default voting ---
        return classifyByWeightedChi2(byClass);
    }

    /**
     * H5e voting: Added Value (AV) — Ahmed et al. (2000), MCWCAR (Wu et al. 2024 EAAI).
     *
     * <p>Formula: AV(rule, class) = confidence(rule) - P(class)</p>
     *
     * <p>Đặc tính (so với conf×Lift):</p>
     * <ul>
     *   <li><b>Asymmetric</b>: AV(X→Y) ≠ AV(Y→X) → đúng bản chất rule</li>
     *   <li><b>Bounded</b>: AV ∈ [-1, 1] → không over-aggressive như conf×Lift</li>
     *   <li><b>Negative correlation</b>: AV &lt; 0 = anti-pattern (loại)</li>
     *   <li><b>Class-size bias MILD</b>: subtractive thay vì multiplicative</li>
     * </ul>
     *
     * <p>Score(c) = Σ_{top-K rules of c có AV &gt; 0} AV(rule, c)</p>
     *
     * <p>Reference: Wu, W., Wang, S., Liu, B., Shao, Y., Xie, W. (2024).
     * "A novel software defect prediction approach via weighted classification
     * based on association rule mining". Engineering Applications of Artificial
     * Intelligence 129, 107622.</p>
     */
    private String classifyByAV(Map<String, List<AssociationRule>> byClass) {
        double n = totalTransactions;
        String bestClass = defaultClass;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (Map.Entry<String, List<AssociationRule>> entry : byClass.entrySet()) {
            String cls = entry.getKey();
            int classCount = classFreq.getOrDefault(cls, 0);
            if (classCount <= 0) continue;
            double pClass = classCount / n;

            List<AssociationRule> classRules = entry.getValue();

            // Sort theo AV DESC để pick top-K
            final double pClassFinal = pClass;
            classRules.sort((r1, r2) -> {
                double av1 = r1.getConfidence() - pClassFinal;
                double av2 = r2.getConfidence() - pClassFinal;
                return Double.compare(av2, av1);
            });

            // Sum AV cho top-K rules có AV > 0 (positive correlation)
            int limit = (voteTopK > 0) ? Math.min(voteTopK, classRules.size())
                                       : classRules.size();
            double score = 0.0;
            for (int i = 0; i < limit; i++) {
                double av = classRules.get(i).getConfidence() - pClass;
                if (av > 0) score += av;  // positive correlation only
            }

            // Cost-sensitive boost: lớp hiếm (classCount nhỏ) được nhân hệ số lớn
            // → tăng recall/F1 minority (King & Zeng 2001 threshold-moving).
            if (costSensitiveBeta > 0 && classCount > 0) {
                score *= Math.pow(n / classCount, costSensitiveBeta);
            }

            // Tie-break tường minh để TÁI LẬP 100%: khi điểm bằng nhau, ưu tiên
            // lớp có classFreq cao hơn (ổn định, không phụ thuộc thứ tự Map).
            if (score > bestScore
                    || (score == bestScore
                        && classCount > classFreq.getOrDefault(bestClass, 0))) {
                bestScore = score;
                bestClass = cls;
            }
        }
        return bestClass;
    }

    /**
     * Weighted chi-square voting (CMAR Li 2001 §4 gốc).
     * score(c) = Σ [χ²(r)]² / maxχ²(r)
     * H6: nếu voteTopK > 0, chỉ sum top-K rules theo χ² strongest per class.
     */
    private String classifyByWeightedChi2(Map<String, List<AssociationRule>> byClass) {
        String bestClass = defaultClass;
        double bestScore = -1.0;

        for (Map.Entry<String, List<AssociationRule>> entry : byClass.entrySet()) {
            String cls = entry.getKey();
            List<AssociationRule> classRules = entry.getValue();

            // Pre-compute weighted χ² cho mỗi rule
            double[] weightedChi2 = new double[classRules.size()];
            for (int i = 0; i < classRules.size(); i++) {
                AssociationRule rule = classRules.get(i);
                double chi2    = computeChiSquare(rule, cls);
                double maxChi2 = computeMaxChiSquare(rule, cls);
                weightedChi2[i] = (maxChi2 > 0) ? (chi2 * chi2) / maxChi2 : 0.0;
            }

            double score = 0.0;
            if (voteTopK > 0 && classRules.size() > voteTopK) {
                Integer[] idx = new Integer[classRules.size()];
                for (int i = 0; i < idx.length; i++) idx[i] = i;
                java.util.Arrays.sort(idx, (a, b) -> Double.compare(weightedChi2[b], weightedChi2[a]));
                int limit = Math.min(voteTopK, idx.length);
                for (int i = 0; i < limit; i++) score += weightedChi2[idx[i]];
            } else {
                for (double w : weightedChi2) score += w;
            }

            if (score > bestScore) {
                bestScore = score;
                bestClass = cls;
            }
        }
        return bestClass;
    }

    /**
     * H5b voting: composite weight = confidence × Lift, top-K rules per class.
     *
     * <p>Formula: score(c) = Σ_{top-K rules of c} conf(r) × lift(r, c)
     *           = Σ_{top-K} conf² × N / classFreq[c]</p>
     *
     * <p><b>Tại sao formula này</b>:
     * <ul>
     *   <li>Confidence: rule prediction accuracy (Li-Han-Pei 2001 CMAR)</li>
     *   <li>Lift: positive correlation strength (Brin 1997 SIGMOD)</li>
     *   <li>Tích: chỉ rule "vừa accurate vừa correlated" mới quan trọng</li>
     * </ul></p>
     *
     * <p><b>Lưu ý</b>: Lift có inherent class-size bias (lift ↑ khi class ↓).
     * Nên dùng SAU SMOTE/Borderline đã balance classes — bias giảm đáng kể.</p>
     */
    private String classifyByConfLift(Map<String, List<AssociationRule>> byClass) {
        double n = totalTransactions;
        String bestClass = defaultClass;
        double bestScore = -1.0;

        for (Map.Entry<String, List<AssociationRule>> entry : byClass.entrySet()) {
            String cls = entry.getKey();
            int classCount = classFreq.getOrDefault(cls, 0);
            if (classCount <= 0) continue;  // class chưa từng thấy trong training

            List<AssociationRule> classRules = entry.getValue();

            // Sort theo conf × lift DESC để pick top-K
            classRules.sort((r1, r2) -> {
                double s1 = r1.getConfidence() * r1.getConfidence() * n / classCount;
                double s2 = r2.getConfidence() * r2.getConfidence() * n / classCount;
                return Double.compare(s2, s1);
            });

            // Sum conf × lift cho top-K rules (chỉ rules có lift ≥ 1)
            int limit = (voteTopK > 0) ? Math.min(voteTopK, classRules.size())
                                       : classRules.size();
            double score = 0.0;
            for (int i = 0; i < limit; i++) {
                AssociationRule rule = classRules.get(i);
                double conf = rule.getConfidence();
                double lift = conf * n / classCount;
                if (lift >= 1.0) {
                    score += conf * lift;
                }
            }

            if (score > bestScore) {
                bestScore = score;
                bestClass = cls;
            }
        }
        return bestClass;
    }

    // -----------------------------------------------------------------------
    // Tính chi-square (dùng chung cho cắt tỉa và phân lớp)
    // -----------------------------------------------------------------------

    protected double computeChiSquare(AssociationRule rule, String cls) {
        double n = totalTransactions;
        double a = rule.getSupportCount();
        double b = rule.getCondsetSupportCount() - a;
        double c = classFreq.getOrDefault(cls, 0) - a;
        double d = n - a - b - c;

        double denom = (a + b) * (c + d) * (a + c) * (b + d);
        if (denom == 0) return 0.0;
        return n * Math.pow(a * d - b * c, 2) / denom;
    }

    protected double computeMaxChiSquare(AssociationRule rule, String cls) {
        double n    = totalTransactions;
        double supP = rule.getCondsetSupportCount();
        double supC = classFreq.getOrDefault(cls, 0);

        double a = Math.min(supP, supC);
        double b = supP - a;
        double c = supC - a;
        double d = n - a - b - c;

        double denom = (a + b) * (c + d) * (a + c) * (b + d);
        if (denom == 0) return 0.0;
        return n * Math.pow(a * d - b * c, 2) / denom;
    }

    // -----------------------------------------------------------------------
    // Dự đoán hàng loạt & đánh giá
    // -----------------------------------------------------------------------

    public List<String> predict(List<Transaction> testData) {
        List<String> predictions = new ArrayList<>();
        for (Transaction t : testData) predictions.add(classify(t));
        return predictions;
    }

    public double evaluate(List<Transaction> testData) {
        if (testData.isEmpty()) return 0.0;
        int correct = 0;
        for (Transaction t : testData) {
            if (classify(t).equals(t.getClassLabel())) correct++;
        }
        return (double) correct / testData.size();
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    public List<AssociationRule> getRules()    { return rules; }
    public CRTree                getCRTree()   { return crTree; }
    public String                getDefaultClass()          { return defaultClass; }
    public int                   getCandidateCount()        { return candidateCount; }
    public int                   getAfterGeneralPruneCount(){ return afterGeneralPruneCount; }
    public int                   getAfterChiPruneCount()    { return afterChiPruneCount; }
    public int                   getAfterCoveragePruneCount(){ return afterCoveragePruneCount; }
    public double                getChiSquareThreshold()    { return chiSquareThreshold; }
    public int                   getCoverageThreshold()     { return coverageThreshold; }
    public Map<String, Integer>  getClassFreq()             { return classFreq; }
    public int                   getTotalTransactions()     { return totalTransactions; }
}
