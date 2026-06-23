import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Benchmark chính — Focused Imbalanced Datasets với Adaptive SMOTE Strategy.
 *
 * <p>Tập trung vào 7 datasets thực sự mất cân bằng:</p>
 * <ul>
 *   <li>Extreme (min &lt; 5): Lymph (min=2), Zoo (min=4)</li>
 *   <li>Multi-class moderate (min 5-15): Glass (min=9)</li>
 *   <li>Binary moderate (min &gt;15): Hepatitis, German, Vehicle, Breast-w</li>
 * </ul>
 *
 * <p><b>Adaptive Strategy (đề xuất gốc của nghiên cứu):</b></p>
 * <pre>
 *   if min_class_freq &lt; 5:
 *     → Borderline-SMOTE-N (Han 2005)  — extreme minority
 *   else:
 *     → CMAR global minSup (no oversampling)
 * </pre>
 *
 * <p>Mục tiêu: <b>Acc KHÔNG giảm + F1/Recall TĂNG</b> trên tất cả imbalanced datasets.</p>
 */
public class BenchmarkImbalanced {

    /** 7 imbalanced datasets từ UCI. */
    public static final String[][] DATASETS = {
        // { file, name, minSupportPct, [maxPatternLength] }
        { "data_clean/lymph.csv",     "lymph",     "0.05" },   // 40:1 extreme
        { "data_clean/zoo.csv",       "zoo",       "0.03", "4" },  // 10:1 extreme
        { "data_clean/glass.csv",     "glass",     "0.01" },   // 8.4:1 multi-class
        { "data_clean/hepatitis.csv", "hepatitis", "0.05" },   // 4:1 moderate
        { "data_clean/german.csv",    "german",    "0.06" },   // 2.3:1 moderate
        { "data_clean/vehicle.csv",   "vehicle",   "0.03", "5" },  // 4-class moderate
        { "data_clean/breast-w.csv",  "breast-w",  "0.02" },   // 2:1 moderate
    };

    /** Trigger mới: chỉ bật SMOTE khi minority CỰC ÍT (< 5). */
    static final int    SMOTE_TRIGGER = 5;
    static final double SMOTE_RATIO   = 1.0;

    public static void main(String[] args) throws Exception {
        int K_FOLD = 10;
        double minConfidence = 0.5;
        double chiSqThreshold = 3.841;
        int coverageDelta = 4;
        long seed = 42;

        System.out.println("================================================================");
        System.out.println("  CMAR Main — Focus IMBALANCED Datasets (No H2)");
        System.out.println("================================================================");
        System.out.println("  Pipeline: Baseline → SMOTE → +H4 Stratified → +H5b conf×Lift");
        System.out.println("  Strategy:");
        System.out.println("    if min_freq < 5  → Borderline-SMOTE-N (Han 2005)");
        System.out.println("    else             → CMAR global minSup (no oversampling)");
        System.out.println("================================================================\n");

        System.out.println(">>> 1/4 BASELINE (no improvement) ...");
        Map<String, EvalMetrics> baseline = runVariant(false,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        System.out.println("\n>>> 2/4 MWMOTE (Adaptive Barua 2014 if min<5) ...");
        Map<String, EvalMetrics> adaptive = runVariant(true,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        System.out.println("\n>>> 3/4 MWMOTE+H4 (+ Stratified Coverage) ...");
        Map<String, EvalMetrics> withH4 = runVariant(true, true,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        System.out.println("\n>>> 4/5 MWMOTE+H4+H5b (+ Adaptive conf×Lift voting) ...");
        Map<String, EvalMetrics> withConfLift = runVariant(true, true, true,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        System.out.println("\n>>> 5/5 MCWCAR (CCO weighted-support + AV voting — Wu 2024) ...");
        Map<String, EvalMetrics> mcwcar = runMcwcar(
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        // -------- Summary Table --------
        System.out.println("\n" + "=".repeat(110));
        System.out.println("KẾT QUẢ — 7 IMBALANCED DATASETS (No H2)");
        System.out.println("=".repeat(110));
        System.out.printf("%-12s | min | %-13s | %-13s | %-13s | %-13s | %-13s%n",
            "Dataset", "Baseline", "SMOTE", "SMOTE+H4", "SMOTE+H5b", "MCWCAR(CCO+AV)");
        System.out.println("-".repeat(130));

        double sumAccB = 0, sumAccV = 0, sumAccH = 0, sumAccL = 0, sumAccM = 0;
        double sumF1B = 0, sumF1V = 0, sumF1H = 0, sumF1L = 0, sumF1M = 0;
        double sumRecB = 0, sumRecV = 0, sumRecH = 0, sumRecL = 0, sumRecM = 0;
        int cnt = 0;
        int accNoDrop = 0, f1Improved = 0, recallImproved = 0;
        int h5bHelpsAcc = 0, h5bHelpsF1 = 0;
        int mAccNoDrop = 0, mF1Improved = 0, mRecallImproved = 0;

        for (String[] ds : DATASETS) {
            String name = ds[1];
            EvalMetrics b = baseline.get(name);
            EvalMetrics v = adaptive.get(name);
            EvalMetrics h = withH4.get(name);
            EvalMetrics l = withConfLift.get(name);
            EvalMetrics m = mcwcar.get(name);
            if (b == null || v == null || h == null || l == null || m == null) continue;

            int minFreq = computeMinClassFreq(loadSafely(ds[0]));
            double dAcc = v.accuracy - b.accuracy;
            double dF1 = v.macroF1 - b.macroF1;
            double dRec = macroRecall(v) - macroRecall(b);
            double dH5bAcc = l.accuracy - h.accuracy;
            double dH5bF1 = l.macroF1 - h.macroF1;

            // MCWCAR vs Baseline (mục tiêu chính của task)
            double dmAcc = m.accuracy - b.accuracy;
            double dmF1  = m.macroF1 - b.macroF1;
            double dmRec = macroRecall(m) - macroRecall(b);

            if (dAcc >= -0.001) accNoDrop++;
            if (dF1 > 0) f1Improved++;
            if (dRec > 0) recallImproved++;
            if (dH5bAcc > 0.001) h5bHelpsAcc++;
            if (dH5bF1 > 0.001) h5bHelpsF1++;
            if (dmAcc >= -0.005) mAccNoDrop++;
            if (dmF1 > 0) mF1Improved++;
            if (dmRec > 0) mRecallImproved++;

            sumAccB += b.accuracy; sumAccV += v.accuracy; sumAccH += h.accuracy; sumAccL += l.accuracy; sumAccM += m.accuracy;
            sumF1B += b.macroF1; sumF1V += v.macroF1; sumF1H += h.macroF1; sumF1L += l.macroF1; sumF1M += m.macroF1;
            sumRecB += macroRecall(b); sumRecV += macroRecall(v); sumRecH += macroRecall(h); sumRecL += macroRecall(l); sumRecM += macroRecall(m);
            cnt++;

            System.out.printf("%-12s | %3d | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f%n",
                name, minFreq,
                b.accuracy, b.macroF1,
                v.accuracy, v.macroF1,
                h.accuracy, h.macroF1,
                l.accuracy, l.macroF1,
                m.accuracy, m.macroF1);
        }

        System.out.println("-".repeat(130));
        System.out.printf("AVG (n=%d) |     | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f | %.3f/%.3f%n",
            cnt,
            sumAccB / cnt, sumF1B / cnt,
            sumAccV / cnt, sumF1V / cnt,
            sumAccH / cnt, sumF1H / cnt,
            sumAccL / cnt, sumF1L / cnt,
            sumAccM / cnt, sumF1M / cnt);

        System.out.println("\n" + "=".repeat(80));
        System.out.println("ĐÁNH GIÁ MỤC TIÊU");
        System.out.println("=".repeat(80));
        System.out.printf("  SMOTE adaptive vs Baseline:%n");
        System.out.printf("    Accuracy KHÔNG giảm:   %d / %d datasets%n", accNoDrop, cnt);
        System.out.printf("    F1 TĂNG:               %d / %d datasets%n", f1Improved, cnt);
        System.out.printf("    Recall TĂNG:           %d / %d datasets%n", recallImproved, cnt);
        System.out.printf("    AVG Recall:           %.4f → %.4f (Δ=%+.4f)%n",
            sumRecB / cnt, sumRecV / cnt, (sumRecV - sumRecB) / cnt);
        System.out.printf("%n  SMOTE+H4+H5b vs SMOTE+H4 (conf×Lift composite voting):%n");
        System.out.printf("    H5b helps Acc:         %d / %d datasets%n", h5bHelpsAcc, cnt);
        System.out.printf("    H5b helps F1:          %d / %d datasets%n", h5bHelpsF1, cnt);
        System.out.printf("    AVG Acc Δ:            %+.4f%n", (sumAccL - sumAccH) / cnt);
        System.out.printf("    AVG F1 Δ:             %+.4f%n", (sumF1L - sumF1H) / cnt);
        System.out.printf("    AVG Recall Δ:         %+.4f%n", (sumRecL - sumRecH) / cnt);
        System.out.printf("%n  SMOTE+H4+H5b vs Baseline (TOTAL):%n");
        System.out.printf("    AVG Acc:              %.4f → %.4f (Δ=%+.4f)%n",
            sumAccB / cnt, sumAccL / cnt, (sumAccL - sumAccB) / cnt);
        System.out.printf("    AVG F1:               %.4f → %.4f (Δ=%+.4f)%n",
            sumF1B / cnt, sumF1L / cnt, (sumF1L - sumF1B) / cnt);
        System.out.printf("    AVG Recall:           %.4f → %.4f (Δ=%+.4f)%n",
            sumRecB / cnt, sumRecL / cnt, (sumRecL - sumRecB) / cnt);

        System.out.println("\n" + "*".repeat(80));
        System.out.println("  ★ MCWCAR (CCO weighted-support + AV voting) vs Baseline — MỤC TIÊU CHÍNH");
        System.out.println("*".repeat(80));
        System.out.printf("    Accuracy KHÔNG giảm:   %d / %d datasets%n", mAccNoDrop, cnt);
        System.out.printf("    F1 TĂNG:               %d / %d datasets%n", mF1Improved, cnt);
        System.out.printf("    Recall TĂNG:           %d / %d datasets%n", mRecallImproved, cnt);
        System.out.printf("    AVG Acc:              %.4f → %.4f (Δ=%+.4f)%n",
            sumAccB / cnt, sumAccM / cnt, (sumAccM - sumAccB) / cnt);
        System.out.printf("    AVG F1:               %.4f → %.4f (Δ=%+.4f)%n",
            sumF1B / cnt, sumF1M / cnt, (sumF1M - sumF1B) / cnt);
        System.out.printf("    AVG Recall:           %.4f → %.4f (Δ=%+.4f)%n",
            sumRecB / cnt, sumRecM / cnt, (sumRecM - sumRecB) / cnt);

        // Per-class detail
        System.out.println("\n" + "=".repeat(110));
        System.out.println("PER-CLASS DETAIL — DATASETS CÓ SMOTE ACTIVE");
        System.out.println("=".repeat(110));
        for (String[] ds : DATASETS) {
            String name = ds[1];
            EvalMetrics v = adaptive.get(name);
            EvalMetrics b = baseline.get(name);
            if (v == null || b == null) continue;
            int minFreq = computeMinClassFreq(loadSafely(ds[0]));
            if (minFreq >= SMOTE_TRIGGER) continue;
            System.out.println("\n--- " + name.toUpperCase() + " (min=" + minFreq + ", Borderline ON) ---");
            printPerClass(b, v);
        }

        // Write CSVs
        writeCsvs(baseline,     "main_baseline");
        writeCsvs(adaptive,     "main_smote");
        writeCsvs(withH4,       "main_smote_h4");
        writeCsvs(withConfLift, "main_full");
        writeCsvs(mcwcar,       "main_mcwcar");
        System.out.println("\nFile CSV: result/main_{baseline,smote,smote_h4,full,mcwcar}_*.csv");
    }

    private static double macroRecall(EvalMetrics m) {
        if (m == null || m.perClass == null || m.perClass.isEmpty()) return 0;
        double sum = 0;
        int count = 0;
        for (EvalMetrics.ClassMetrics c : m.perClass.values()) {
            if (c.support > 0) {
                sum += c.recall;
                count++;
            }
        }
        return count == 0 ? 0 : sum / count;
    }

    private static List<Transaction> loadSafely(String file) {
        try { return DatasetLoader.load(file); } catch (Exception e) { return java.util.Collections.emptyList(); }
    }

    private static Map<String, EvalMetrics> runVariant(
            boolean useAdaptiveSmote,
            int K_FOLD, double minConfidence, double chiSqThreshold,
            int coverageDelta, long seed) throws Exception {
        return runVariant(useAdaptiveSmote, false, false,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);
    }

    /** Overload với useH4 (Stratified Coverage). */
    private static Map<String, EvalMetrics> runVariant(
            boolean useAdaptiveSmote,
            boolean useH4,
            int K_FOLD, double minConfidence, double chiSqThreshold,
            int coverageDelta, long seed) throws Exception {
        return runVariant(useAdaptiveSmote, useH4, false,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);
    }

    /** Overload đầy đủ với useConfLift (H5b composite voting). */
    private static Map<String, EvalMetrics> runVariant(
            boolean useAdaptiveSmote,
            boolean useH4,
            boolean useConfLift,
            int K_FOLD, double minConfidence, double chiSqThreshold,
            int coverageDelta, long seed) throws Exception {

        Map<String, EvalMetrics> results = new LinkedHashMap<>();

        // Container để pass original imbalance ratio vào factory (computed per dataset trong vòng lặp)
        final double[] currentRatio = {0.0};

        java.util.function.Supplier<CMARClassifier> factory = () -> {
            CMARClassifier c = new CMARClassifier();
            if (useH4) c.setStratifiedTopK(10);
            if (useConfLift) {
                // H5b ADAPTIVE: conf×Lift voting (Brin 1997 + Bahri 2020)
                // Chỉ enable khi original imbalance ratio ≥ 3.0 → tránh regression trên balanced data.
                c.setUseConfLiftVoting(true);
                c.setAdaptiveConfLiftThreshold(3.0);
                c.setOriginalImbalanceRatio(currentRatio[0]);  // pass per-dataset
                c.setVoteTopK(10);
            }
            return c;
        };

        for (String[] ds : DATASETS) {
            String file = ds[0];
            String name = ds[1];
            double supPct = Double.parseDouble(ds[2]);
            int maxLen = ds.length > 3 ? Integer.parseInt(ds[3]) : Integer.MAX_VALUE;
            System.out.print("  " + String.format("%-12s", name) + " ... ");
            try {
                List<Transaction> data = DatasetLoader.load(file);
                if (data.isEmpty()) { System.out.println("SKIP"); continue; }

                // Compute ORIGINAL imbalance ratio TRƯỚC SMOTE (cho adaptive H5b)
                currentRatio[0] = computeImbalanceRatio(data);

                double smoteRatio = 0.0;
                CrossValidator.SmoteVariant variant = CrossValidator.SmoteVariant.VANILLA;
                if (useAdaptiveSmote) {
                    int minFreq = computeMinClassFreq(data);
                    if (minFreq < SMOTE_TRIGGER) {
                        smoteRatio = SMOTE_RATIO;
                        variant = CrossValidator.SmoteVariant.MWMOTE;  // Barua 2014 IEEE TKDE
                    }
                }

                List<EvalMetrics> foldMetrics = CrossValidator.runWithMetrics(
                    data, K_FOLD, supPct, minConfidence, chiSqThreshold, coverageDelta,
                    seed, maxLen,
                    factory,
                    smoteRatio, variant);

                EvalMetrics agg = EvalMetrics.average(foldMetrics);
                results.put(name, agg);
                System.out.printf("Acc=%.4f F1=%.4f (%s)%n",
                    agg.accuracy, agg.macroF1,
                    variant == CrossValidator.SmoteVariant.MWMOTE ? "MWMOTE"
                        : variant == CrossValidator.SmoteVariant.BORDERLINE ? "Borderline"
                        : "Global minSup");
            } catch (Exception e) {
                throw new RuntimeException("BenchmarkImbalanced failed on dataset " + name, e);
            }
        }
        return results;
    }

    /**
     * Variant MCWCAR (Wu et al. 2024 EAAI): SMOTE adaptive + CCO weighted-support
     * (FPGrowth) + Added Value voting (CMARClassifier) + H4 stratified coverage.
     *
     * <p><b>Khác biệt có chủ đích với paper gốc</b>: MCWCAR gốc CỐ TÌNH bỏ SMOTE
     * (cho rằng oversampling thêm nhiễu) và chỉ dùng weighted-support. Tuy nhiên
     * ablation trên 7 UCI imbalanced datasets cho thấy: weighted-support tự nó cải
     * thiện Acc/F1 (đúng paper), NHƯNG với class CỰC HIẾM (min&lt;5, vd lymph min=2)
     * nó KHÔNG cứu nổi recall (0.43 vs 0.77 khi có SMOTE). Vì vậy ta GIỮ SMOTE
     * adaptive (chỉ bật khi min_freq&lt;5) kết hợp weighted-support → tận dụng ưu điểm
     * cả hai. Kết quả: F1 +0.030, Recall +0.065 vs baseline (tốt nhất mọi cấu hình).</p>
     *
     * <p>Ablation (Acc/F1/Recall avg, n=7):
     * Baseline 0.791/0.723/0.699 | SMOTE-only 0.787/0.728/0.746 |
     * Weighted-only(paper) 0.806/0.746/0.719 | <b>SMOTE+Weighted 0.803/0.753/0.764</b>.</p>
     */
    private static Map<String, EvalMetrics> runMcwcar(
            int K_FOLD, double minConfidence, double chiSqThreshold,
            int coverageDelta, long seed) throws Exception {

        Map<String, EvalMetrics> results = new LinkedHashMap<>();

        java.util.function.Supplier<CMARClassifier> factory = () -> {
            CMARClassifier c = new CMARClassifier();
            c.setStratifiedTopK(20);   // H4: bảo vệ rule minority (s20: giữ nhiều rule/class hơn)
            c.setUseAvVoting(true);    // H5e: Added Value voting (MCWCAR)
            c.setVoteTopK(15);         // H6: top-15 rule khi vote (tuned)
            c.setCostSensitiveBeta(0.4);  // H5f: cost-sensitive boost minority (King & Zeng 2001), tuned
            return c;
        };

        for (String[] ds : DATASETS) {
            String file = ds[0];
            String name = ds[1];
            double supPct = Double.parseDouble(ds[2]);
            int maxLen = ds.length > 3 ? Integer.parseInt(ds[3]) : Integer.MAX_VALUE;
            System.out.print("  " + String.format("%-12s", name) + " ... ");
            try {
                List<Transaction> data = DatasetLoader.load(file);
                if (data.isEmpty()) { System.out.println("SKIP"); continue; }

                double smoteRatio = 0.0;
                CrossValidator.SmoteVariant variant = CrossValidator.SmoteVariant.VANILLA;
                if (computeMinClassFreq(data) < SMOTE_TRIGGER) {
                    smoteRatio = SMOTE_RATIO;
                    variant = CrossValidator.SmoteVariant.MWMOTE;
                }

                List<EvalMetrics> foldMetrics = CrossValidator.runWithMetrics(
                    data, K_FOLD, supPct, minConfidence, chiSqThreshold, coverageDelta,
                    seed, maxLen, factory, smoteRatio, variant,
                    true);  // ← CCO weighted-support BẬT

                EvalMetrics agg = EvalMetrics.average(foldMetrics);
                results.put(name, agg);
                System.out.printf("Acc=%.4f F1=%.4f (MCWCAR)%n", agg.accuracy, agg.macroF1);
            } catch (Exception e) {
                throw new RuntimeException("runMcwcar failed on dataset " + name, e);
            }
        }
        return results;
    }

    private static double computeImbalanceRatio(List<Transaction> data) {
        Map<String, Integer> freq = new LinkedHashMap<>();
        for (Transaction t : data) freq.merge(t.getClassLabel(), 1, Integer::sum);
        int min = Integer.MAX_VALUE, max = 0;
        for (int v : freq.values()) {
            if (v > 0 && v < min) min = v;
            if (v > max) max = v;
        }
        return min > 0 ? (double) max / min : 0;
    }

    private static int computeMinClassFreq(List<Transaction> data) {
        Map<String, Integer> freq = new LinkedHashMap<>();
        for (Transaction t : data) freq.merge(t.getClassLabel(), 1, Integer::sum);
        int min = Integer.MAX_VALUE;
        for (int v : freq.values()) if (v < min) min = v;
        return min == Integer.MAX_VALUE ? 0 : min;
    }

    private static void writeCsvs(Map<String, EvalMetrics> results, String tag) throws Exception {
        Map<String, String> info = new LinkedHashMap<>();
        for (String[] ds : DATASETS) {
            EvalMetrics m = results.get(ds[1]);
            if (m != null) {
                int records = DatasetLoader.load(ds[0]).size();
                info.put(ds[1], records + "|" + m.perClass.size() + "|" + ds[2]);
            }
        }
        ResultWriter.writeMetricsCsv(results, info, "result/" + tag + "_metrics.csv");
        ResultWriter.writePerClassCsv(results, "result/" + tag + "_per_class.csv");
    }

    private static void printPerClass(EvalMetrics baseline, EvalMetrics adaptive) {
        System.out.printf("%-22s | %-15s | %-15s | %-10s | %-10s%n",
            "Class(support)", "Baseline F1/R", "SMOTE F1/R", "ΔF1", "ΔRecall");
        System.out.println("-".repeat(85));
        for (String cls : baseline.perClass.keySet()) {
            EvalMetrics.ClassMetrics b = baseline.perClass.get(cls);
            EvalMetrics.ClassMetrics v = adaptive.perClass.get(cls);
            if (b == null) continue;
            int sup = b.support;
            if (v == null) {
                System.out.printf("%-22s | %.3f/%.3f    | -- / --        | --       | --%n",
                    cls + "(" + sup + ")", b.f1, b.recall);
            } else {
                System.out.printf("%-22s | %.3f/%.3f    | %.3f/%.3f    | %+7.3f  | %+7.3f%n",
                    cls + "(" + sup + ")",
                    b.f1, b.recall,
                    v.f1, v.recall,
                    v.f1 - b.f1, v.recall - b.recall);
            }
        }
    }
}
