import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * BenchmarkFinal — Main pipeline trên 19 UCI vs paper CMAR 2001.
 *
 * <p>Mục đích: comprehensive comparison cho thesis defense.</p>
 *
 * <p>2 variants × 19 UCI datasets × 10-fold stratified CV:</p>
 * <ol>
 *   <li><b>Baseline</b> — CMAR gốc Li-Han-Pei 2001</li>
 *   <li><b>+L1 (Adaptive Borderline-SMOTE)</b> — data-level balancing, no H2</li>
 *   <li><b>+L2 (H4)</b> — stratified coverage</li>
 *   <li><b>+L3 (H5b conf×Lift)</b> — composite voting ⭐ ĐÓNG GÓP MỚI</li>
 * </ol>
 *
 * <p>Mỗi layer target metric khác:</p>
 * <ul>
 *   <li>L1 → F1, Recall (cân bằng minority)</li>
 *   <li>L2 → marginal refinement</li>
 *   <li>L3 → Accuracy (voting tốt hơn)</li>
 * </ul>
 */
public class BenchmarkFinal {

    static final int    TOP_K           = 10;
    static final int    SMOTE_TRIGGER   = 5;
    static final double SMOTE_RATIO     = 1.0;

    /** 19 UCI standard từ paper CMAR Table 1, kèm Acc paper để so. */
    static final String[][] DATASETS = {
        // { file, name, minSupportPct, paperCMAR, [maxPatternLength] }
        { "data_clean/breast-w.csv",    "breast-w",    "0.02", "96.42" },
        { "data_clean/cleve.csv",       "cleve",       "0.02", "82.18" },
        { "data_clean/crx.csv",         "crx",         "0.04", "85.36" },
        { "data_clean/diabetes.csv",    "diabetes",    "0.03", "75.81" },
        { "data_clean/german.csv",      "german",      "0.06", "73.40" },
        { "data_clean/glass.csv",       "glass",       "0.01", "70.09" },
        { "data_clean/heart.csv",       "heart",       "0.03", "82.59" },
        { "data_clean/hepatitis.csv",   "hepatitis",   "0.05", "80.65" },
        { "data_clean/horse.csv",       "horse",       "0.03", "82.61" },
        { "data_clean/iris.csv",        "iris",        "0.03", "94.00" },
        { "data_clean/labor.csv",       "labor",       "0.05", "89.47" },
        { "data_clean/led7.csv",        "led7",        "0.03", "71.90" },
        { "data_clean/lymph.csv",       "lymph",       "0.05", "82.43" },
        { "data_clean/sonar.csv",       "sonar",       "0.05", "79.33", "5" },
        { "data_clean/tic-tac-toe.csv", "tic-tac-toe", "0.003","99.27" },
        { "data_clean/vehicle.csv",     "vehicle",     "0.03", "68.68", "5" },
        { "data_clean/waveform.csv",    "waveform",    "0.01", "80.17", "5" },
        { "data_clean/wine.csv",        "wine",        "0.03", "95.51" },
        { "data_clean/zoo.csv",         "zoo",         "0.03", "96.04", "4" },
    };

    public static void main(String[] args) throws Exception {
        int K_FOLD = 10;
        double minConfidence = 0.5;
        double chiSqThreshold = 3.841;
        int coverageDelta = 4;
        long seed = 42;

        System.out.println("=================================================================");
        System.out.println("  BenchmarkFinal — 3-Layer Pipeline (No H2) vs Paper CMAR 2001");
        System.out.println("=================================================================");
        System.out.println("  2 variants × 19 UCI datasets × 10-fold stratified CV");
        System.out.println("  L1=Adaptive Borderline-SMOTE (No H2)   L2=H4   L3=H5b conf×Lift voting");
        System.out.println("=================================================================\n");

        System.out.println(">>> 1/2 BASELINE (CMAR Li 2001) ...");
        Map<String, EvalMetrics> baseline = runVariant(
            false, false, false,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        System.out.println("\n>>> 2/2 NoH2+H4+H5b (FULL — 3-Layer Pipeline) ...");
        Map<String, EvalMetrics> full = runVariant(
            true, true, true,
            K_FOLD, minConfidence, chiSqThreshold, coverageDelta, seed);

        // -------- Summary --------
        System.out.println("\n" + "=".repeat(120));
        System.out.println("KET QUA — 19 UCI datasets, 3-Layer Pipeline vs Baseline vs Paper");
        System.out.println("=".repeat(120));
        System.out.printf("%-13s | Baseline(Acc/F1/R)      | Full(Acc/F1/R)          | Paper | dPaper  | Best%n",
            "Dataset");
        System.out.println("-".repeat(120));

        double sumBaseAcc=0, sumFullAcc=0, sumBaseF1=0, sumFullF1=0, sumBaseRec=0, sumFullRec=0;
        double sumPaperAcc = 0;
        int cnt = 0;
        int accImproved = 0, f1Improved = 0, recImproved = 0;
        int beatsPaper = 0, within5pctPaper = 0;

        for (String[] ds : DATASETS) {
            String name = ds[1];
            double paperAcc = Double.parseDouble(ds[3]);
            EvalMetrics b = baseline.get(name);
            EvalMetrics f = full.get(name);
            if (b == null || f == null) continue;

            double bRec = macroRecall(b);
            double fRec = macroRecall(f);

            double diffPaper = f.accuracy * 100 - paperAcc;
            if (diffPaper > 0) beatsPaper++;
            if (Math.abs(diffPaper) <= 5.0) within5pctPaper++;
            if (f.accuracy > b.accuracy + 0.001) accImproved++;
            if (f.macroF1 > b.macroF1 + 0.001) f1Improved++;
            if (fRec > bRec + 0.001) recImproved++;

            sumBaseAcc += b.accuracy; sumFullAcc += f.accuracy;
            sumBaseF1 += b.macroF1; sumFullF1 += f.macroF1;
            sumBaseRec += bRec; sumFullRec += fRec;
            sumPaperAcc += paperAcc;
            cnt++;

            String marker = diffPaper > 0 ? "⭐" : (diffPaper > -2 ? "≈" : "");
            System.out.printf("%-13s | %.3f/%.3f/%.3f      | %.3f/%.3f/%.3f      | %5.2f | %+6.2f%% | %s%n",
                name,
                b.accuracy, b.macroF1, bRec,
                f.accuracy, f.macroF1, fRec,
                paperAcc, diffPaper, marker);
        }

        System.out.println("-".repeat(120));
        System.out.printf("AVG (n=%d)    | %.3f/%.3f/%.3f      | %.3f/%.3f/%.3f      | %5.2f | %+6.2f%% |%n",
            cnt,
            sumBaseAcc / cnt, sumBaseF1 / cnt, sumBaseRec / cnt,
            sumFullAcc / cnt, sumFullF1 / cnt, sumFullRec / cnt,
            sumPaperAcc / cnt, (sumFullAcc * 100 / cnt) - sumPaperAcc / cnt);

        System.out.println("\n" + "=".repeat(70));
        System.out.println("FULL (3-Layer) vs BASELINE — Improvement statistics");
        System.out.println("=".repeat(70));
        System.out.printf("  Acc cải thiện:     %d / %d datasets (%.0f%%)%n",
            accImproved, cnt, 100.0 * accImproved / cnt);
        System.out.printf("  F1 cải thiện:      %d / %d datasets (%.0f%%)%n",
            f1Improved, cnt, 100.0 * f1Improved / cnt);
        System.out.printf("  Recall cải thiện:  %d / %d datasets (%.0f%%)%n",
            recImproved, cnt, 100.0 * recImproved / cnt);

        System.out.println("\n" + "=".repeat(70));
        System.out.println("FULL vs PAPER CMAR 2001");
        System.out.println("=".repeat(70));
        System.out.printf("  Beats paper:           %d / %d datasets (%.0f%%)%n",
            beatsPaper, cnt, 100.0 * beatsPaper / cnt);
        System.out.printf("  Within 5%% paper:       %d / %d datasets (%.0f%%)%n",
            within5pctPaper, cnt, 100.0 * within5pctPaper / cnt);
        System.out.printf("  AVG Acc gap vs paper:  %+.2f%%%n",
            (sumFullAcc * 100 / cnt) - sumPaperAcc / cnt);

        System.out.println("\n" + "=".repeat(70));
        System.out.println("METRIC SUMMARY (Baseline → Full)");
        System.out.println("=".repeat(70));
        System.out.printf("  Accuracy:  %.4f → %.4f  (Δ=%+.4f, %+.2f%%)%n",
            sumBaseAcc / cnt, sumFullAcc / cnt,
            (sumFullAcc - sumBaseAcc) / cnt,
            100 * (sumFullAcc - sumBaseAcc) / sumBaseAcc);
        System.out.printf("  MacroF1:   %.4f → %.4f  (Δ=%+.4f, %+.2f%%)%n",
            sumBaseF1 / cnt, sumFullF1 / cnt,
            (sumFullF1 - sumBaseF1) / cnt,
            100 * (sumFullF1 - sumBaseF1) / sumBaseF1);
        System.out.printf("  MacroRec:  %.4f → %.4f  (Δ=%+.4f, %+.2f%%)%n",
            sumBaseRec / cnt, sumFullRec / cnt,
            (sumFullRec - sumBaseRec) / cnt,
            100 * (sumFullRec - sumBaseRec) / sumBaseRec);

        writeCsvs(baseline, "final_baseline");
        writeCsvs(full,     "final_full");
        System.out.println("\nFile CSV: result/final_{baseline,full}_*.csv");
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

    private static Map<String, EvalMetrics> runVariant(
            boolean useAdaptiveSmote,
            boolean useL2H4,
            boolean useL3H5bConfLift,
            int K_FOLD, double minConfidence, double chiSqThreshold,
            int coverageDelta, long seed) throws Exception {

        Map<String, EvalMetrics> results = new LinkedHashMap<>();

        // Container để pass original imbalance ratio per dataset
        final double[] currentRatio = {0.0};

        Supplier<CMARClassifier> factory = () -> {
            CMARClassifier c = new CMARClassifier();
            if (useL2H4) c.setStratifiedTopK(TOP_K);
            if (useL3H5bConfLift) {
                // H5b ADAPTIVE: chỉ enable conf×Lift khi original imbalance ratio ≥ 3.0
                c.setUseConfLiftVoting(true);
                c.setAdaptiveConfLiftThreshold(3.0);
                c.setOriginalImbalanceRatio(currentRatio[0]);
                c.setVoteTopK(TOP_K);
            }
            return c;
        };

        for (String[] ds : DATASETS) {
            String file = ds[0];
            String name = ds[1];
            double supPct = Double.parseDouble(ds[2]);
            int maxLen = ds.length > 4 ? Integer.parseInt(ds[4]) : Integer.MAX_VALUE;
            System.out.print("  " + String.format("%-13s", name) + " ... ");
            try {
                List<Transaction> data = DatasetLoader.load(file);
                if (data.isEmpty()) { System.out.println("SKIP"); continue; }

                // Compute ORIGINAL imbalance ratio TRƯỚC SMOTE (cho adaptive H5b)
                currentRatio[0] = computeImbalanceRatio(data);

                double smoteRatio = 0.0;
                boolean useBorderline = false;
                if (useAdaptiveSmote) {
                    int minFreq = computeMinClassFreq(data);
                    if (minFreq < SMOTE_TRIGGER) {
                        smoteRatio = SMOTE_RATIO;
                        useBorderline = true;
                    }
                }
                List<EvalMetrics> foldMetrics = CrossValidator.runWithMetrics(
                    data, K_FOLD, supPct, minConfidence, chiSqThreshold, coverageDelta,
                    seed, maxLen,
                    factory,
                    smoteRatio, useBorderline);

                EvalMetrics agg = EvalMetrics.average(foldMetrics);
                results.put(name, agg);
                System.out.printf("Acc=%.4f F1=%.4f%n", agg.accuracy, agg.macroF1);
            } catch (Exception e) {
                throw new RuntimeException("BenchmarkFinal failed on dataset " + name, e);
            } catch (OutOfMemoryError e) {
                throw e;
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
}
