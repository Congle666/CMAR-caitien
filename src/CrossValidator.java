import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

/**
 * Kiểm chứng chéo K-fold có phân tầng (Stratified K-Fold CV) cho CMAR.
 *
 * <p>Đúng giao thức đánh giá của paper CMAR 2001:</p>
 * <ul>
 *   <li>10-fold stratified cross-validation</li>
 *   <li>Mỗi fold bảo toàn phân phối lớp</li>
 *   <li>Báo cáo accuracy + Macro-F1 + Weighted-F1 + per-class P/R/F1</li>
 * </ul>
 *
 * <p>Hỗ trợ cải tiến cho dữ liệu mất cân bằng (SMOTE / Borderline-SMOTE).</p>
 */
public class CrossValidator {

    /** Convenience: chỉ trả về accuracy array. */
    public static double[] run(List<Transaction> data, int k,
                                double minSupportPct, double minConfidence,
                                double chiSqThreshold, int coverageDelta,
                                long seed) {
        List<EvalMetrics> foldMetrics = runWithMetrics(
            data, k, minSupportPct, minConfidence,
            chiSqThreshold, coverageDelta, seed, Integer.MAX_VALUE);
        double[] accs = new double[foldMetrics.size()];
        for (int i = 0; i < foldMetrics.size(); i++) accs[i] = foldMetrics.get(i).accuracy;
        return accs;
    }

    /** Baseline CMAR (không SMOTE). */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength) {
        return runWithMetrics(data, k, minSupportPct, minConfidence,
            chiSqThreshold, coverageDelta, seed, maxPatternLength,
            CMARClassifier::new, 0.0, SmoteVariant.VANILLA);
    }

    /** SMOTE variant — chọn vanilla SMOTE-N, Borderline-SMOTE, hay MWMOTE-N. */
    public enum SmoteVariant { VANILLA, BORDERLINE, MWMOTE }

    /** Legacy boolean overload — useBorderline=true → BORDERLINE, false → VANILLA. */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength,
            Supplier<CMARClassifier> classifierFactory,
            double smoteTargetRatio,
            boolean useBorderlineSMOTE) {
        return runWithMetrics(data, k, minSupportPct, minConfidence,
            chiSqThreshold, coverageDelta, seed, maxPatternLength,
            classifierFactory, smoteTargetRatio,
            useBorderlineSMOTE ? SmoteVariant.BORDERLINE : SmoteVariant.VANILLA);
    }

    /**
     * Overload đầy đủ — bật/tắt từng cải tiến qua tham số.
     *
     * @param smoteTargetRatio  SMOTE — > 0 để bật. 1.0 = balance hoàn toàn. 0 = tắt.
     * @param smoteVariant      VANILLA (Chawla 2002), BORDERLINE (Han 2005), MWMOTE (Barua 2014).
     */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength,
            Supplier<CMARClassifier> classifierFactory,
            double smoteTargetRatio,
            SmoteVariant smoteVariant) {
        return runWithMetrics(data, k, minSupportPct, minConfidence, chiSqThreshold,
            coverageDelta, seed, maxPatternLength, classifierFactory,
            smoteTargetRatio, smoteVariant, false);
    }

    /** Overload với cờ MCWCAR CCO weighted-support (Wu et al. 2024). */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength,
            Supplier<CMARClassifier> classifierFactory,
            double smoteTargetRatio,
            SmoteVariant smoteVariant,
            boolean useCcoWeighting) {
        return runWithMetrics(data, k, minSupportPct, minConfidence, chiSqThreshold,
            coverageDelta, seed, maxPatternLength, classifierFactory,
            smoteTargetRatio, smoteVariant, useCcoWeighting, FPGrowth.WeightMode.CCO);
    }

    /** Overload đầy đủ với WeightMode (CCO / MI / HYBRID) — MCWCAR Wu et al. 2024. */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength,
            Supplier<CMARClassifier> classifierFactory,
            double smoteTargetRatio,
            SmoteVariant smoteVariant,
            boolean useCcoWeighting,
            FPGrowth.WeightMode weightMode) {
        return runWithMetrics(data, k, minSupportPct, minConfidence, chiSqThreshold,
            coverageDelta, seed, maxPatternLength, classifierFactory,
            smoteTargetRatio, smoteVariant, useCcoWeighting, weightMode, false);
    }

    /** Overload đầy đủ nhất — thêm cờ per-class minSup (MSApriori-lite, Phase 1). */
    public static List<EvalMetrics> runWithMetrics(
            List<Transaction> data, int k,
            double minSupportPct, double minConfidence,
            double chiSqThreshold, int coverageDelta,
            long seed, int maxPatternLength,
            Supplier<CMARClassifier> classifierFactory,
            double smoteTargetRatio,
            SmoteVariant smoteVariant,
            boolean useCcoWeighting,
            FPGrowth.WeightMode weightMode,
            boolean usePerClassMinSup) {

        // --- Chia có phân tầng: nhóm theo lớp, rồi phân phối ---
        List<Transaction> shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, new Random(seed));

        Map<String, List<Transaction>> byClass = new java.util.LinkedHashMap<>();
        for (Transaction t : shuffled) {
            byClass.computeIfAbsent(t.getClassLabel(), c -> new ArrayList<>()).add(t);
        }

        @SuppressWarnings("unchecked")
        List<Transaction>[] folds = new ArrayList[k];
        for (int i = 0; i < k; i++) folds[i] = new ArrayList<>();
        for (List<Transaction> classGroup : byClass.values()) {
            for (int i = 0; i < classGroup.size(); i++) {
                folds[i % k].add(classGroup.get(i));
            }
        }

        List<EvalMetrics> results = new ArrayList<>();

        for (int fold = 0; fold < k; fold++) {
            List<Transaction> testData = folds[fold];
            List<Transaction> trainData = new ArrayList<>();
            for (int j = 0; j < k; j++) if (j != fold) trainData.addAll(folds[j]);

            int minSupport = Math.max(2, (int) Math.round(trainData.size() * minSupportPct));

            // --- SMOTE: áp dụng oversampling minority class TRƯỚC khi tính thresholds ---
            if (smoteTargetRatio > 0) {
                int beforeSize = trainData.size();
                String tag;
                switch (smoteVariant) {
                    case BORDERLINE:
                        trainData = BorderlineSMOTE.apply(trainData, 5, smoteTargetRatio, seed + fold);
                        tag = "Borderline-SMOTE";
                        break;
                    case MWMOTE:
                        trainData = MWMOTE.apply(trainData, 5, smoteTargetRatio, seed + fold);
                        tag = "MWMOTE";
                        break;
                    case VANILLA:
                    default:
                        trainData = SMOTE.apply(trainData, 5, smoteTargetRatio, seed + fold);
                        tag = "SMOTE";
                        break;
                }
                minSupport = Math.max(2, (int) Math.round(trainData.size() * minSupportPct));
                if (fold == 0) {
                    System.out.println("    " + tag + " applied (fold 0): " + beforeSize
                        + " -> " + trainData.size() + " records");
                }
            }

            // --- Mining CAR ---
            FPGrowth fpGrowth = new FPGrowth(minSupport);
            fpGrowth.setMaxPatternLength(maxPatternLength);
            fpGrowth.setUseCcoWeighting(useCcoWeighting);  // MCWCAR weighted-support
            fpGrowth.setWeightMode(weightMode);            // CCO / MI / HYBRID
            // Phase 1: per-class minSup (MSApriori-lite) — tính trên distribution SAU SMOTE
            if (usePerClassMinSup) {
                Map<String, Integer> classCount = new java.util.HashMap<>();
                for (Transaction t : trainData) classCount.merge(t.getClassLabel(), 1, Integer::sum);
                fpGrowth.setMinSupByClass(FPGrowth.computeMinSupByClass(classCount, minSupport));
            }
            List<AssociationRule> candidates = fpGrowth.mine(trainData, minConfidence);

            // --- Huấn luyện classifier ---
            CMARClassifier classifier = classifierFactory.get();
            classifier.setChiSquareThreshold(chiSqThreshold);
            classifier.setCoverageThreshold(coverageDelta);
            classifier.train(candidates, trainData);

            // --- Dự đoán & đánh giá ---
            List<String> predictions = classifier.predict(testData);
            EvalMetrics metrics = EvalMetrics.compute(testData, predictions);
            results.add(metrics);

            System.out.printf("    Fold %2d: acc=%.4f macroF1=%.4f  (train=%d, test=%d, minSup=%d, rules=%d)%n",
                fold + 1, metrics.accuracy, metrics.macroF1,
                trainData.size(), testData.size(), minSupport,
                classifier.getRules().size());
        }

        return results;
    }
}
