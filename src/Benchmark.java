import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Chạy 10-fold stratified cross-validation trên toàn bộ dataset UCI
 * dùng trong paper CMAR (Li, Han, Pei — ICDM 2001, Table 1).
 *
 * Gom {@link EvalMetrics} của từng fold (Accuracy + Macro-F1 + Weighted-F1
 * + P/R/F1 theo lớp) và xuất ra hai file CSV để so sánh sau này:
 *   - result/baseline_metrics.csv       (1 dòng cho mỗi dataset)
 *   - result/baseline_per_class.csv     (1 dòng cho mỗi cặp (dataset, lớp))
 *
 * Dùng minSupport theo phần trăm (đã tinh chỉnh riêng từng dataset) để
 * khớp với cấu hình của paper.
 */
public class Benchmark {

    // Kết quả từ paper để so sánh (Table 1, ICDM 2001)
    // minSupportPct được tinh chỉnh từng dataset để mining khả thi.
    public static final String[][] DATASETS = {
        // { file, name, minSupportPct, paperCMAR, paperCBA, paperC45, [maxPatternLength] }
        // Cột thứ 7 (tuỳ chọn): giới hạn độ dài pattern để tránh bùng nổ
        // trên dataset nhiều chiều (zoo có 16 thuộc tính nhị phân → ~2^16 ứng viên)
        { "data_clean/breast-w.csv",    "breast-w",    "0.02", "96.42", "96.28", "95.00" },
        { "data_clean/cleve.csv",       "cleve",       "0.02", "82.18", "82.83", "78.24" },
        { "data_clean/crx.csv",         "crx",         "0.04", "85.36", "84.93", "84.94" },
        { "data_clean/diabetes.csv",    "diabetes",    "0.03", "75.81", "74.47", "74.18" },
        { "data_clean/german.csv",      "german",      "0.06", "73.40", "73.40", "72.30" },
        { "data_clean/glass.csv",       "glass",       "0.01", "70.09", "67.76", "68.22" },
        { "data_clean/heart.csv",       "heart",       "0.03", "82.59", "81.85", "80.74" },
        { "data_clean/hepatitis.csv",   "hepatitis",   "0.05", "80.65", "81.29", "80.00" },
        { "data_clean/horse.csv",       "horse",       "0.03", "82.61", "82.07", "82.61" },
        { "data_clean/iris.csv",        "iris",        "0.03", "94.00", "94.67", "95.33" },
        { "data_clean/labor.csv",       "labor",       "0.05", "89.47", "86.33", "79.33" },
        { "data_clean/led7.csv",        "led7",        "0.03", "71.90", "71.70", "73.50" },
        { "data_clean/lymph.csv",       "lymph",       "0.05", "82.43", "77.03", "73.51" },
        { "data_clean/sonar.csv",       "sonar",       "0.05", "79.33", "76.92", "73.56", "5" },
        { "data_clean/tic-tac-toe.csv", "tic-tac-toe", "0.003","99.27", "99.06", "99.37" },
        { "data_clean/vehicle.csv",     "vehicle",     "0.03", "68.68", "67.73", "72.34", "5" },
        { "data_clean/waveform.csv",    "waveform",    "0.01", "80.17", "79.93", "78.10", "5" },
        { "data_clean/wine.csv",        "wine",        "0.03", "95.51", "95.51", "92.70" },
        { "data_clean/zoo.csv",         "zoo",         "0.03", "96.04", "97.03", "93.07", "4" },
    };

    static final String OUT_METRICS_CSV   = "result/baseline_metrics.csv";
    static final String OUT_PER_CLASS_CSV = "result/baseline_per_class.csv";

    public static void main(String[] args) throws Exception {
        int K = 10;
        double minConfidence = 0.5;
        double chiSqThreshold = 3.841;
        int coverageDelta = 4;
        long seed = 42;

        // Tuỳ chọn lọc dataset: truyền tên qua args để chạy chỉ subset.
        // Ví dụ: java Benchmark lymph glass german
        Set<String> filter = args.length > 0 ? new HashSet<>(Arrays.asList(args)) : null;

        System.out.println("=================================================================");
        System.out.println("  CMAR Benchmark — 10-Fold Stratified CV + F1 Metrics");
        System.out.println("  Paper: Li, Han, Pei (ICDM 2001)");
        if (filter != null) System.out.println("  Filter: " + filter);
        System.out.println("=================================================================\n");

        // Header bảng
        System.out.printf("%-14s %5s %6s | %6s %7s %7s %5s | %6s %6s %6s  Diff%n",
            "Dataset", "N", "supPct", "Acc", "MacroF1", "WF1", "Std", "Paper", "CBA", "C4.5");
        System.out.println("-".repeat(90));

        // Các container tổng hợp để ghi ra CSV
        Map<String, EvalMetrics> aggregated = new LinkedHashMap<>();
        Map<String, String> info = new LinkedHashMap<>();

        double totalOursAcc = 0, totalOursF1 = 0, totalPaper = 0;
        double totalDiff = 0;
        int count = 0;
        int within5 = 0;

        for (String[] ds : DATASETS) {
            String file = ds[0];
            String name = ds[1];
            if (filter != null && !filter.contains(name)) continue;
            double supPct = Double.parseDouble(ds[2]);
            double paperCMAR = Double.parseDouble(ds[3]);
            double paperCBA  = Double.parseDouble(ds[4]);
            double paperC45  = Double.parseDouble(ds[5]);
            int maxPatLen    = ds.length > 6 ? Integer.parseInt(ds[6]) : Integer.MAX_VALUE;

            System.out.println("\n>>> " + name + " (" + file + ") ...");

            try {
                List<Transaction> data = DatasetLoader.load(file);
                if (data.isEmpty()) {
                    System.out.println("    BO QUA (dataset rong)");
                    continue;
                }

                List<EvalMetrics> foldMetrics = CrossValidator.runWithMetrics(
                    data, K, supPct, minConfidence, chiSqThreshold, coverageDelta, seed, maxPatLen);

                EvalMetrics agg = EvalMetrics.average(foldMetrics);
                aggregated.put(name, agg);
                info.put(name, data.size() + "|" + agg.perClass.size() + "|" + supPct);

                double accPct = agg.accuracy * 100;
                double stdPct = agg.accuracyStd * 100;
                double diff   = accPct - paperCMAR;

                System.out.printf("  => %-14s %5d %5.1f%% | %6.2f %7.4f %7.4f %5.2f | %6.2f %6.2f %6.2f  %+5.2f%n",
                    name, data.size(), supPct * 100,
                    accPct, agg.macroF1, agg.weightedF1, stdPct,
                    paperCMAR, paperCBA, paperC45, diff);

                totalOursAcc += accPct;
                totalOursF1  += agg.macroF1;
                totalPaper   += paperCMAR;
                totalDiff    += Math.abs(diff);
                count++;
                if (Math.abs(diff) <= 5.0) within5++;

            } catch (OutOfMemoryError e) {
                System.out.println("    HET BO NHO — hay tang -Xmx hoac tang minSupport");
            } catch (Exception e) {
                System.out.println("    LOI: " + e.getMessage());
                e.printStackTrace(System.out);
            }
        }

        // -----------------------------------------------------------------
        // Ghi các file CSV kết quả
        // -----------------------------------------------------------------
        ResultWriter.writeMetricsCsv(aggregated, info, OUT_METRICS_CSV);
        ResultWriter.writePerClassCsv(aggregated, OUT_PER_CLASS_CSV);

        // -----------------------------------------------------------------
        // Tổng kết
        // -----------------------------------------------------------------
        System.out.println("\n" + "=".repeat(90));
        System.out.println("TONG KET");
        System.out.println("=".repeat(90));
        if (count > 0) {
            System.out.printf("So dataset danh gia:          %d / %d%n", count, DATASETS.length);
            System.out.printf("Accuracy trung binh (ta):     %.2f%%%n", totalOursAcc / count);
            System.out.printf("Macro-F1 trung binh (ta):     %.4f%n",   totalOursF1 / count);
            System.out.printf("Accuracy trung binh (paper):  %.2f%%%n", totalPaper / count);
            System.out.printf("Chenh lech tuyet doi vs paper: %.2f%%%n", totalDiff / count);
            System.out.printf("Trong khoang 5%% cua paper:   %d / %d%n", within5, count);
        }
        System.out.println();
        System.out.println("File CSV xuat ra:");
        System.out.println("  " + OUT_METRICS_CSV);
        System.out.println("  " + OUT_PER_CLASS_CSV);
    }
}
