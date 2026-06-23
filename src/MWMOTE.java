import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * MWMOTE-N — Majority Weighted Minority Oversampling Technique, categorical adaptation.
 *
 * <p>Faithful implementation of Algorithm 1 in:
 * Barua, S., Islam, M.M., Yao, X., &amp; Murase, K. (2014).
 * "MWMOTE—Majority Weighted Minority Oversampling Technique for Imbalanced
 * Data Set Learning". IEEE Transactions on Knowledge and Data Engineering 26(2),
 * 405-425. DOI: 10.1109/TKDE.2012.232.</p>
 *
 * <p><b>Algorithm (Barua 2014 §4):</b></p>
 * <ol>
 *   <li>Filter noise: Sminf = {x ∈ Smin : not all k1-NN are majority}</li>
 *   <li>Sbmaj = ⋃_{x∈Sminf} Nmaj(x, k2) — borderline majority</li>
 *   <li>Simin = ⋃_{y∈Sbmaj} Nmin(y, k3) ∩ Sminf — informative minority</li>
 *   <li>For each (yi, xi) ∈ Sbmaj × Simin, compute Cf(yi,xi), Df(yi,xi), Iw(yi,xi)</li>
 *   <li>Selection weight Sw(xi) = Σ_yi Iw(yi, xi); normalize → prob distribution</li>
 *   <li>Cluster Smin using avg-linkage agglomerative clustering, Th = davg × Cp</li>
 *   <li>For N synthetic: pick x by Sw, pick y from same cluster, synthesize</li>
 * </ol>
 *
 * <p><b>Categorical adaptations:</b></p>
 * <ul>
 *   <li>Hamming distance thay Euclidean (Chawla 2002 §6.2 SMOTE-N)</li>
 *   <li>Cf cut-off function: f(z) = min(z, CMAX)</li>
 *   <li>Synthetic generation: mode voting (SMOTE-N) thay α-interpolation</li>
 *   <li>dn = dist (không chia dimension — Hamming đã proportional)</li>
 * </ul>
 */
public final class MWMOTE {

    // Defaults per Barua 2014 Table 1 (recommended)
    private static final int    K1     = 5;    // noise filter
    private static final int    K2     = 3;    // borderline majority
    private static final int    K3     = 5;    // informative minority
    private static final double CMAX   = 2.0;  // closeness max
    private static final double CF_TH  = 5.0;  // cut-off threshold for f
    private static final double CP     = 3.0;  // clustering threshold multiplier

    private MWMOTE() {}

    public static List<Transaction> apply(List<Transaction> data, int k,
                                            double targetRatio, long seed) {
        if (data.isEmpty()) return new ArrayList<>(data);

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
            List<Transaction> smin = e.getValue();
            if (smin.size() >= target) continue;
            int needed = target - smin.size();

            if (smin.size() < 2) {
                for (int i = 0; i < needed; i++) augmented.add(smin.get(i % smin.size()));
                continue;
            }

            // Smaj = all non-minority records
            List<Transaction> smaj = new ArrayList<>();
            for (Transaction t : data) if (!t.getClassLabel().equals(cls)) smaj.add(t);
            if (smaj.isEmpty()) {
                for (int i = 0; i < needed; i++) augmented.add(smin.get(rng.nextInt(smin.size())));
                continue;
            }

            // === Step 1: Sminf — filter noise
            // x is noise IFF all K1-NN (in entire training data) are majority
            int k1Eff = Math.min(K1, data.size() - 1);
            List<Transaction> sminf = new ArrayList<>();
            for (Transaction x : smin) {
                List<Transaction> knn = kNN(x, data, k1Eff);
                int majCount = 0;
                for (Transaction n : knn) if (!n.getClassLabel().equals(cls)) majCount++;
                if (majCount < k1Eff) sminf.add(x);
            }
            if (sminf.isEmpty()) sminf = new ArrayList<>(smin);

            // === Step 2: Sbmaj — union of k2-nearest majority for each x in Sminf
            int k2Eff = Math.min(K2, smaj.size());
            Set<Transaction> sbmaj = new LinkedHashSet<>();
            for (Transaction x : sminf) sbmaj.addAll(kNN(x, smaj, k2Eff));
            if (sbmaj.isEmpty()) sbmaj = new LinkedHashSet<>(smaj);

            // === Step 3: Simin — union of k3-nearest filtered-minority for each y in Sbmaj
            int k3Eff = Math.min(K3, sminf.size());
            Set<Transaction> siminSet = new LinkedHashSet<>();
            // ALSO track Nmin(y) per y — needed for Cf=0 outside k3-NN constraint
            Map<Transaction, Set<Transaction>> nminOfY = new HashMap<>();
            for (Transaction y : sbmaj) {
                List<Transaction> nn = kNN(y, sminf, k3Eff);
                nminOfY.put(y, new HashSet<>(nn));
                siminSet.addAll(nn);
            }
            if (siminSet.isEmpty()) siminSet = new LinkedHashSet<>(sminf);
            List<Transaction> simin = new ArrayList<>(siminSet);

            // === Step 4-5: Cf, Df, Iw, Sw
            // Cf(yi, xi) = (f(1/dn / CF_TH)) * CMAX, where f cut-off, dn = normalized dist
            // Cf = 0 if xi ∉ Nmin(yi)  ← BUGFIX
            // Df(yi, xi) = Cf(yi, xi) / Σ_q∈Simin Cf(yi, q)  ← BUGFIX (proper density)
            // Iw(yi, xi) = Cf × Df
            // Sw(xi) = Σ_yi Iw(yi, xi)
            int dim = estimateDimension(data);
            int sm = simin.size();
            double[][] cf = new double[sbmaj.size()][sm];
            double[]   cfRowSum = new double[sbmaj.size()];

            int yi = 0;
            for (Transaction y : sbmaj) {
                Set<Transaction> nminY = nminOfY.get(y);
                for (int xi = 0; xi < sm; xi++) {
                    Transaction x = simin.get(xi);
                    if (nminY == null || !nminY.contains(x)) {
                        cf[yi][xi] = 0.0;  // Bug-fixed: Cf=0 ngoài k3-NN
                        continue;
                    }
                    double dist = hammingDistance(y.getItems(), x.getItems());
                    if (dist <= 0) {
                        cf[yi][xi] = CMAX;  // Identical records: max closeness
                    } else {
                        double dn = dist / (double) Math.max(1, dim);  // normalized
                        double inv = 1.0 / dn;
                        // Cut-off function: f(z) = min(z, CF_TH) / CF_TH
                        double fv = Math.min(inv, CF_TH) / CF_TH;
                        cf[yi][xi] = fv * CMAX;
                    }
                    cfRowSum[yi] += cf[yi][xi];
                }
                yi++;
            }

            double[] sw = new double[sm];
            double swSum = 0.0;
            for (int yiIdx = 0; yiIdx < cf.length; yiIdx++) {
                double rowSum = cfRowSum[yiIdx];
                if (rowSum <= 0) continue;
                for (int xi = 0; xi < sm; xi++) {
                    double cfVal = cf[yiIdx][xi];
                    if (cfVal <= 0) continue;
                    double df = cfVal / rowSum;     // Bug-fixed: Eq 8 proper density factor
                    double iw = cfVal * df;
                    sw[xi] += iw;
                }
            }
            for (double v : sw) swSum += v;
            if (swSum <= 0) {
                java.util.Arrays.fill(sw, 1.0 / sm);
            } else {
                for (int i = 0; i < sw.length; i++) sw[i] /= swSum;
            }

            // === Step 6: Cluster Smin (TRUE agglomerative avg-linkage)
            // davg = avg MIN distance per record in Sminf
            // Th = davg × Cp
            double davg = computeAvgMinDistance(sminf);
            double th = davg * CP;
            // Cluster Smin (full minority, not Sminf — per paper Step 10)
            int[] cidSmin = agglomerativeAvgLink(smin, th);
            // Map cluster-id from Smin → Simin members
            Map<Transaction, Integer> minToCluster = new HashMap<>();
            for (int i = 0; i < smin.size(); i++) minToCluster.put(smin.get(i), cidSmin[i]);
            // Build cluster → members (in Smin) for sampling y from same cluster
            Map<Integer, List<Transaction>> clusterMembers = new HashMap<>();
            for (int i = 0; i < smin.size(); i++) {
                clusterMembers.computeIfAbsent(cidSmin[i], c -> new ArrayList<>()).add(smin.get(i));
            }

            // === Step 7: Generate synthetic
            for (int s = 0; s < needed; s++) {
                int xIdx = sampleByWeight(sw, rng);
                Transaction x = simin.get(xIdx);
                Integer xCluster = minToCluster.get(x);
                Transaction y;
                List<Transaction> cm = (xCluster != null) ? clusterMembers.get(xCluster) : null;
                if (cm != null && cm.size() >= 2) {
                    Transaction picked;
                    do { picked = cm.get(rng.nextInt(cm.size())); }
                    while (picked == x && cm.size() > 1);
                    y = picked;
                } else if (cm != null) {
                    y = cm.get(0);  // singleton cluster (noise sample) → duplicate
                } else {
                    y = smin.get(rng.nextInt(smin.size()));
                }
                augmented.add(createSynthetic(x, y, cls, rng));
            }
        }
        return augmented;
    }

    public static List<Transaction> apply(List<Transaction> data) {
        return apply(data, 5, 1.0, 42);
    }

    // -------------------- Helpers --------------------

    /** Estimate dimension (number of distinct attribute names) for dn normalization. */
    private static int estimateDimension(List<Transaction> data) {
        Set<String> attrs = new HashSet<>();
        int probe = Math.min(50, data.size());
        for (int i = 0; i < probe; i++) {
            for (String item : data.get(i).getItems()) {
                int idx = item.indexOf('=');
                if (idx > 0) attrs.add(item.substring(0, idx));
            }
        }
        return Math.max(1, attrs.size());
    }

    /** Eq 9: davg = avg over x of MIN distance to any other y ∈ Sminf. */
    private static double computeAvgMinDistance(List<Transaction> sminf) {
        if (sminf.size() < 2) return 1.0;
        double sum = 0.0;
        int count = 0;
        for (Transaction x : sminf) {
            int minD = Integer.MAX_VALUE;
            for (Transaction y : sminf) {
                if (x == y) continue;
                int d = hammingDistance(x.getItems(), y.getItems());
                if (d < minD) minD = d;
            }
            if (minD != Integer.MAX_VALUE) {
                sum += minD;
                count++;
            }
        }
        return (count > 0) ? sum / count : 1.0;
    }

    /**
     * TRUE agglomerative average-linkage clustering với Hamming distance.
     * Merge closest cluster pair (by avg pairwise dist) until min > th.
     */
    private static int[] agglomerativeAvgLink(List<Transaction> items, double th) {
        int n = items.size();
        int[] cid = new int[n];
        for (int i = 0; i < n; i++) cid[i] = i;
        if (n <= 1) return cid;

        // Active clusters
        Map<Integer, List<Integer>> clusters = new HashMap<>();
        for (int i = 0; i < n; i++) {
            List<Integer> list = new ArrayList<>();
            list.add(i);
            clusters.put(i, list);
        }

        // Pairwise distance cache
        int[][] dist = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                dist[i][j] = dist[j][i] = hammingDistance(items.get(i).getItems(), items.get(j).getItems());
            }
        }

        while (clusters.size() > 1) {
            // Find closest pair by AVG-LINK distance
            double minAvgDist = Double.MAX_VALUE;
            int bestA = -1, bestB = -1;
            Integer[] keys = clusters.keySet().toArray(new Integer[0]);
            for (int i = 0; i < keys.length; i++) {
                for (int j = i + 1; j < keys.length; j++) {
                    List<Integer> ca = clusters.get(keys[i]);
                    List<Integer> cb = clusters.get(keys[j]);
                    double sum = 0;
                    int cnt = 0;
                    for (int a : ca) for (int b : cb) { sum += dist[a][b]; cnt++; }
                    double avg = (cnt > 0) ? sum / cnt : Double.MAX_VALUE;
                    if (avg < minAvgDist) {
                        minAvgDist = avg;
                        bestA = keys[i]; bestB = keys[j];
                    }
                }
            }
            if (bestA < 0 || minAvgDist > th) break;
            // Merge bestB into bestA
            clusters.get(bestA).addAll(clusters.get(bestB));
            clusters.remove(bestB);
        }

        // Assign final cluster IDs
        int newId = 0;
        for (Map.Entry<Integer, List<Integer>> e : clusters.entrySet()) {
            for (int idx : e.getValue()) cid[idx] = newId;
            newId++;
        }
        return cid;
    }

    private static int sampleByWeight(double[] probs, Random rng) {
        double r = rng.nextDouble();
        double cum = 0;
        for (int i = 0; i < probs.length; i++) {
            cum += probs[i];
            if (r <= cum) return i;
        }
        return probs.length - 1;
    }

    private static List<Transaction> kNN(Transaction base, List<Transaction> pool, int k) {
        List<Map.Entry<Transaction, Integer>> distances = new ArrayList<>();
        List<String> baseItems = base.getItems();
        for (Transaction t : pool) {
            if (t == base) continue;
            distances.add(new java.util.AbstractMap.SimpleEntry<>(t, hammingDistance(baseItems, t.getItems())));
        }
        distances.sort(Comparator.comparingInt(Map.Entry::getValue));
        List<Transaction> result = new ArrayList<>();
        for (int i = 0; i < Math.min(k, distances.size()); i++) result.add(distances.get(i).getKey());
        return result;
    }

    private static int hammingDistance(List<String> a, List<String> b) {
        Map<String, String> ma = itemsToMap(a);
        Map<String, String> mb = itemsToMap(b);
        Set<String> all = new HashSet<>(ma.keySet());
        all.addAll(mb.keySet());
        int diff = 0;
        for (String attr : all) {
            String va = ma.get(attr), vb = mb.get(attr);
            if (va == null || vb == null || !va.equals(vb)) diff++;
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

    private static Transaction createSynthetic(Transaction a, Transaction b, String cls, Random rng) {
        Map<String, Map<String, Integer>> attrVal = new HashMap<>();
        addToCount(a.getItems(), attrVal);
        addToCount(b.getItems(), attrVal);
        List<String> syn = new ArrayList<>();
        for (Map.Entry<String, Map<String, Integer>> e : attrVal.entrySet()) {
            int maxC = 0;
            for (int c : e.getValue().values()) if (c > maxC) maxC = c;
            List<String> modes = new ArrayList<>();
            for (Map.Entry<String, Integer> ve : e.getValue().entrySet()) {
                if (ve.getValue() == maxC) modes.add(ve.getKey());
            }
            syn.add(e.getKey() + "=" + modes.get(rng.nextInt(modes.size())));
        }
        return new Transaction(syn, cls);
    }

    private static void addToCount(List<String> items, Map<String, Map<String, Integer>> counts) {
        for (String item : items) {
            int idx = item.indexOf('=');
            if (idx <= 0) continue;
            counts.computeIfAbsent(item.substring(0, idx), a -> new HashMap<>())
                  .merge(item.substring(idx + 1), 1, Integer::sum);
        }
    }
}
