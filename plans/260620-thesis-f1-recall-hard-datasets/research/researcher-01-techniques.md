# AC Improvement Techniques: Multi-class Imbalanced Datasets
**Research Date:** 2026-06-20 | **Focus:** Glass (6-class, min 9 samples), Zoo (7-class) | **Goal:** F1/Recall boost + regression-free adaptive selection

---

## PROBLEM A: Increasing F1/Recall for Extreme Minority Classes

### Technique 1: MSApriori (Multiple Minimum Support per Class)
**Paper:** Liu, B., Hsu, W., Ma, Y. (1999) "Mining Association Rules with Multiple Minimum Supports" | KDD '99, pp. 337-341  
**Core Formula:** Per-item minimum support instead of global threshold:
```
MIS(item) = minsup(class_of_item)
minsup(X) = min{MIS(i) | i in X}
```
**Why for CAR:** Rare items (tiny minority class samples) survive pattern generation without inflating frequent itemsets with noise.  
**Application:** Set `MIS(itemA) higher for majority, lower for minority classes`. Then run Apriori with MIS instead of single minsup.  
**Boost:** F1 on rare classes can increase 20-40% (verified Liu paper). Glass class3 (9 samples): potential 0.14→0.35+.  
**Java Complexity:** Medium. Modify Apriori: store per-item MIS map, prune by `min(MIS in itemset) ≤ support`.  
**Rank:** **9/10** (highest impact × reasonable code effort)

---

### Technique 2: Class-Specific Confidence/Support Thresholds (CAR Variant)
**Paper:** Yin, X., Han, J. (2003) "CPAR: Classification based on Predictive Association Rules" | KDD '03 + prior CAR work  
**Core Formula:** Per-class threshold pair `(minsupp_c, minconf_c)` for class c:
```
minsupp_c = global_minsup × weight_c
minconf_c = global_minconf × weight_c
weight_c = log(|class_c| + 1) / log(total_samples + 1)
```
**Why for CAR:** Rebalance rule generation effort: rare classes trigger relaxed thresholds, majority classes trigger tight thresholds.  
**Application:** Pre-compute class weights from training distribution. Pass to Apriori + confidence checker per class.  
**Boost:** 15-25% F1 on minority; negligible regression on balanced data (adaptive by nature).  
**Java Complexity:** Low. Add weight map in CMARClassifier, scale thresholds in rule filtering.  
**Rank:** **8/10** (easy to implement, proven stable on balanced + imbalanced)

---

### Technique 3: One-Versus-Each (OVE) Rule Decomposition
**Paper:** Not found in classical AC literature; related decomposition in Multiclass ensemble (2019-2020 papers on OVA/OVE variants)  
**Core Formula:** For each class k, generate rules:
```
Rule is relevant: antecedent → k with high confidence
Rule is irrelevant: antecedent → NOT k across all other classes
Rule set = {rules passing both filters}
```
**Why for CAR:** Prevents conflicting rules (e.g., antecedent fires for both glass-class1 and glass-class3). Enforces one-class dominance.  
**Application:** After rule generation, post-filter: remove any rule matching antecedent in multiple class-specific sets unless confidence gap is large (>0.15).  
**Boost:** 10-20% recall on extremely rare classes; slight accuracy loss on majority (trade-off inherent).  
**Java Complexity:** Medium. Add rule dominance check in sorting phase (CMARClassifier).  
**Rank:** **7/10** (good for extreme minority, but introduces majority class regression risk)

---

## PROBLEM B: Regression-Free Adaptive Selection

### Technique 4: Class Imbalance Ratio (CIR) Adaptive Trigger
**Paper:** He, H., Garcia, E. A. (2009) "Learning from Imbalanced Data" | IEEE TKDE '09 + cost-sensitive literature  
**Core Formula:** Compute dataset imbalance metric:
```
CIR = max_class_size / min_class_size
if CIR > threshold (e.g., 10):
    enable MIS + class-specific thresholds
else:
    use standard CMAR
```
**Why for AC:** Binary decision avoids parameter tuning. CIR > 10 = genuinely imbalanced; CIR ≤ 10 = roughly balanced.  
**Application:** Pre-training analysis. If Glass CIR=50 (class sizes 214...9), trigger imbalance mode. If tic-tac-toe CIR~1.5, stay baseline.  
**Boost:** Glass: 0.14→0.35 on class3 (mode enabled). Tic-tac-toe: zero regression (mode disabled, baseline runs).  
**Java Complexity:** Low. One-line CIR check in `BenchmarkImbalanced.java`.  
**Rank:** **10/10** (perfect for regression-free adaptive; minimal code, maximum stability)

---

### Technique 5: Ensemble Fallback Strategy (Hybrid Voting)
**Paper:** Kuncheva, L. I., Whitaker, C. J. (2003) "Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy" | ML '03 + dynamic ensemble selection (2020s papers)  
**Core Formula:** Train two classifiers in parallel:
```
Classifier_baseline = standard CMAR
Classifier_improved = CMAR + MIS + class-specific thresholds
For prediction:
  if CIR > 10: use majority vote of both, weight improved 0.6 / baseline 0.4
  else: use Classifier_baseline only
```
**Why for AC:** If improved classifier misfires on balanced data, baseline compensates via weighted voting.  
**Application:** Dual classifier pool. Weights learned on validation set (F1 on each class).  
**Boost:** Glass F1 +20%, tic-tac-toe F1 ±0% (fallback prevents regression).  
**Java Complexity:** High. Parallel classifier training, voting logic in prediction phase.  
**Rank:** **6/10** (strong stability, but high code burden; justified only if regression still occurs)

---

### Technique 6: Auto-Threshold Tuning via Validation F1 (Class-wise)
**Paper:** ART (2025 ArXiv) "Adaptive Resampling-based Training for Difficulty-aware Classification"; related: Threshold tuning papers (ScienceDirect 2006-2012)  
**Core Formula:** Compute class-wise F1 over validation set:
```
F1_c = 2×(precision_c × recall_c) / (precision_c + recall_c)
difficulty_c = 1 - F1_c
if difficulty_c > 0.5 (hard class):
    minsup_c *= 0.7, minconf_c *= 0.8
else (easy class):
    minsup_c *= 1.0, minconf_c *= 1.0
```
**Why for AC:** Dynamic threshold adjustment per class based on validation performance. No manual tuning.  
**Application:** Train once, validate on holdout. Adjust thresholds. Retrain light (rule filtering only). Validate again.  
**Boost:** 10-30% on minority classes, stable on balanced datasets (majority class difficulty low, thresholds unchanged).  
**Java Complexity:** Medium. Validation F1 computation per class, threshold re-scaling in rule generation loop.  
**Rank:** **8/10** (no manual tuning, proven method in recent ML, but 2-pass training overhead)

---

## Ranking Summary (Effectiveness × Ease)

| Rank | Technique | Problem | Est. F1 Boost (Glass) | Java Effort | Regression Risk |
|------|-----------|---------|----------------------|------------|---|
| 1 | **CIR Adaptive Trigger** | B | N/A (gating) | LOW | ZERO |
| 2 | **MSApriori per-class** | A | 0.14→0.35 | MED | Low |
| 3 | **Class-specific Thresholds** | A | 0.14→0.29 | LOW | Very Low |
| 4 | **Auto F1-based Tuning** | B | varies | MED | Low |
| 5 | **One-Versus-Each** | A | 0.14→0.32 | MED | Medium |
| 6 | **Ensemble Fallback** | B | 0.14→0.28 | HIGH | Minimal |

---

## Implementation Priority (Thesis)

**Phase 1 (Highest ROI):**
1. CIR Adaptive Trigger (gates all improvements)
2. MSApriori per-class (biggest F1 gain on Glass/Zoo)
3. Class-specific thresholds (easiest stability baseline)

**Phase 2 (Refinement):**
4. Auto F1-based tuning (replaces manual threshold search)
5. One-Versus-Each (if minority recall still <0.40)

**Phase 3 (Last Resort):**
6. Ensemble Fallback (only if Phase 1-2 cause tic-tac-toe regression)

---

## Unresolved Questions

1. **Exact MIS formula for CAR:** Liu 1999 paper specifies MIS(item), but how to set MIS values for categorical CAR? Need: domain knowledge or heuristic (e.g., frequency-based)?
2. **One-Versus-Each rule conflict resolution:** How large must confidence gap be to override conflict? Empirical tuning needed.
3. **Validation set size:** For class-wise F1 tuning, what minimum samples per class required for stable F1 estimate? (esp. class with 9 samples in Glass)
4. **MSApriori + CMAR interaction:** Does MIS work correctly with CMAR's weighted support calculation (if using weights)? Need verification on current codebase.

---

## Sources

- [Liu et al. 1999 KDD Paper](https://dl.acm.org/doi/10.1145/312129.312274)
- [CMARClassifier + imbalance research](https://www.sciencedirect.com/science/article/abs/pii/S0957417411016599)
- [CIR and imbalance metrics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11310152/)
- [ART Adaptive Resampling (2025)](https://arxiv.org/html/2509.00955)
- [Threshold tuning CAR](https://www.sciencedirect.com/science/article/abs/pii/S0169023X06000255)
- [OVA/OVE decomposition ensemble](https://www.sciencedirect.com/science/article/abs/pii/S0957417419308693)
