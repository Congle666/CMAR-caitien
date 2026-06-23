# Scout Report — CMAR Codebase (cho plan tăng F1/Recall)

## Kiến trúc (plain Java, JDK 17, không Maven/Gradle)
Build: `javac -encoding UTF-8 -d out src/*.java` · Run: `java -cp out BenchmarkImbalanced`

## File cốt lõi & điểm can thiệp

| File | Vai trò | Điểm can thiệp cho plan |
|------|---------|-------------------------|
| `src/FPGrowth.java` | Khai phá CAR. Có `WeightMode{CCO,MI,HYBRID}`, `computeCcoWeights`, weighted-support `count×(1+w)≥minSupport` (dòng ~108, ~150). `mine(trainData, minConf)` nhận `minSupport` qua constructor. | **MSApriori per-class minSup** sửa ở đây: hiện minSupport là 1 số global. Cần cho phép minSup riêng theo class khi sinh CAR (dòng ~150 `classSup < minSupport`). |
| `src/CMARClassifier.java` | Train + classify. Có sẵn nhiều voting: weightedChi2, confLift, Laplace, ensemble, AV. Cờ: `useAvVoting`, `costSensitiveBeta`, `adaptiveAvThreshold`, `stratifiedTopK`, `voteTopK`. classify() có adaptive gating qua `originalImbalanceRatio`. | Thêm voting/threshold mới ở đây. classifyByAV (dòng ~442) đã có cost-sensitive β. |
| `src/CrossValidator.java` | 10-fold stratified CV. `runWithMetrics(...)` nhiều overload; SMOTE chỉ áp train (đã verify no-leak). minSup tính lại sau SMOTE. Truyền `useCcoWeighting`, `WeightMode`. | Thêm tham số per-dataset (ratio, per-class minSup) qua overload mới. |
| `src/EvalMetrics.java` | Acc, macroF1, weightedF1, per-class P/R/F1. `average()` aggregate qua folds. | **Thêm G-mean, MCC, balanced-acc** ở đây (researcher-02 đề xuất cho imbalanced). |
| `src/BenchmarkImbalanced.java` | Driver 7 imbalanced datasets. Factory hiện: stratifiedTopK=10, AV, voteTopK=10, costSensitiveBeta=0.3. | Driver chính để verify. `DATASETS` public. |
| `src/Benchmark.java` | Driver 19 datasets, `DATASETS` public, có minSup tinh chỉnh sẵn. | Dùng cho benchmark full + win/tie/loss. |
| `src/{SMOTE,BorderlineSMOTE,MWMOTE}.java` | Oversampling. MWMOTE (Barua 2014) đang dùng khi min<5. | — |

## Trạng thái cải tiến hiện tại (đã làm)
1. SMOTE/MWMOTE adaptive (min<5). 2. CCO weighted-support `(1+w)`. 3. AV voting. 4. Adaptive gating (ratio≥2 mới bật). 5. Cost-sensitive β=0.3.

## Bottleneck đã chẩn đoán (số thật từ CSV)
- **glass class3: F1=0.138** (P=0.17, R=0.12) — tệ nhất, multi-class 6 lớp.
- **glass class6 (9 mẫu): F1=0.526**. **zoo class3 (5 mẫu): F1=0.60**.
- **hepatitis class1: P=0.52** (nhiều false positive).
- **Degradation khi MCWCAR luôn bật:** tic-tac-toe F1 0.970→0.895 (−0.075), horse 0.830→0.787 (−0.043). Gating hiện tắt cải tiến trên các dataset này (giữ baseline, ko tăng ko giảm).

## 5 dataset imbalanced (ratio≥2) — phạm vi chính
lymph(40.5), zoo(10.3), glass(8.4), hepatitis(3.8), german(2.3).
14 dataset còn lại ratio<2 → gating giữ baseline.

## Đánh giá metric hiện tại
EvalMetrics CHƯA có G-mean/MCC/balanced-acc — researcher-02 nói đây là metric chuẩn để chứng minh cải tiến imbalanced. Nên bổ sung.
