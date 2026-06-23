import os
import sys

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except Exception as e:
    print("Missing Python packages. Please install with:")
    print("    pip install pandas matplotlib numpy")
    raise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BASE_METRICS = os.path.join(ROOT, 'result', 'final_baseline_metrics.csv')
FULL_METRICS = os.path.join(ROOT, 'result', 'final_full_metrics.csv')
BASE_PERCLASS = os.path.join(ROOT, 'result', 'final_baseline_per_class.csv')
FULL_PERCLASS = os.path.join(ROOT, 'result', 'final_full_per_class.csv')
OUT_DIR = os.path.join(ROOT, 'report')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Read metrics
bm = pd.read_csv(BASE_METRICS)
fm = pd.read_csv(FULL_METRICS)

# Helper: compute macro recall per dataset from per-class CSV
def compute_macro_recall(perclass_path):
    pc = pd.read_csv(perclass_path)
    # consider classes with support > 0
    pc = pc[pc['support'] > 0]
    # group by dataset, compute mean recall per dataset
    mr = pc.groupby('dataset')['recall'].mean().reset_index().rename(columns={'recall':'macroRecall'})
    return mr

bm_rec = compute_macro_recall(BASE_PERCLASS)
fm_rec = compute_macro_recall(FULL_PERCLASS)

# Merge metrics
merge_base = bm.merge(bm_rec, on='dataset', how='left')
merge_full = fm.merge(fm_rec, on='dataset', how='left')

# Build summary table
cols = ['dataset']
summary = pd.DataFrame()
summary['dataset'] = merge_base['dataset']
summary['baseline_acc'] = merge_base['accuracy']
summary['full_acc'] = merge_full['accuracy']
summary['delta_acc'] = summary['full_acc'] - summary['baseline_acc']
summary['baseline_macroF1'] = merge_base['macroF1']
summary['full_macroF1'] = merge_full['macroF1']
summary['delta_macroF1'] = summary['full_macroF1'] - summary['baseline_macroF1']
summary['baseline_macroRecall'] = merge_base['macroRecall']
summary['full_macroRecall'] = merge_full['macroRecall']
summary['delta_macroRecall'] = summary['full_macroRecall'] - summary['baseline_macroRecall']

OUT_CSV = os.path.join(OUT_DIR, 'improvement_metrics.csv')
summary.to_csv(OUT_CSV, index=False)
print('Wrote', OUT_CSV)

# Plotting helper
def bar_compare(x, y1, y2, labels, title, outpath, ylabel=''):
    n = len(x)
    ind = np.arange(n)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, n*0.35), 6))
    ax.bar(ind - width/2, y1, width, label='Baseline')
    ax.bar(ind + width/2, y2, width, label='Full')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print('Saved', outpath)

labels = summary['dataset'].tolist()
bar_compare(summary['dataset'], summary['baseline_acc'], summary['full_acc'], labels,
            'Accuracy: Baseline vs Full', os.path.join(PLOT_DIR, 'accuracy_compare.png'), 'Accuracy')
bar_compare(summary['dataset'], summary['baseline_macroF1'], summary['full_macroF1'], labels,
            'Macro-F1: Baseline vs Full', os.path.join(PLOT_DIR, 'macroF1_compare.png'), 'Macro-F1')
bar_compare(summary['dataset'], summary['baseline_macroRecall'], summary['full_macroRecall'], labels,
            'Macro-Recall: Baseline vs Full', os.path.join(PLOT_DIR, 'macroRecall_compare.png'), 'Macro-Recall')

print('All done. Plots in', PLOT_DIR)
