"""
Phase 4 - Regenerate manuscript figures as image files from real
reproduced results (REPRO_NOTES.md is the source of truth for all numbers
used here).

Produces (in repro/figures/):
  - figure3_actual_pipeline_direct_outputs.png
                              : actual submitted pipeline diagram (direct
                                artifacts/results/figures outputs)
  - class_distribution.png   : bar chart of the 8-class label distribution
  - confmat_<model>.png      : confusion matrix heatmap for each Protocol-A model
  - per_class_prf.png        : grouped bar chart of per-class precision/recall/F1
                                for all 4 Protocol-A models
  - cv_boxplot.png           : boxplot of repeated-CV macro-F1 (RF, GB) -
                                only produced if results_cv/cv_metrics.json exists
  - model_comparison.png     : accuracy/macro-F1 bar chart for all 4 Protocol-A
                                models plus the rule-based baseline - only
                                produced if results_rule_baseline/metrics.json
                                exists
"""
import json
import os
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ART_DIR = 'artifacts'
RESULTS_DIR = 'results'
CV_DIR = 'results_cv'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(ART_DIR, 'meta.json')) as f:
    meta = json.load(f)
with open(os.path.join(RESULTS_DIR, 'metrics.json')) as f:
    metrics = json.load(f)

classes = meta['classes']
n_classes = len(classes)

# ---------------------------------------------------------------------------
# 0. Actual submitted pipeline diagram (no MLflow)
# ---------------------------------------------------------------------------
def plot_actual_pipeline():
    stages = [
        (
            '1',
            'Data generation and telemetry',
            'Custom shell simulator plus linux_audit.rules produce Linux audit events; '
            'Auditbeat 7.16.3 forwards them to the Elasticsearch auditbeat-* index.',
            '#2563EB',
            '#EAF1FF',
        ),
        (
            '2',
            'Extraction, cleaning, and labels',
            'Apache Spark flattens nested JSON to Parquet, removes development-environment '
            'events, and maps tags_str into 8 manuscript classes (86,689 events).',
            '#0F766E',
            '#E7F6F3',
        ),
        (
            '3',
            'Leakage-controlled feature engineering',
            'Chronological 70/30 split; StandardScaler, CatBoostEncoder, and the >0.99 '
            'correlation filter are fit on TRAIN only. Final matrix: 57 features.',
            '#7C3AED',
            '#F1EAFF',
        ),
        (
            '4',
            'Model evaluation',
            'train_models.py evaluates Linear SVM, Random Forest, Gradient Boosting, and '
            'DNN; rule_baseline.py evaluates the rule baseline on the same held-out test split.',
            '#D97706',
            '#FFF4E6',
        ),
        (
            '5',
            'Direct reproducibility outputs',
            'repro/artifacts/ stores matrices and meta.json; repro/results*/ stores metrics, '
            'predictions, confusion matrices, and DNN history; repro/figures/ stores PNGs.',
            '#16A34A',
            '#EAF8EF',
        ),
    ]

    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    ax.text(
        0.5,
        0.965,
        'Actual submitted Linux audit-log ML pipeline',
        ha='center',
        va='top',
        fontsize=15,
        fontweight='bold',
        color='#111827',
    )
    ax.text(
        0.5,
        0.915,
        'The submitted code writes reproducibility artifacts directly to local folders.',
        ha='center',
        va='top',
        fontsize=9,
        color='#4B5563',
    )

    x = 0.055
    w = 0.89
    h = 0.125
    ys = [0.755, 0.585, 0.415, 0.245, 0.075]
    for idx, ((num, title, body, accent, fill), y) in enumerate(zip(stages, ys)):
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.012,rounding_size=0.018',
            linewidth=1.2,
            edgecolor=accent,
            facecolor=fill,
        )
        ax.add_patch(rect)
        badge = FancyBboxPatch(
            (x + 0.018, y + 0.025),
            0.064,
            h - 0.05,
            boxstyle='round,pad=0.006,rounding_size=0.012',
            linewidth=0,
            facecolor=accent,
        )
        ax.add_patch(badge)
        ax.text(
            x + 0.05,
            y + h / 2,
            num,
            ha='center',
            va='center',
            fontsize=15,
            color='white',
            fontweight='bold',
        )
        ax.text(
            x + 0.105,
            y + h - 0.031,
            title,
            ha='left',
            va='top',
            fontsize=10,
            fontweight='bold',
            color='#111827',
        )
        ax.text(
            x + 0.105,
            y + h - 0.064,
            textwrap.fill(body, width=118),
            ha='left',
            va='top',
            fontsize=8.2,
            color='#374151',
            linespacing=1.18,
        )
        if idx < len(ys) - 1:
            ax.annotate(
                '',
                xy=(0.5, ys[idx + 1] + h + 0.018),
                xytext=(0.5, y - 0.012),
                arrowprops=dict(arrowstyle='-|>', color='#64748B', lw=1.35),
            )

    ax.text(
        0.5,
        0.023,
        'The artifact folders provide the audit trail for each reproduced run.',
        ha='center',
        va='bottom',
        fontsize=8.2,
        color='#475569',
    )
    plt.savefig(os.path.join(FIG_DIR, 'figure3_actual_pipeline_direct_outputs.png'), dpi=240)
    plt.close(fig)


plot_actual_pipeline()

# ---------------------------------------------------------------------------
# 1. Class distribution bar chart
# ---------------------------------------------------------------------------
dist = meta['label_distribution_total']
labels = sorted(dist, key=lambda k: -dist[k])
counts = [dist[k] for k in labels]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(range(len(labels)), counts, color='#4C72B0')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Number of events')
ax.set_title(f'Class distribution (n = {sum(counts):,} events)')
for i, c in enumerate(counts):
    ax.text(i, c + max(counts) * 0.01, f'{c:,}', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'class_distribution.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Confusion matrices (Protocol A)
# ---------------------------------------------------------------------------
short_labels = [c.replace('_System_Network_Configuration_Discovery', '\n(NetConfig)')
                 .replace('_System_Information_Discovery', '\n(SysInfo)')
                 .replace('_Account_Discovery', '\n(AcctDisc)')
                 .replace('_Seuid_and_Setgid', '\n(SetUID)')
                for c in classes]

def plot_confmat(model_name, res, suffix=''):
    cm = np.array(res['confusion_matrix'], dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    acc = res['accuracy']
    f1m = res['macro_f1']
    title_suffix = ' (identifier columns dropped)' if suffix else ''
    ax.set_title(f'{model_name}{title_suffix} - Confusion Matrix (row-normalised)\n'
                  f'accuracy={acc:.2f}%  macro-F1={f1m:.2f}%')
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_norm[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if val > 0.5 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'confmat_{model_name}{suffix}.png'), dpi=200)
    plt.close(fig)


for model_name in metrics:
    plot_confmat(model_name, metrics[model_name])

# Protocol B (identifier-column ablation) confusion matrices, if available
ablation_path = os.path.join('results_ablation', 'metrics.json')
if os.path.exists(ablation_path):
    with open(ablation_path) as f:
        metrics_ablation = json.load(f)
    for model_name in metrics_ablation:
        plot_confmat(model_name, metrics_ablation[model_name], suffix='_ablation')

# ---------------------------------------------------------------------------
# 3. Per-class precision/recall/F1 grouped bars, one subplot per model
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)
if len(metrics) == 1:
    axes = [axes]

x = np.arange(n_classes)
width = 0.25

for ax, (model_name, res) in zip(axes, metrics.items()):
    report = res['classification_report']
    prec = [report[c]['precision'] for c in classes]
    rec = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    ax.bar(x - width, prec, width, label='Precision', color='#4C72B0')
    ax.bar(x, rec, width, label='Recall', color='#DD8452')
    ax.bar(x + width, f1, width, label='F1', color='#55A868')
    ax.set_title(model_name)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower right')

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
fig.suptitle('Per-class precision / recall / F1 (Protocol A, time-based test split)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'per_class_prf.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. CV boxplot (only if results_cv/cv_metrics.json exists)
# ---------------------------------------------------------------------------
cv_path = os.path.join(CV_DIR, 'cv_metrics.json')
if os.path.exists(cv_path):
    with open(cv_path) as f:
        cv = json.load(f)
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [cv[name]['macro_f1_folds'] for name in cv]
    ax.boxplot(data, tick_labels=list(cv.keys()))
    ax.set_ylabel('Macro-F1 (%)')
    ax.set_title('Repeated stratified 5-fold CV (3 repeats) - macro-F1')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cv_boxplot.png'), dpi=200)
    plt.close(fig)
    print('Wrote cv_boxplot.png')
else:
    print(f'Skipping CV boxplot - {cv_path} not found yet')

# ---------------------------------------------------------------------------
# 5. Model comparison bar chart (Protocol A + rule baseline)
# ---------------------------------------------------------------------------
RULE_DIR = 'results_rule_baseline'
rule_path = os.path.join(RULE_DIR, 'metrics.json')
if os.path.exists(rule_path):
    with open(rule_path) as f:
        rule_metrics = json.load(f)
    model_order = ['RandomForest', 'GradientBoosting', 'SVM', 'DNN']
    display_names = ['Random Forest', 'Gradient Boosting', 'Linear SVM', 'DNN', 'Rule-based\nbaseline']
    accs = [metrics[m]['accuracy'] for m in model_order] + [rule_metrics['accuracy']]
    f1s = [metrics[m]['macro_f1'] for m in model_order] + [rule_metrics['macro_f1']]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(display_names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, accs, width, label='Accuracy')
    bars2 = ax.bar(x + width / 2, f1s, width, label='Macro-F1')
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                     f'{b.get_height():.2f}', ha='center', va='bottom', rotation=90, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel('Percent')
    ax.set_ylim(0, 110)
    ax.set_title('Figure 9: Model Comparison - Test Accuracy and Macro-F1\n'
                 '(Held-out test set, time-based split; Protocol A, 57 features)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'), dpi=200)
    plt.close(fig)
    print('Wrote model_comparison.png')
else:
    print(f'Skipping model_comparison.png - {rule_path} not found yet')

print('Figures written to', FIG_DIR)
