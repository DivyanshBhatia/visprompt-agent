#!/usr/bin/env python3
"""Generate all paper figures from experimental data.

Creates publication-quality matplotlib figures for:
1. Cost vs Accuracy tradeoff across LLMs (Figure 1)
2. Few-shot comparison (Figure 2)
3. Description scaling curves (Figure 3)
4. Retrieval gains bar chart (Figure 4)
5. Weight ablation curves (Figure 5)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Publication style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'ours': '#2196F3',
    'templates': '#757575',
    'gpt4o': '#10A37F',
    'gpt52': '#FF6B35',
    'sonnet4': '#7B68EE',
    'opus45': '#E91E63',
    'fewshot': '#FF9800',
    'accent1': '#4CAF50',
    'accent2': '#9C27B0',
}

output_dir = "/mnt/user-data/outputs/figures"

import os
os.makedirs(output_dir, exist_ok=True)


# ================================================================
# FIGURE 1: Cost vs Accuracy across LLMs (cross-ablation data)
# ================================================================
def fig1_cost_vs_accuracy():
    """Scatter plot: LLM cost vs best accuracy improvement, per dataset."""
    
    # Data from cross-ablation results (best Δ per LLM per dataset, averaged across backbones)
    # Format: {llm: {dataset: (avg_cost, avg_best_delta)}}
    data = {
        'GPT-4o': {
            'CIFAR-100': (0.20, 0.68),
            'Flowers102': (0.18, 1.85),
            'Oxford Pets': (0.19, 1.10),
            'DTD': (0.10, 2.22),
        },
        'GPT-5.2': {
            'CIFAR-100': (0.67, 0.35),
            'Flowers102': (0.77, 2.84),
            'Oxford Pets': (0.68, 1.59),
            'DTD': (0.31, 7.58),
        },
        'Claude\nSonnet 4': {
            'CIFAR-100': (0.22, 0.45),
            'Flowers102': (0.22, 0.57),
            'Oxford Pets': (0.22, 1.43),
            'DTD': (0.19, 3.61),
        },
        'Claude\nOpus 4.5': {
            'CIFAR-100': (0.35, 0.87),
            'Flowers102': (0.34, 4.70),
            'Oxford Pets': (0.34, 2.89),
            'DTD': (0.33, 5.17),
        },
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: Cost vs Avg Delta (bubble per LLM, size = n_datasets_won)
    ax = axes[0]
    llm_colors = {'GPT-4o': COLORS['gpt4o'], 'GPT-5.2': COLORS['gpt52'],
                  'Claude\nSonnet 4': COLORS['sonnet4'], 'Claude\nOpus 4.5': COLORS['opus45']}
    llm_markers = {'GPT-4o': 'o', 'GPT-5.2': 's', 'Claude\nSonnet 4': '^', 'Claude\nOpus 4.5': 'D'}
    
    for llm, datasets in data.items():
        costs = [v[0] for v in datasets.values()]
        deltas = [v[1] for v in datasets.values()]
        avg_cost = np.mean(costs)
        avg_delta = np.mean(deltas)
        
        # Individual dataset points (smaller, transparent)
        for ds_name, (cost, delta) in datasets.items():
            ax.scatter(cost, delta, c=llm_colors[llm], marker=llm_markers[llm],
                      s=40, alpha=0.3, edgecolors='none')
        
        # Average point (larger, with label)
        ax.scatter(avg_cost, avg_delta, c=llm_colors[llm], marker=llm_markers[llm],
                  s=150, alpha=0.9, edgecolors='white', linewidths=1.5,
                  label=f'{llm} (avg ${avg_cost:.2f})', zorder=5)
    
    ax.set_xlabel('LLM Cost per Dataset (USD)')
    ax.set_ylabel('Avg Best Classification $\\Delta$ (%)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_title('(a) Cost vs Accuracy Improvement')
    
    # Right: Per-dataset breakdown as grouped bars
    ax = axes[1]
    datasets = ['CIFAR-100', 'Flowers102', 'Oxford Pets', 'DTD']
    llms = ['GPT-4o', 'GPT-5.2', 'Claude\nSonnet 4', 'Claude\nOpus 4.5']
    x = np.arange(len(datasets))
    width = 0.18
    
    for i, llm in enumerate(llms):
        deltas = [data[llm][ds][1] for ds in datasets]
        bars = ax.bar(x + (i - 1.5) * width, deltas, width,
                     label=llm.replace('\n', ' '), color=llm_colors[llm], alpha=0.85)
    
    ax.set_ylabel('Best Classification $\\Delta$ (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.set_title('(b) Improvement by Dataset & LLM')
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_cost_vs_accuracy.pdf')
    plt.savefig(f'{output_dir}/fig1_cost_vs_accuracy.png')
    plt.close()
    print("  Fig 1: Cost vs Accuracy ✓")


# ================================================================
# FIGURE 2: Few-shot comparison
# ================================================================
def fig2_fewshot():
    """Line plot: k-shot accuracy vs our zero-shot horizontal line."""
    
    fewshot_data = {
        'Oxford Pets': {
            'templates': 90.24, 'ours': 92.04,
            'k': [1, 2, 4, 8, 16],
            'acc': [54.82, 67.26, 78.52, 82.47, 87.21],
        },
        'CIFAR-100': {
            'templates': 73.99, 'ours': 74.77,
            'k': [1, 2, 4, 8, 16],
            'acc': [35.39, 55.39, 66.72, 73.99, 74.77],  # approx
        },
        'DTD': {
            'templates': 52.34, 'ours': 57.55,
            'k': [1, 2, 4, 8, 16],
            'acc': [37.60, 46.12, 54.31, 59.16, 62.55],
        },
        'Flowers102': {
            'templates': 68.48, 'ours': 74.48,
            'k': [1, 2, 4, 8, 16],
            'acc': [78.98, 85.27, 88.78, 91.00, 91.40],
        },
    }
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    
    dataset_order = ['Oxford Pets', 'DTD', 'CIFAR-100', 'Flowers102']
    crossover = ['>16', '8', '>16', '1']
    
    for idx, (ds, cross_k) in enumerate(zip(dataset_order, crossover)):
        ax = axes[idx]
        d = fewshot_data[ds]
        
        # k-shot line
        ax.plot(d['k'], d['acc'], 'o-', color=COLORS['fewshot'],
                linewidth=2, markersize=6, label='k-shot probe', zorder=3)
        
        # Our zero-shot (horizontal line)
        ax.axhline(y=d['ours'], color=COLORS['ours'], linewidth=2,
                   linestyle='--', label=f'Ours (zero-shot)', zorder=4)
        
        # Templates baseline (horizontal line, gray)
        ax.axhline(y=d['templates'], color=COLORS['templates'], linewidth=1.5,
                   linestyle=':', label='Templates', alpha=0.7)
        
        # Fill area where we win
        for i in range(len(d['k'])):
            if d['acc'][i] < d['ours']:
                if i == 0:
                    ax.axvspan(0.5, d['k'][i] + 0.5, alpha=0.08, color=COLORS['ours'])
                else:
                    ax.axvspan(d['k'][i-1] + 0.5, d['k'][i] + 0.5, alpha=0.08, color=COLORS['ours'])
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(d['k'])
        ax.set_xticklabels([str(k) for k in d['k']])
        ax.set_xlabel('k (shots per class)')
        if idx == 0:
            ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{ds}\ncrossover: {cross_k}-shot')
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_fewshot.pdf')
    plt.savefig(f'{output_dir}/fig2_fewshot.png')
    plt.close()
    print("  Fig 2: Few-shot comparison ✓")


# ================================================================
# FIGURE 3: Description scaling
# ================================================================
def fig3_desc_scaling():
    """Line plot: accuracy vs number of descriptions per class."""
    
    scaling_data = {
        'CIFAR-100\n(70/30)': {
            'n': [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
            'acc': [74.70, 73.54, 74.74, 75.20, 75.37, 75.59, 75.82, 75.79, 75.78, 75.64],
            'color': COLORS['gpt4o'],
            'style': '-',
        },
        'DTD\n(40/60)': {
            'n': [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
            'acc': [52.82, 52.39, 54.68, 55.48, 56.28, 56.17, 55.48, 56.38, 56.60, 56.97],
            'color': COLORS['opus45'],
            'style': '-',
        },
        'Food101\n(40/60)': {
            'n': [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
            'acc': [90.84, 90.48, 90.93, 91.34, 91.29, 91.53, 91.53, 91.53, 91.68, 91.66],
            'color': COLORS['accent1'],
            'style': '-',
        },
        'Flowers102\n(0/100)': {
            'n': [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
            'acc': [68.09, 70.60, 74.68, 75.12, 73.48, 73.18, 72.42, 72.42, 72.95, 73.36],
            'color': COLORS['ours'],
            'style': '--',
        },
        'Pets\n(0/100)': {
            'n': [0, 1, 2, 3, 4, 5, 6, 8, 10, 15],
            'acc': [90.16, 89.70, 88.72, 90.19, 90.81, 89.53, 89.62, 88.63, 88.69, 88.58],
            'color': COLORS['accent2'],
            'style': '--',
        },
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    for label, d in scaling_data.items():
        # Normalize to delta vs baseline
        baseline = d['acc'][0]
        deltas = [a - baseline for a in d['acc']]
        
        ax.plot(d['n'], deltas, d['style'], color=d['color'],
                linewidth=2, marker='o', markersize=5, label=label)
    
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xlabel('Number of LLM Descriptions per Class')
    ax.set_ylabel('$\\Delta$ Accuracy vs Templates-Only (%)')
    ax.set_title('Description Scaling: Anchored (solid) vs Unanchored (dashed)')
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    
    # Annotate the key insight
    ax.annotate('Anchored: more = better',
                xy=(10, 1.0), fontsize=9, color='gray', style='italic')
    ax.annotate('Unanchored: peak at 2-3',
                xy=(6, -1.0), fontsize=9, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_desc_scaling.pdf')
    plt.savefig(f'{output_dir}/fig3_desc_scaling.png')
    plt.close()
    print("  Fig 3: Description scaling ✓")


# ================================================================
# FIGURE 4: Retrieval gains (10 datasets, all positive)
# ================================================================
def fig4_retrieval():
    """Horizontal bar chart: retrieval mAP improvement per dataset."""
    
    retrieval = [
        ('DTD', 43.10, 47.36, '40/60'),
        ('EuroSAT', 44.66, 48.56, '0/100'),
        ('Flowers102', 70.48, 72.99, '40/60'),
        ('CIFAR-100', 66.18, 67.42, '40/60'),
        ('Country211', 14.97, 16.10, '55/45'),
        ('Food101', 90.24, 91.15, '55/45'),
        ('FGVC-Aircraft', 24.33, 25.20, '40/60'),
        ('CIFAR-10', 91.55, 92.26, '55/45'),
        ('Caltech101', 90.33, 90.95, '40/60'),
        ('Oxford Pets', 87.79, 87.81, '95/5'),
    ]
    
    # Sort by delta
    retrieval.sort(key=lambda x: x[2] - x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = [r[0] for r in retrieval]
    deltas = [r[2] - r[1] for r in retrieval]
    weights = [r[3] for r in retrieval]
    
    y = np.arange(len(names))
    bars = ax.barh(y, deltas, color=COLORS['ours'], alpha=0.85, height=0.6)
    
    # Add value labels
    for i, (bar, delta, w) in enumerate(zip(bars, deltas, weights)):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'+{delta:.2f}% ({w})', va='center', fontsize=9)
    
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('$\\Delta$ mAP (%)')
    ax.set_title('Retrieval Improvement: 10/10 Datasets Positive')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_retrieval.pdf')
    plt.savefig(f'{output_dir}/fig4_retrieval.png')
    plt.close()
    print("  Fig 4: Retrieval gains ✓")


# ================================================================
# FIGURE 5: Weight ablation curves (multiple datasets)
# ================================================================
def fig5_weight_ablation():
    """Line plot: accuracy vs weight ratio for multiple datasets."""
    
    weights = [0, 15, 30, 45, 60, 80, 100]  # desc weight %
    
    ablation_data = {
        'CIFAR-100': [74.70, 75.52, 75.89, 75.90, 75.66, 75.00, 74.03],
        'Flowers102': [68.09, 69.07, 70.21, 71.20, 72.78, 73.93, 75.49],
        'DTD': [52.82, 54.57, 54.73, 54.10, 53.03, 52.61, 51.81],
        'Oxford Pets': [90.16, 91.41, 92.08, 92.48, 92.26, 91.66, 90.27],
        'Food101': [90.84, 91.23, 91.53, 91.77, 91.73, 91.40, 90.74],
    }
    
    colors = {
        'CIFAR-100': COLORS['gpt4o'],
        'Flowers102': COLORS['ours'],
        'DTD': COLORS['opus45'],
        'Oxford Pets': COLORS['accent2'],
        'Food101': COLORS['accent1'],
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for ds, accs in ablation_data.items():
        ax.plot(weights, accs, 'o-', color=colors[ds], linewidth=2,
                markersize=5, label=ds)
        # Mark the peak
        peak_idx = np.argmax(accs)
        ax.plot(weights[peak_idx], accs[peak_idx], '*', color=colors[ds],
                markersize=14, zorder=5)
    
    ax.set_xlabel('Description Weight (%)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Weight Ablation: Optimal Ratio is Domain-Dependent')
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xticks(weights)
    ax.set_xticklabels(['0\n(templates\nonly)', '15', '30', '45', '60', '80',
                        '100\n(desc\nonly)'])
    
    # Annotate domain insight
    ax.annotate('Generic: prefer templates',
                xy=(20, 76.1), fontsize=8, color='gray', style='italic',
                ha='center')
    ax.annotate('Fine-grained:\nprefer descriptions',
                xy=(85, 75.0), fontsize=8, color='gray', style='italic',
                ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_weight_ablation.pdf')
    plt.savefig(f'{output_dir}/fig5_weight_ablation.png')
    plt.close()
    print("  Fig 5: Weight ablation ✓")


# ================================================================
# FIGURE 6: Classification results overview (10 datasets)
# ================================================================
def fig6_classification_overview():
    """Grouped bar chart: Ours vs best baseline across 10 datasets."""
    
    results = [
        ('DTD', 54.47, 57.66, 'CuPL+e'),
        ('Oxford Pets', 90.27, 92.48, 'CuPL+e'),
        ('Flowers102', 74.21, 75.54, 'DCLIP'),
        ('Country211', 23.84, 24.20, 'Waffle'),
        ('CIFAR-100', 75.71, 75.81, 'CLIP-E'),
        ('Food101', 91.84, 91.77, 'DCLIP'),
        ('CIFAR-10', 95.15, 94.46, 'Frolic'),
        ('Caltech101', 90.14, 89.45, 'CuPL'),
        ('FGVC-Aircraft', 27.24, 26.49, 'DCLIP'),
        ('EuroSAT', 48.06, 37.30, 'CLIP-E'),
    ]
    
    # Sort by our improvement
    results.sort(key=lambda x: x[2] - x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    names = [r[0] for r in results]
    best_baseline = [r[1] for r in results]
    ours = [r[2] for r in results]
    baseline_names = [r[3] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_baseline, width, label='Best Baseline',
                   color=COLORS['templates'], alpha=0.7)
    bars2 = ax.bar(x + width/2, ours, width, label='Ours',
                   color=COLORS['ours'], alpha=0.85)
    
    # Add delta labels
    for i, (b, o) in enumerate(zip(best_baseline, ours)):
        delta = o - b
        color = '#2196F3' if delta > 0 else '#F44336'
        ax.text(x[i] + width/2, o + 0.5, f'{delta:+.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Classification: Ours vs Best Baseline (ViT-L/14)')
    ax.legend()
    
    # Draw win/loss divider
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_classification.pdf')
    plt.savefig(f'{output_dir}/fig6_classification.png')
    plt.close()
    print("  Fig 6: Classification overview ✓")


# ================================================================
# FIGURE 7: Action Recognition (UCF-101)
# ================================================================
def fig7_action_recognition():
    """Bar chart: UCF-101 baselines vs our method."""
    
    methods = [
        ('WaffleCLIP', 69.38),
        ('Class name', 69.96),
        ('CuPL (desc)', 69.96),
        ('DCLIP', 71.78),
        ('CLIP-Enhance', 71.78),
        ('Templates (90)', 72.25),
        ('CuPL+ens', 72.25),
        ('Single tmpl', 72.48),
        ('Ours (55/45)', 75.88),
    ]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    names = [m[0] for m in methods]
    accs = [m[1] for m in methods]
    colors = [COLORS['ours'] if 'Ours' in n else COLORS['templates'] for n in names]
    
    bars = ax.barh(range(len(names)), accs, color=colors, alpha=0.85, height=0.6)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=9)
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('UCF-101 Zero-Shot Action Recognition (ViT-L/14)')
    ax.set_xlim(68, 78)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig7_action_recognition.pdf')
    plt.savefig(f'{output_dir}/fig7_action_recognition.png')
    plt.close()
    print("  Fig 7: Action recognition ✓")


# ================================================================
# Generate all figures
# ================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Output: {output_dir}/\n")
    
    fig1_cost_vs_accuracy()
    fig2_fewshot()
    fig3_desc_scaling()
    fig4_retrieval()
    fig5_weight_ablation()
    fig6_classification_overview()
    fig7_action_recognition()
    
    print(f"\nAll figures saved to {output_dir}/")
    print("PDF versions for LaTeX, PNG versions for preview.")
