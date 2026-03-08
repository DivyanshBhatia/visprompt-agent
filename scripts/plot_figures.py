#!/usr/bin/env python3
"""Generate all paper figures from experiment results.

Run on Colab after experiments. Generates:
  - Figure 1: Cost vs Accuracy tradeoff across LLMs
  - Figure 2: Weight ablation curves (multiple datasets)
  - Figure 3: Few-shot comparison
  - Figure 4: Description scaling curves
  - Figure 5: Retrieval improvements bar chart
  - Figure 6: Per-class improvement scatter

Usage:
    python scripts/plot_figures.py --results-dir experiments/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("pip install matplotlib")
    sys.exit(1)

# Consistent style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'GPT-4o': '#1f77b4',
    'GPT-5.2': '#ff7f0e',
    'Claude-Sonnet-4': '#2ca02c',
    'Claude-Opus-4.5': '#d62728',
}

DATASET_COLORS = {
    'cifar100': '#1f77b4',
    'flowers102': '#ff7f0e',
    'dtd': '#2ca02c',
    'oxford_pets': '#d62728',
    'food101': '#9467bd',
}


def plot_cost_vs_accuracy(results_dir, output_dir):
    """Figure 1: Cost vs Accuracy tradeoff across LLMs.
    
    Uses cross-ablation results to show cost-accuracy Pareto frontier.
    """
    cross_file = Path(results_dir) / "cross_ablation" / "cross_ablation_results.json"
    if not cross_file.exists():
        print(f"  Skipping cost_vs_accuracy: {cross_file} not found")
        return
    
    with open(cross_file) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), sharey=False)
    if len(data) == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, sorted(data.keys())):
        llm_costs = {}  # llm -> [costs]
        llm_deltas = {}  # llm -> [deltas]
        
        for backbone in data[dataset]:
            templates_acc = data[dataset][backbone].get('templates_only', {}).get('classification', 0)
            
            for llm in data[dataset][backbone]:
                if llm == 'templates_only':
                    continue
                entry = data[dataset][backbone][llm]
                if not isinstance(entry, dict) or 'cost' not in entry:
                    continue
                
                cost = entry['cost']
                weights = entry.get('weights', [])
                if not weights:
                    continue
                best_cls = max(w.get('classification', 0) for w in weights)
                delta = (best_cls - templates_acc) * 100
                
                if llm not in llm_costs:
                    llm_costs[llm] = []
                    llm_deltas[llm] = []
                llm_costs[llm].append(cost)
                llm_deltas[llm].append(delta)
        
        for llm in sorted(llm_costs.keys()):
            avg_cost = np.mean(llm_costs[llm])
            avg_delta = np.mean(llm_deltas[llm])
            color = COLORS.get(llm, '#333333')
            ax.scatter(avg_cost, avg_delta, c=color, s=100, zorder=5, edgecolors='black', linewidths=0.5)
            ax.annotate(llm.replace('Claude-', ''), (avg_cost, avg_delta),
                       textcoords="offset points", xytext=(5, 5), fontsize=8)
            
            # Also plot individual backbone points (smaller)
            for c, d in zip(llm_costs[llm], llm_deltas[llm]):
                ax.scatter(c, d, c=color, s=20, alpha=0.3, zorder=3)
        
        ax.set_xlabel('Cost (USD)')
        ax.set_ylabel('Δ Accuracy (%)')
        ax.set_title(dataset.upper())
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('Cost vs Accuracy Improvement by LLM', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_cost_vs_accuracy.pdf')
    plt.savefig(output_dir / 'fig1_cost_vs_accuracy.png')
    print(f"  Saved fig1_cost_vs_accuracy")
    plt.close()


def plot_fewshot_comparison(results_dir, output_dir):
    """Figure 3: Few-shot comparison — accuracy vs k-shots."""
    fewshot_file = Path(results_dir) / "fewshot" / "fewshot_comparison.json"
    if not fewshot_file.exists():
        print(f"  Skipping fewshot: {fewshot_file} not found")
        return
    
    with open(fewshot_file) as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4))
    if len(data) == 1:
        axes = [axes]
    
    for ax, dataset in zip(axes, sorted(data.keys())):
        res = data[dataset]
        
        # Few-shot points
        k_values = sorted(int(k) for k in res['fewshot'].keys())
        means = [res['fewshot'][str(k)]['mean'] * 100 for k in k_values]
        stds = [res['fewshot'][str(k)]['std'] * 100 for k in k_values]
        
        ax.errorbar(k_values, means, yerr=stds, marker='o', markersize=6,
                    capsize=4, color='#1f77b4', label=f'{k_values[0]}-{k_values[-1]} shot probe',
                    linewidth=2, zorder=3)
        
        # Templates horizontal line
        templates = res['zero_shot_templates'] * 100
        ax.axhline(y=templates, color='gray', linestyle='--', alpha=0.6,
                   label=f'Templates ({templates:.1f}%)')
        
        # Ours horizontal line
        ours = res['zero_shot_ours'] * 100
        ax.axhline(y=ours, color='#d62728', linestyle='-', linewidth=2.5,
                   label=f'Ours ({ours:.1f}%)', zorder=4)
        
        # Fill region where ours beats probe
        ax.fill_between(k_values, [ours]*len(k_values), means,
                        where=[m < ours for m in means],
                        alpha=0.15, color='#d62728')
        
        ax.set_xlabel('k (shots per class)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(dataset.replace('_', ' ').title())
        ax.set_xticks(k_values)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.2)
    
    fig.suptitle('Zero-Shot Ours vs k-Shot Linear Probe', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_fewshot_comparison.pdf')
    plt.savefig(output_dir / 'fig3_fewshot_comparison.png')
    print(f"  Saved fig3_fewshot_comparison")
    plt.close()


def plot_description_scaling(results_dir, output_dir):
    """Figure 4: Description scaling — accuracy vs N descriptions."""
    scaling_dir = Path(results_dir) / "desc_scaling"
    if not scaling_dir.exists():
        print(f"  Skipping desc_scaling: {scaling_dir} not found")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    for scaling_file in sorted(scaling_dir.glob("desc_scaling_*.json")):
        with open(scaling_file) as f:
            data = json.load(f)
        
        dataset = data['dataset']
        baseline = data['baseline_accuracy'] * 100
        weight_label = f"{data['base_weight']:.0%}/{data['desc_weight']:.0%}"
        
        n_descs = [0] + [r['n_descriptions'] for r in data['scaling_results']]
        accs = [baseline] + [r['accuracy'] * 100 for r in data['scaling_results']]
        
        color = DATASET_COLORS.get(dataset, '#333333')
        label = f"{dataset} ({weight_label})"
        ax.plot(n_descs, accs, marker='o', markersize=5, color=color,
                linewidth=2, label=label)
    
    ax.set_xlabel('Number of Descriptions per Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Description Count')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 8, 10, 15])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_desc_scaling.pdf')
    plt.savefig(output_dir / 'fig4_desc_scaling.png')
    print(f"  Saved fig4_desc_scaling")
    plt.close()


def plot_retrieval_bar(output_dir):
    """Figure 5: Retrieval improvements across 10 datasets."""
    # Hardcoded from our results (sorted by delta)
    datasets = [
        ('DTD', 43.10, 47.36, 4.25),
        ('EuroSAT', 44.66, 48.56, 3.90),
        ('Flowers102', 70.48, 72.99, 2.52),
        ('CIFAR-100', 66.18, 67.42, 1.24),
        ('Country211', 14.97, 16.10, 1.14),
        ('Food101', 90.24, 91.15, 0.90),
        ('FGVC-Airc.', 24.33, 25.20, 0.88),
        ('CIFAR-10', 91.55, 92.26, 0.71),
        ('Caltech101', 90.33, 90.95, 0.62),
        ('Oxford Pets', 87.79, 87.81, 0.02),
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    
    names = [d[0] for d in datasets]
    deltas = [d[3] for d in datasets]
    
    bars = ax.barh(range(len(names)), deltas, color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel('Δ mAP (%)')
    ax.set_title('Retrieval Improvement: 10/10 Positive')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2)
    
    # Add value labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        ax.text(delta + 0.05, i, f'+{delta:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_retrieval_bar.pdf')
    plt.savefig(output_dir / 'fig5_retrieval_bar.png')
    print(f"  Saved fig5_retrieval_bar")
    plt.close()


def plot_classification_overview(output_dir):
    """Figure 6: Classification results overview — grouped bar chart."""
    # From our 10-dataset results
    datasets = ['CIFAR-10', 'CIFAR-100', 'Flowers', 'DTD', 'Food101',
                'Pets', 'Caltech', 'Aircraft', 'EuroSAT', 'Country']
    ours =     [94.46, 75.81, 75.54, 57.66, 91.77, 92.48, 89.45, 26.49, 37.30, 24.20]
    best_bl =  [95.15, 75.71, 74.21, 54.47, 91.84, 90.27, 90.14, 27.24, 48.06, 23.84]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_bl, width, label='Best Baseline',
                   color='#aec7e8', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, ours, width, label='Ours',
                   color='#d62728', edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Mark wins/losses
    for i, (o, b) in enumerate(zip(ours, best_bl)):
        if o > b + 0.1:
            ax.text(i + width/2, o + 0.5, '✓', ha='center', fontsize=10, color='green')
        elif b > o + 0.1:
            ax.text(i + width/2, o + 0.5, '✗', ha='center', fontsize=10, color='red')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Zero-Shot Classification: Ours vs Best Baseline (ViT-L/14)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_classification_overview.pdf')
    plt.savefig(output_dir / 'fig6_classification_overview.png')
    print(f"  Saved fig6_classification_overview")
    plt.close()


def plot_llm_heatmap(results_dir, output_dir):
    """Figure 7: LLM × Backbone heatmap for one dataset."""
    cross_file = Path(results_dir) / "cross_ablation" / "cross_ablation_results.json"
    if not cross_file.exists():
        print(f"  Skipping heatmap: {cross_file} not found")
        return
    
    with open(cross_file) as f:
        data = json.load(f)
    
    # Pick the most interesting dataset (flowers102 or dtd)
    for dataset in ['flowers102', 'dtd', 'cifar100', 'oxford_pets']:
        if dataset not in data:
            continue
        
        backbones = sorted(data[dataset].keys())
        llms = []
        for bb in backbones:
            for llm in data[dataset][bb]:
                if llm != 'templates_only' and llm not in llms:
                    llms.append(llm)
        
        if not llms:
            continue
        
        # Build delta matrix
        matrix = np.zeros((len(llms), len(backbones)))
        for j, bb in enumerate(backbones):
            templates_acc = data[dataset][bb].get('templates_only', {}).get('classification', 0)
            for i, llm in enumerate(llms):
                entry = data[dataset][bb].get(llm, {})
                if isinstance(entry, dict) and 'weights' in entry:
                    best = max(w.get('classification', 0) for w in entry['weights'])
                    matrix[i, j] = (best - templates_acc) * 100
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
        
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=np.max(matrix) + 0.5)
        
        # Labels
        bb_short = [b.replace('MetaCLIP-', 'MC-').replace('SigLIP-', 'SL-') for b in backbones]
        ax.set_xticks(range(len(backbones)))
        ax.set_xticklabels(bb_short, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(llms)))
        ax.set_yticklabels(llms, fontsize=9)
        
        # Cell values
        for i in range(len(llms)):
            for j in range(len(backbones)):
                val = matrix[i, j]
                color = 'white' if val > np.max(matrix) * 0.6 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center', fontsize=8, color=color)
        
        plt.colorbar(im, ax=ax, label='Δ Accuracy (%)', shrink=0.8)
        ax.set_title(f'LLM × Backbone Improvement: {dataset.upper()}')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'fig7_heatmap_{dataset}.pdf')
        plt.savefig(output_dir / f'fig7_heatmap_{dataset}.png')
        print(f"  Saved fig7_heatmap_{dataset}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", type=str, default="experiments/")
    parser.add_argument("--output-dir", type=str, default="figures/")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating paper figures...")
    print(f"  Results: {args.results_dir}")
    print(f"  Output: {output_dir}\n")
    
    plot_cost_vs_accuracy(args.results_dir, output_dir)
    plot_fewshot_comparison(args.results_dir, output_dir)
    plot_description_scaling(args.results_dir, output_dir)
    plot_retrieval_bar(output_dir)
    plot_classification_overview(output_dir)
    plot_llm_heatmap(args.results_dir, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
