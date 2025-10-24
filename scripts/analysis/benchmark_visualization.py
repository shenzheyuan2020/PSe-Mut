#!/usr/bin/env python3
"""
Advanced Benchmark Visualization Script - SCI Paper Style
Uses blue-white-gray color scheme to create bubble charts, radar charts, box plots, heatmaps and other advanced charts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Set matplotlib backend to avoid rendering issues
plt.switch_backend('Agg')

# Set SCI style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# Color scheme - blue-white-gray
COLORS = {
    # Green-centric palette to match the provided figure style
    'primary': '#2E7D32',      # Dark green (Material Green 800)
    'accent': '#81C784',       # Light green (Green 300)
    'green':  '#A5D6A7',       # Soft green (Green 200) for legacy references
    'siamese': '#4CAF50',      # Green (Green 500)
    'neutral': '#90A4AE',      # Blue Gray 300
    'light': '#ECEFF1',        # Blue Gray 50
    'background': '#FFFFFF',   # White
}

def load_results(results_path):
    """Load benchmark test results"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def augment_results_with_seed_files(results: dict, results_path: str):
    """Augment results dict using per-seed val_metrics.json when some methods (e.g., xgb) are missing.
    results_path should be .../outputs/benchmarks/all/summary_all_seeds.json.
    """
    try:
        base_dir = os.path.dirname(results_path)  # .../outputs/benchmarks/all
        seeds = results.get('seeds', [])
        per_seed = results.setdefault('per_seed', {})
        for seed in seeds:
            seed_dir = os.path.join(base_dir, f"run_seed{seed}")
            # tree family
            tree_metrics_path = os.path.join(seed_dir, 'tree', 'val_metrics.json')
            if os.path.exists(tree_metrics_path):
                try:
                    vm = json.load(open(tree_metrics_path, 'r', encoding='utf-8'))
                    tree = per_seed.setdefault(seed, {}).setdefault('tree', {})
                    for k, v in vm.items():
                        # accept only dict-like metrics with rmse
                        if isinstance(v, dict) and 'rmse' in v:
                            if not (isinstance(tree.get(k), dict) and 'rmse' in tree.get(k, {})):
                                tree[k] = v
                except Exception:
                    pass
    except Exception:
        pass

def create_performance_bubble_chart(results, save_path):
    """Create performance bubble chart - R² vs RMSE, bubble size represents MAE"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data - only select the best methods to avoid overly complex graphics
    methods_data = []

    # Traditional methods - only select the best
    best_traditional = {
        'ElasticNet': ('linear', 'elasticnet'),
        'SVR-RBF': ('svr', 'svr_rbf'),
        'RandomForest': ('tree', 'rf'),
        'MLP': ('mlp', 'mlp_sklearn'),
        'KernelRidge': ('kernel_knn', 'kernel_ridge_rbf')
    }

    for method_name, (category, submethod) in best_traditional.items():
        r2_vals = [results['per_seed'][seed][category][submethod]['r2'] for seed in results['seeds']]
        rmse_vals = [results['per_seed'][seed][category][submethod]['rmse'] for seed in results['seeds']]
        mae_vals = [results['per_seed'][seed][category][submethod]['mae'] for seed in results['seeds']]

        methods_data.append({
            'method': method_name,
            'type': 'Traditional ML',
            'r2_mean': np.mean(r2_vals),
            'rmse_mean': np.mean(rmse_vals),
            'mae_mean': np.mean(mae_vals)
        })

    # Siamese model data
    siamese_methods = ['base', 'no_exchange', 'diff_only']
    for method_name in siamese_methods:
        r2_vals = [results['per_seed'][seed]['model'][method_name]['r2'] for seed in results['seeds']]
        rmse_vals = [results['per_seed'][seed]['model'][method_name]['rmse'] for seed in results['seeds']]
        mae_vals = [results['per_seed'][seed]['model'][method_name]['mae'] for seed in results['seeds']]

        methods_data.append({
            'method': f'Siamese-{method_name}',
            'type': 'Siamese Models',
            'r2_mean': np.mean(r2_vals),
            'rmse_mean': np.mean(rmse_vals),
            'mae_mean': np.mean(mae_vals)
        })

    df = pd.DataFrame(methods_data)

    # Use seaborn to draw scatter plot
    colors = {'Traditional ML': COLORS['green'], 'Siamese Models': COLORS['siamese']}

    # Bubble size based on MAE (normalized)
    sizes = (df['mae_mean'] - df['mae_mean'].min()) / (df['mae_mean'].max() - df['mae_mean'].min())
    sizes = sizes * 500 + 100  # Scale to appropriate size

    for type_name in df['type'].unique():
        subset = df[df['type'] == type_name]
        ax.scatter(subset['r2_mean'], subset['rmse_mean'],
                  s=sizes[df['type'] == type_name], c=colors[type_name],
                  alpha=0.8, edgecolors='white', linewidth=2, label=type_name)

    # Add labels
    for _, row in df.iterrows():
        ax.annotate(row['method'].replace('Siamese-', ''),
                   (row['r2_mean'], row['rmse_mean']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.9, fontweight='bold')

    # Set axes
    ax.set_xlabel('Coefficient of Determination (R²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Root Mean Square Error (RMSE)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison\n(Bubble size represents MAE)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()

def create_radar_chart(results, save_path):
    """Create radar chart to show Siamese model ablation experiments"""
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Prepare data
    ablations = ['base', 'no_exchange', 'diff_only', 'mut_only', 'wt_only']
    metrics = ['r2', 'pearson', 'auc']

    # Calculate average metrics for each ablation experiment
    ablation_data = {}
    for ablation in ablations:
        scores = []
        for metric in metrics:
            # check existence against first available seed
            seed0 = results['seeds'][0]
            seed0_has = (seed0 in results['per_seed'] and
                         'model' in results['per_seed'][seed0] and
                         ablation in results['per_seed'][seed0]['model'] and
                         metric in results['per_seed'][seed0]['model'][ablation])
            if seed0_has:
                vals = []
                for seed in results['seeds']:
                    try:
                        vals.append(results['per_seed'][seed]['model'][ablation][metric])
                    except Exception:
                        continue
                scores.append(np.mean(vals) if vals else 0)
            else:
                scores.append(0)
        ablation_data[ablation] = {'scores': scores}

    # Radar chart parameters
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]  # Close the shape

    # Draw each ablation experiment
    colors = [COLORS['siamese'], COLORS['green'], '#6C757D', '#8EC6C5', '#AEC6CF']
    labels = ['Base', 'No Exchange', 'Diff Only', 'Mut Only', 'WT Only']

    for i, ablation in enumerate(ablations):
        scores = ablation_data[ablation]['scores'] + ablation_data[ablation]['scores'][:1]  # Close the shape
        ax.plot(angles, scores, 'o-', linewidth=2, label=labels[i], color=colors[i], alpha=0.8)
        ax.fill(angles, scores, alpha=0.1, color=colors[i])

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['R²', 'Pearson', 'AUC'])
    ax.set_ylim(0, 1)
    ax.set_title('Siamese Model Ablation Study\n(Radar Chart)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()

def create_rmse_boxplot_baselines(results, save_path, split: str = 'valid'):
    """Create RMSE-only box plot for traditional baselines across seeds.
    Includes a wider set of methods when available (robust to missing).
    """
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {}) if split == 'valid' else results.get('per_seed_test', {})

    # Expanded set of traditional methods by family
    families = {
        'linear': ['linear', 'ridge', 'elasticnet', 'lasso', 'bayesian_ridge'],
        'svr': ['svr_rbf'],
        'tree': ['rf', 'extratrees'],  # drop 'xgb'
        'mlp': ['mlp_sklearn'],
        'kernel_knn': ['kernel_ridge_rbf', 'knn_reg'],
    }
    # Pretty labels
    pretty = {
        'linear': 'Linear', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet', 'lasso': 'Lasso', 'bayesian_ridge': 'BayesianRidge',
        'svr_rbf': 'SVR-RBF',
        'rf': 'RandomForest', 'extratrees': 'ExtraTrees', 'xgb': 'XGBoost',
        'mlp_sklearn': 'MLP',
        'kernel_ridge_rbf': 'KernelRidge', 'knn_reg': 'KNN',
    }

    rows = []
    for seed in seeds:
        seed_dict = per_seed.get(seed, {})
        for category, methods in families.items():
            cat = seed_dict.get(category, {})
            for m in methods:
                entry = cat.get(m)
                if isinstance(entry, dict) and ('rmse' in entry):
                    rmse = entry['rmse']
                    # Filter out obvious placeholders like error-only xgb
                    if isinstance(rmse, (int, float)):
                        rows.append({'method': pretty.get(m, m), 'rmse': rmse, 'family': category})

    if len(rows) == 0:
        # 兜底：生成空图并提示
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        return

    df = pd.DataFrame(rows)
    # If test metrics available, overlay with a second color
    rows_te = []
    per_seed_te = results.get('per_seed_test', {})
    for seed in seeds:
        seed_dict = per_seed_te.get(seed, {})
        for category, methods in families.items():
            cat = seed_dict.get(category, {})
            for m in methods:
                entry = cat.get(m)
                if isinstance(entry, dict) and ('rmse' in entry) and isinstance(entry['rmse'], (int, float)):
                    rows_te.append({'method': pretty.get(m, m), 'rmse': entry['rmse'], 'split': 'test'})
    if rows_te:
        df['split'] = 'valid'
        df = pd.concat([df, pd.DataFrame(rows_te)], ignore_index=True)
    # Also include our model (Psemut, formerly Base) in baseline chart for comparison
    if seeds and 'model' in per_seed.get(seeds[0], {}):
        base_vals = []
        for seed in seeds:
            v = per_seed.get(seed, {}).get('model', {}).get('base', {}).get('rmse')
            if v is not None:
                df = pd.concat([df, pd.DataFrame([{'method': 'Psemut', 'rmse': v, 'family': 'model'}])], ignore_index=True)

    # Sort methods by median RMSE ascending for visual clarity (left to right better)
    med = df.groupby('method')['rmse'].median().sort_values()
    order = list(med.index)

    # Palette: highlight Siamese-Base
    # Paper-like palette (blue/green/gray), Base highlighted in deep blue
    uniq_methods = list(dict.fromkeys(order))
    try:
        # Build a green-forward palette with subtle variations
        pal_list = [
            COLORS['siamese'], COLORS['accent'], COLORS['primary'],
            '#A5D6A7', '#66BB6A', '#43A047', '#2E7D32', '#1B5E20',
            COLORS['neutral']
        ]
    except Exception:
        pal_list = [COLORS['siamese'], COLORS['accent'], COLORS['primary'], COLORS['neutral']]
    palette = {m: pal_list[i % len(pal_list)] for i, m in enumerate(uniq_methods)}
    if 'Psemut' in palette:
        palette['Psemut'] = COLORS['siamese']

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='method', y='rmse', order=order, palette=palette)
    # Overlay seed points（深灰）
    sns.stripplot(data=df, x='method', y='rmse', order=order, hue='method', dodge=False, palette=palette, legend=False, size=3, alpha=0.6)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    split_name = 'Valid' if split == 'valid' else 'Test'
    plt.title(f'Baseline RMSE Across Seeds ({split_name})', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    # Zoom y-axis for better visual separation (avoid nearly flat boxes)
    y_min, y_max = float(df['rmse'].min()), float(df['rmse'].max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def create_rmse_boxplot_ablation(results, save_path, split: str = 'valid'):
    """Create RMSE-only box plot for Psemut ablations across seeds, sorted by median."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {}) if split == 'valid' else results.get('per_seed_test', {})

    ablations = [
        ('Psemut', 'base'),
        ('No Exchange', 'no_exchange'),
        ('Diff Only', 'diff_only'),
        ('Mut Only', 'mut_only'),
        ('WT Only', 'wt_only'),
    ]

    plot_data, plot_labels = [], []
    plot_data_te, plot_labels_te = [], []
    for label, key in ablations:
        vals = []
        for seed in seeds:
            v = per_seed.get(seed, {}).get('model', {}).get(key, {}).get('rmse')
            if v is not None:
                vals.append(v)
        if len(vals) > 0:
            plot_data.append(vals)
            plot_labels.append(label)

    if len(plot_data) == 0:
        # 兜底：生成空图
        plt.text(0.5, 0.5, 'No data available', ha='center', va='center')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        return

    # Sort by median RMSE ascending for better comparison
    medians = [float(np.median(v)) if len(v) else float('inf') for v in plot_data]
    order_idx = np.argsort(medians)
    plot_data = [plot_data[i] for i in order_idx]
    plot_labels = [plot_labels[i] for i in order_idx]
    # Gather test metrics if available
    per_seed_te = results.get('per_seed_test', {})
    for label, key in ablations:
        vals = []
        for seed in seeds:
            v = per_seed_te.get(seed, {}).get('model', {}).get(key, {}).get('rmse')
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            plot_data_te.append(vals)
            plot_labels_te.append(label)

    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.7)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['green'] if split == 'valid' else COLORS['accent'])
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1.5)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1.5)
    for median in bp['medians']:
        median.set(color='white', linewidth=2)

    ax.set_xticklabels(plot_labels, rotation=30, ha='right')
    ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
    split_name = 'Valid' if split == 'valid' else 'Test'
    ax.set_title(f'Ablation RMSE Across Seeds ({split_name})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Overlay seed points
    try:
        import numpy as _np
        xs = _np.arange(1, len(plot_data) + 1)
        for x, vals in zip(xs, plot_data):
            xj = _np.random.normal(loc=x, scale=0.02, size=len(vals))
            ax.scatter(xj, vals, s=10, c='k', alpha=0.6)
    except Exception:
        pass
    # Overlay test split as secondary colored dots (if available)
    # Scatter seed points with color cycling
    try:
        import numpy as _np
        pal = sns.color_palette('tab10', n_colors=len(plot_labels))
        xs = _np.arange(1, len(plot_data) + 1)
        for i, (x, vals) in enumerate(zip(xs, plot_data)):
            color = pal[i % len(pal)]
            xj = _np.random.normal(loc=x, scale=0.03, size=len(vals))
            ax.scatter(xj, vals, s=12, c=[color], alpha=0.7)
    except Exception:
        pass

    # Zoom y-axis
    import numpy as _np
    allv = _np.concatenate([_np.array(v, dtype=float) for v in plot_data]) if len(plot_data) else _np.array([0.0])
    y_min, y_max = float(allv.min()), float(allv.max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    ax.set_ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def create_auc_boxplot_baselines(results, save_path, split: str = 'valid'):
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {}) if split == 'valid' else results.get('per_seed_test', {})
    families = {
        'linear': ['elasticnet'],
        'svr': ['svr_rbf'],
        'tree': ['rf'],
        'mlp': ['mlp_sklearn'],
        'kernel_knn': ['kernel_ridge_rbf'],
    }
    pretty = {
        'elasticnet': 'ElasticNet', 'svr_rbf': 'SVR-RBF', 'rf': 'RandomForest',
        'mlp_sklearn': 'MLP', 'kernel_ridge_rbf': 'KernelRidge'
    }
    rows = []
    for seed in seeds:
        seed_dict = per_seed.get(seed, {})
        for cat, methods in families.items():
            catd = seed_dict.get(cat, {})
            for m in methods:
                d = catd.get(m, {})
                auc = d.get('auc')
                if isinstance(auc, (int, float)):
                    rows.append({'method': pretty.get(m, m), 'auc': float(auc)})
    if not rows:
        plt.figure(figsize=(8, 4)); plt.text(0.5,0.5,'No AUC data',ha='center',va='center'); plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(); return
    df = pd.DataFrame(rows)
    med = df.groupby('method')['auc'].median().sort_values(ascending=False)
    order = list(med.index)
    pal = {m: COLORS['accent'] for m in order}
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x='method', y='auc', order=order, palette=pal)
    sns.stripplot(data=df, x='method', y='auc', order=order, color=COLORS['neutral'], size=3, alpha=0.6)
    plt.xticks(rotation=30, ha='right'); plt.ylabel('AUC-ROC');
    split_name = 'Valid' if split=='valid' else 'Test'
    plt.title(f'Baseline AUC-ROC Across Seeds ({split_name})'); plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


def create_auc_boxplot_ablation(results, save_path, split: str = 'valid'):
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {}) if split == 'valid' else results.get('per_seed_test', {})
    abls = [('Psemut','base'),('No Exchange','no_exchange'),('Diff Only','diff_only'),('Mut Only','mut_only'),('WT Only','wt_only')]
    rows=[]
    for label,key in abls:
        for seed in seeds:
            auc = per_seed.get(seed,{}).get('model',{}).get(key,{}).get('auc')
            if isinstance(auc,(int,float)):
                rows.append({'method': label, 'auc': float(auc)})
    if not rows:
        plt.figure(figsize=(8, 4)); plt.text(0.5,0.5,'No AUC data',ha='center',va='center'); plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close(); return
    df = pd.DataFrame(rows)
    med = df.groupby('method')['auc'].median().sort_values(ascending=False)
    order = list(med.index)
    pal = {m: COLORS['primary'] for m in order}
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='method', y='auc', order=order, palette=pal)
    sns.stripplot(data=df, x='method', y='auc', order=order, color=COLORS['neutral'], size=3, alpha=0.6)
    plt.xticks(rotation=30, ha='right'); plt.ylabel('AUC-ROC');
    split_name = 'Valid' if split=='valid' else 'Test'
    plt.title(f'Ablation AUC-ROC Across Seeds ({split_name})'); plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); os.makedirs(os.path.dirname(save_path), exist_ok=True); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


def create_r2_boxplot_baselines(results, save_path):
    """Create R2-only box plot for traditional baselines across seeds, sorted by median ascending."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})
    families = {
        'linear': ['linear', 'ridge', 'elasticnet', 'lasso', 'bayesian_ridge'],
        'svr': ['svr_rbf'],
        'tree': ['rf', 'extratrees'],  # drop 'xgb'
        'mlp': ['mlp_sklearn'],
        'kernel_knn': ['kernel_ridge_rbf', 'knn_reg'],
    }
    pretty = {
        'linear': 'Linear', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet', 'lasso': 'Lasso', 'bayesian_ridge': 'BayesianRidge',
        'svr_rbf': 'SVR-RBF',
        'rf': 'RandomForest', 'extratrees': 'ExtraTrees', 'xgb': 'XGBoost',
        'mlp_sklearn': 'MLP',
        'kernel_ridge_rbf': 'KernelRidge', 'knn_reg': 'KNN',
    }
    rows = []
    for seed in seeds:
        sd = per_seed.get(seed, {})
        for cat, methods in families.items():
            catd = sd.get(cat, {})
            for m in methods:
                d = catd.get(m)
                if isinstance(d, dict) and isinstance(d.get('r2'), (int, float)):
                    rows.append({'method': pretty.get(m, m), 'r2': d['r2']})
        v = sd.get('model', {}).get('base', {}).get('r2')
        if isinstance(v, (int, float)):
            rows.append({'method': 'Psemut', 'r2': v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # sort by median ascending (small to large) per user preference
    med = df.groupby('method')['r2'].median().sort_values()
    order = list(med.index)
    pal_default = COLORS['green']
    palette = {m: pal_default for m in df['method'].unique()}
    if 'Psemut' in df['method'].unique():
        palette['Psemut'] = COLORS['siamese']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='method', y='r2', order=order, palette=palette)
    sns.stripplot(data=df, x='method', y='r2', order=order, color=COLORS['neutral'], size=3, alpha=0.5)
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('R²', fontsize=12, fontweight='bold')
    plt.title('Baseline R² Across Seeds', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    y_min, y_max = float(df['r2'].min()), float(df['r2'].max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def plot_roc_curves(results, save_dir: str, split: str = 'valid'):
    """Draw ROC curves with AUC for representative baselines and siamese base.
    Requires raw files saved by training (auc_raw_val.json / auc_raw_test.json, and siamese auc_raw_test.json).
    """
    seeds = results.get('seeds', [])
    exp1_root = results.get('exp1_root', os.path.join('outputs','experiments','exp1_three_split'))
    rep = {
        'linear': 'ElasticNet',
        'svr': 'SVR-RBF',
        'tree': 'RandomForest',
        'mlp': 'MLP',
        'kernel_knn': 'KernelRidge',
        'model': 'Psemut',
    }
    fam_dir = {
        'linear': 'linear', 'svr': 'svr', 'tree': 'tree', 'mlp': 'mlp', 'kernel_knn': 'kernel_knn'
    }
    os.makedirs(save_dir, exist_ok=True)

    # per-seed ROC on one canvas (baseline families)
    for fam, dname in fam_dir.items():
        plt.figure(figsize=(6,5))
        for sd in seeds:
            base_dir = os.path.join(exp1_root, f'baselines_seed_{sd}', dname)
            fn = 'auc_raw_val.json' if split=='valid' else 'auc_raw_test.json'
            raw = None
            try:
                raw = json.load(open(os.path.join(base_dir, fn), 'r', encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(raw, dict) or 'y_true_binary' not in raw:
                continue
            y = np.asarray(raw['y_true_binary'], dtype=int)
            scores = np.asarray(raw.get('probs', raw.get('scores', [])), dtype=float)
            if y.size==0 or scores.size==0:
                continue
            fpr, tpr, _ = roc_curve(y, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1.2, label=f'seed {sd} (AUC={roc_auc:.3f})')
        plt.plot([0,1],[0,1], ls='--', c=COLORS['neutral'])
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{rep[fam]} ROC ({split})'); plt.legend(loc='lower right', fontsize=8)
        out = os.path.join(save_dir, f'roc_{fam}_{split}.png')
        plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()

    # siamese base
    plt.figure(figsize=(6,5))
    for sd in seeds:
        seed_dir = os.path.join(exp1_root, 'model_base', f'seed_{sd}')
        fn = 'auc_raw_val.json' if split=='valid' else 'auc_raw_test.json'
        try:
            raw = json.load(open(os.path.join(seed_dir, fn), 'r', encoding='utf-8'))
        except Exception:
            continue
        if not isinstance(raw, dict) or 'y_true_binary' not in raw:
            continue
        y = np.asarray(raw['y_true_binary'], dtype=int)
        probs = np.asarray(raw.get('probs', []), dtype=float)
        if y.size==0 or probs.size==0:
            continue
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.2, label=f'seed {sd} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1], ls='--', c=COLORS['neutral'])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'{rep["model"]} ROC ({split})'); plt.legend(loc='lower right', fontsize=8)
    out = os.path.join(save_dir, f'roc_model_base_{split}.png')
    plt.tight_layout(); plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()


def plot_roc_aggregate_baselines(results, save_path: str, split: str = 'valid'):
    """聚合每个基线方法（与箱线图一致的全量集合）的多种子ROC为一条均值曲线 + 阴影(标准差)。"""
    seeds = results.get('seeds', [])
    exp1_root = results.get('exp1_root', os.path.join('outputs','experiments','exp1_three_split'))
    # 显示名 -> (子目录, 方法key=raw文件后缀)
    methods = {
        'Linear':        ('linear',      'linear'),
        'Ridge':         ('linear',      'ridge'),
        'ElasticNet':    ('linear',      'elasticnet'),
        'Lasso':         ('linear',      'lasso'),
        'BayesianRidge': ('linear',      'bayesian_ridge'),
        'SVR-Linear':    ('svr',         'svr_linear'),
        'SVR-RBF':       ('svr',         'svr_rbf'),
        'RandomForest':  ('tree',        'rf'),
        'ExtraTrees':    ('tree',        'extratrees'),
        'MLP':           ('mlp',         'mlp_sklearn'),
        'KernelRidge':   ('kernel_knn',  'kernel_ridge_rbf'),
        'KNN':           ('kernel_knn',  'knn_reg'),
    }
    method_list = list(methods.keys())
    pal = sns.color_palette('tab10', n_colors=max(10, len(method_list)))
    color_map = {m: pal[i % len(pal)] for i, m in enumerate(method_list)}
    grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(7,6))
    for m in method_list:
        dname, key = methods[m]
        curves, aucs = [], []
        for sd in seeds:
            base_dir = os.path.join(exp1_root, f'baselines_seed_{sd}', dname)
            fn = (f'auc_raw_val_{key}.json' if split=='valid' else f'auc_raw_test_{key}.json')
            try:
                raw = json.load(open(os.path.join(base_dir, fn), 'r', encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(raw, dict) or 'y_true_binary' not in raw:
                continue
            y = np.asarray(raw['y_true_binary'], dtype=int)
            scores = np.asarray(raw.get('probs', raw.get('scores', [])), dtype=float)
            if y.size==0 or scores.size==0:
                continue
            fpr, tpr, _ = roc_curve(y, scores)
            # 确保单调，插值到公共网格
            fpr, idx = np.unique(fpr, return_index=True)
            tpr = tpr[idx]
            tpr_i = np.interp(grid, fpr, tpr, left=0, right=1)
            curves.append(tpr_i)
            aucs.append(auc(fpr, tpr))
        if not curves:
            continue
        C = np.vstack(curves)
        mean_tpr = C.mean(axis=0)
        std_tpr = C.std(axis=0)
        auc_mean = float(np.mean(aucs)); auc_std = float(np.std(aucs))
        plt.plot(grid, mean_tpr, lw=2.0, c=color_map[m], label=f'{m} (AUC={auc_mean:.3f}±{auc_std:.3f})')
        plt.fill_between(grid, np.clip(mean_tpr-std_tpr, 0, 1), np.clip(mean_tpr+std_tpr, 0, 1),
                         color=color_map[m], alpha=0.12)
    plt.plot([0,1],[0,1], ls='--', c=COLORS['neutral'])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Baselines ROC ({split})')
    plt.legend(loc='lower right', fontsize=8, ncol=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight');
    try:
        plt.savefig(os.path.splitext(save_path)[0] + '.svg', format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def plot_roc_aggregate_ablation(results, save_path: str, split: str = 'valid'):
    """聚合每个消融方法的多种子ROC为一条均值曲线 + 阴影(标准差)。"""
    seeds = results.get('seeds', [])
    exp1_root = results.get('exp1_root', os.path.join('outputs','experiments','exp1_three_split'))
    abls = {
        'Psemut': 'model_base',
        'No Exchange': 'model_no_exchange',
        'Diff Only': 'model_diff_only',
        'Mut Only': 'model_mut_only',
        'WT Only': 'model_wt_only',
    }
    method_list = list(abls.keys())
    pal = sns.color_palette('tab10', n_colors=len(method_list))
    color_map = {m: pal[i % len(pal)] for i, m in enumerate(method_list)}
    grid = np.linspace(0, 1, 201)
    plt.figure(figsize=(7,6))
    for label, tag in abls.items():
        curves, aucs = [], []
        for sd in seeds:
            seed_dir = os.path.join(exp1_root, tag, f'seed_{sd}')
            fn = 'auc_raw_val.json' if split=='valid' else 'auc_raw_test.json'
            try:
                raw = json.load(open(os.path.join(seed_dir, fn), 'r', encoding='utf-8'))
            except Exception:
                continue
            if not isinstance(raw, dict) or 'y_true_binary' not in raw:
                continue
            y = np.asarray(raw['y_true_binary'], dtype=int)
            probs = np.asarray(raw.get('probs', []), dtype=float)
            if y.size==0 or probs.size==0:
                continue
            fpr, tpr, _ = roc_curve(y, probs)
            fpr, idx = np.unique(fpr, return_index=True)
            tpr = tpr[idx]
            tpr_i = np.interp(grid, fpr, tpr, left=0, right=1)
            curves.append(tpr_i)
            aucs.append(auc(fpr, tpr))
        if not curves:
            continue
        C = np.vstack(curves)
        mean_tpr = C.mean(axis=0)
        std_tpr = C.std(axis=0)
        auc_mean = float(np.mean(aucs)); auc_std = float(np.std(aucs))
        plt.plot(grid, mean_tpr, lw=2.0, c=color_map[label], label=f'{label} (AUC={auc_mean:.3f}±{auc_std:.3f})')
        plt.fill_between(grid, np.clip(mean_tpr-std_tpr, 0, 1), np.clip(mean_tpr+std_tpr, 0, 1),
                         color=color_map[label], alpha=0.12)
    plt.plot([0,1],[0,1], ls='--', c=COLORS['neutral'])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Ablation ROC ({split})')
    plt.legend(loc='lower right', fontsize=8, ncol=1)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight');
    try:
        plt.savefig(os.path.splitext(save_path)[0] + '.svg', format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def create_r2_boxplot_ablation(results, save_path):
    """Create R2-only box plot for Psemut ablations across seeds, sorted by median ascending."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})
    abls = [('Psemut', 'base'), ('No Exchange', 'no_exchange'), ('Diff Only', 'diff_only'), ('Mut Only', 'mut_only'), ('WT Only', 'wt_only')]
    plot_data, plot_labels = [], []
    for label, key in abls:
        vals = []
        for seed in seeds:
            v = per_seed.get(seed, {}).get('model', {}).get(key, {}).get('r2')
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            plot_data.append(vals)
            plot_labels.append(label)
    if not plot_data:
        return
    medians = [float(np.median(v)) if len(v) else float('inf') for v in plot_data]
    order_idx = np.argsort(medians)
    plot_data = [plot_data[i] for i in order_idx]
    plot_labels = [plot_labels[i] for i in order_idx]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(plot_data, patch_artist=True, widths=0.7)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['siamese'])
        patch.set_alpha(0.7)
    for w in bp['whiskers']:
        w.set(color='gray', linewidth=1.5)
    for c in bp['caps']:
        c.set(color='gray', linewidth=1.5)
    for m in bp['medians']:
        m.set(color='white', linewidth=2)
    ax.set_xticklabels(plot_labels, rotation=30, ha='right')
    ax.set_ylabel('R²', fontsize=12, fontweight='bold')
    ax.set_title('Ablation R² Across Seeds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    try:
        import numpy as _np
        xs = _np.arange(1, len(plot_data) + 1)
        for x, vals in zip(xs, plot_data):
            xj = _np.random.normal(loc=x, scale=0.02, size=len(vals))
            ax.scatter(xj, vals, s=10, c='k', alpha=0.6)
    except Exception:
        pass
    import numpy as _np
    allv = _np.concatenate([_np.array(v, dtype=float) for v in plot_data]) if len(plot_data) else _np.array([0.0])
    y_min, y_max = float(allv.min()), float(allv.max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    ax.set_ylim(y_min - margin, y_max + margin)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()
def create_rmse_swarm_baselines(results, save_path):
    """Swarm (beeswarm) of seed RMSEs + mean±std overlay for baselines and our Base."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})

    families = {
        'linear': ['linear', 'ridge', 'elasticnet', 'lasso', 'bayesian_ridge'],
        'svr': ['svr_rbf'],
        'tree': ['rf', 'extratrees', 'xgb'],
        'mlp': ['mlp_sklearn'],
        'kernel_knn': ['kernel_ridge_rbf', 'knn_reg'],
    }
    pretty = {
        'linear': 'Linear', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet', 'lasso': 'Lasso', 'bayesian_ridge': 'BayesianRidge',
        'svr_rbf': 'SVR-RBF',
        'rf': 'RandomForest', 'extratrees': 'ExtraTrees', 'xgb': 'XGBoost',
        'mlp_sklearn': 'MLP',
        'kernel_ridge_rbf': 'KernelRidge', 'knn_reg': 'KNN',
    }

    rows = []
    for seed in seeds:
        seed_dict = per_seed.get(seed, {})
        for category, methods in families.items():
            cat = seed_dict.get(category, {})
            for m in methods:
                entry = cat.get(m)
                if isinstance(entry, dict) and ('rmse' in entry) and isinstance(entry['rmse'], (int, float)):
                    rows.append({'method': pretty.get(m, m), 'rmse': entry['rmse']})
        v = seed_dict.get('model', {}).get('base', {}).get('rmse')
        if isinstance(v, (int, float)):
            rows.append({'method': 'Base', 'rmse': v})

    if not rows:
        return

    df = pd.DataFrame(rows)
    order = []
    for fam in families.values():
        for k in fam:
            name = pretty.get(k, k)
            if name not in order and name in df['method'].unique():
                order.append(name)
    if 'Base' in df['method'].unique():
        order.append('Base')

    plt.figure(figsize=(12, 6))
    sns.swarmplot(data=df, x='method', y='rmse', order=order, color=COLORS['neutral'], size=5)
    # mean±std overlay
    g = df.groupby('method')['rmse'].agg(['mean', 'std']).reindex(order)
    xs = range(len(order))
    plt.errorbar(xs, g['mean'].values, yerr=g['std'].values, fmt='D', color=COLORS['siamese'],
                 ecolor=COLORS['siamese'], elinewidth=2, capsize=4, markersize=6)
    plt.xticks(xs, order, rotation=30, ha='right')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Baseline RMSE (Seeds as points; mean±std)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    y_min, y_max = float(df['rmse'].min()), float(df['rmse'].max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_rmse_swarm_ablation(results, save_path):
    """Swarm (beeswarm) of seed RMSEs + mean±std overlay for Siamese ablations."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})

    abls = [('Psemut', 'base'), ('No Exchange', 'no_exchange'), ('Diff Only', 'diff_only'), ('Mut Only', 'mut_only'), ('WT Only', 'wt_only')]
    rows = []
    for label, key in abls:
        for seed in seeds:
            v = per_seed.get(seed, {}).get('model', {}).get(key, {}).get('rmse')
            if isinstance(v, (int, float)):
                rows.append({'method': label, 'rmse': v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    order = [k for k, _ in abls if k in df['method'].unique()]

    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=df, x='method', y='rmse', order=order, color=COLORS['neutral'], size=5)
    g = df.groupby('method')['rmse'].agg(['mean', 'std']).reindex(order)
    xs = range(len(order))
    plt.errorbar(xs, g['mean'].values, yerr=g['std'].values, fmt='D', color=COLORS['green'],
                 ecolor=COLORS['green'], elinewidth=2, capsize=4, markersize=6)
    plt.xticks(xs, order, rotation=30, ha='right')
    plt.ylabel('RMSE', fontsize=12, fontweight='bold')
    plt.title('Ablation RMSE (Seeds as points; mean±std)', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    y_min, y_max = float(df['rmse'].min()), float(df['rmse'].max())
    rng = max(1e-6, y_max - y_min)
    margin = max(0.02, 0.08 * rng)
    plt.ylim(y_min - margin, y_max + margin)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_rmse_meanstd_baselines(results, save_path):
    """Mean±Std point chart for baselines + Base, sorted by mean RMSE."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})
    families = {
        'linear': ['linear', 'ridge', 'elasticnet', 'lasso', 'bayesian_ridge'],
        'svr': ['svr_linear', 'svr_rbf'],
        'tree': ['rf', 'extratrees'],  # drop 'xgb'
        'mlp': ['mlp_sklearn'],
        'kernel_knn': ['kernel_ridge_rbf', 'knn_reg'],
    }
    pretty = {
        'linear': 'Linear', 'ridge': 'Ridge', 'elasticnet': 'ElasticNet', 'lasso': 'Lasso', 'bayesian_ridge': 'BayesianRidge',
        'svr_linear': 'SVR-Linear', 'svr_rbf': 'SVR-RBF',
        'rf': 'RandomForest', 'extratrees': 'ExtraTrees', 'xgb': 'XGBoost',
        'mlp_sklearn': 'MLP',
        'kernel_ridge_rbf': 'KernelRidge', 'knn_reg': 'KNN',
    }
    rows = []
    for seed in seeds:
        sd = per_seed.get(seed, {})
        for cat, methods in families.items():
            for m in methods:
                v = sd.get(cat, {}).get(m, {})
                if isinstance(v, dict) and isinstance(v.get('rmse'), (int, float)):
                    rows.append({'method': pretty.get(m, m), 'rmse': v['rmse']})
        v = sd.get('model', {}).get('base', {}).get('rmse')
        if isinstance(v, (int, float)):
            rows.append({'method': 'Base', 'rmse': v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    stat = df.groupby('method')['rmse'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    plt.figure(figsize=(10, max(5, 0.4*len(stat))))
    y = range(len(stat))
    plt.errorbar(stat['mean'].values, y, xerr=stat['std'].values, fmt='o', color=COLORS['green'], ecolor=COLORS['green'], capsize=4)
    if 'Base' in stat.index:
        idx = list(stat.index).index('Base')
        plt.errorbar([stat.loc['Base','mean']], [idx], xerr=[stat.loc['Base','std']], fmt='o', color=COLORS['siamese'], ecolor=COLORS['siamese'], capsize=4)
    plt.yticks(y, stat.index)
    plt.xlabel('RMSE (mean ± std)')
    plt.title('Baseline RMSE (mean±std, sorted)')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()


def create_rmse_meanstd_ablation(results, save_path):
    """Mean±Std point chart for ablations, sorted by mean RMSE."""
    seeds = results.get('seeds', [])
    per_seed = results.get('per_seed', {})
    abls = [('Psemut', 'base'), ('No Exchange', 'no_exchange'), ('Diff Only', 'diff_only'), ('Mut Only', 'mut_only'), ('WT Only', 'wt_only')]
    rows = []
    for label, key in abls:
        vals = []
        for seed in seeds:
            v = per_seed.get(seed, {}).get('model', {}).get(key, {}).get('rmse')
            if isinstance(v, (int, float)):
                vals.append(v)
        if vals:
            for v in vals:
                rows.append({'method': label, 'rmse': v})
    if not rows:
        return
    df = pd.DataFrame(rows)
    stat = df.groupby('method')['rmse'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    plt.figure(figsize=(8, max(4, 0.4*len(stat))))
    y = range(len(stat))
    plt.errorbar(stat['mean'].values, y, xerr=stat['std'].values, fmt='o', color=COLORS['green'], ecolor=COLORS['green'], capsize=4)
    if 'Base' in stat.index:
        idx = list(stat.index).index('Base')
        plt.errorbar([stat.loc['Base','mean']], [idx], xerr=[stat.loc['Base','std']], fmt='o', color=COLORS['siamese'], ecolor=COLORS['siamese'], capsize=4)
    plt.yticks(y, stat.index)
    plt.xlabel('RMSE (mean ± std)')
    plt.title('Ablation RMSE (mean±std, sorted)')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()

def create_heatmap_comparison(results, save_path):
    """Create heatmap to compare performance of various methods"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data
    methods = []
    metrics_data = {'R²': [], 'RMSE': [], 'MAE': []}

    # Traditional methods
    traditional_methods = {
        'ElasticNet': 'linear/elasticnet',
        'SVR-RBF': 'svr/svr_rbf',
        'RandomForest': 'tree/rf',
        'MLP': 'mlp/mlp_sklearn',
        'KernelRidge': 'kernel_knn/kernel_ridge_rbf'
    }

    # Siamese methods
    siamese_methods = {
        'Psemut': 'model/base',
        'Siamese-NoEx': 'model/no_exchange',
        'Siamese-Diff': 'model/diff_only'
    }

    all_methods = {**traditional_methods, **siamese_methods}

    for method_name, path in all_methods.items():
        category, submethod = path.split('/')
        methods.append(method_name)

        for metric_display, metric_key in [('R²', 'r2'), ('RMSE', 'rmse'), ('MAE', 'mae')]:
            values = [results['per_seed'][seed][category][submethod][metric_key]
                     for seed in results['seeds']]
            metrics_data[metric_display].append(np.mean(values))

    # Create heatmap data
    heatmap_data = np.array([metrics_data['R²'], metrics_data['RMSE'], metrics_data['MAE']]).T

    # Normalize data for color mapping
    from sklearn.preprocessing import MinMaxScaler
    scaler_r2 = MinMaxScaler()
    scaler_error = MinMaxScaler()

    # Higher R² is better (green), lower RMSE/MAE is better (red)
    r2_normalized = scaler_r2.fit_transform(heatmap_data[:, 0].reshape(-1, 1)).flatten()
    rmse_normalized = 1 - scaler_error.fit_transform(heatmap_data[:, 1].reshape(-1, 1)).flatten()
    mae_normalized = 1 - scaler_error.fit_transform(heatmap_data[:, 2].reshape(-1, 1)).flatten()

    # Combined score (simple average)
    combined_scores = (r2_normalized + rmse_normalized + mae_normalized) / 3

    # Draw heatmap
    im = ax.imshow(combined_scores.reshape(-1, 1), cmap='YlGnBu', aspect='auto', alpha=0.85)

    # Add numerical labels
    for i, (method, score) in enumerate(zip(methods, combined_scores)):
        ax.text(0, i, '.3f',
               ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xticks([0])
    ax.set_xticklabels(['Combined Score'])
    ax.set_title('Model Performance Heatmap\n(Higher scores are better)', fontsize=14, fontweight='bold', pad=20)

    # Add color bar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    try:
        svg_path = os.path.splitext(save_path)[0] + '.svg'
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
    except Exception:
        pass
    plt.close()

def create_summary_stats(results, save_path):
    """Create summary statistics table"""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data
    summary_data = []

    # Traditional methods
    traditional_methods = {
        'ElasticNet': ('linear', 'elasticnet'),
        'SVR-RBF': ('svr', 'svr_rbf'),
        'RandomForest': ('tree', 'rf'),
        'MLP': ('mlp', 'mlp_sklearn'),
        'KernelRidge': ('kernel_knn', 'kernel_ridge_rbf')
    }

    # Siamese methods
    siamese_methods = {
        'Siamese-Base': ('model', 'base'),
        'Siamese-NoEx': ('model', 'no_exchange'),
        'Siamese-Diff': ('model', 'diff_only'),
        'Siamese-Mut': ('model', 'mut_only'),
        'Siamese-WT': ('model', 'wt_only')
    }

    for method_name, (category, submethod) in {**traditional_methods, **siamese_methods}.items():
        metrics = {}
        for metric in ['r2', 'rmse', 'mae']:
            values = [results['per_seed'][seed][category][submethod][metric]
                     for seed in results['seeds']]
            metrics[f'{metric}_mean'] = np.mean(values)
            metrics[f'{metric}_std'] = np.std(values)

        summary_data.append({
            'Method': method_name,
            'R²': '.3f',
            'RMSE': '.3f',
            'MAE': '.3f',
            'Type': 'Siamese' if 'Siamese' in method_name else 'Traditional'
        })

    df = pd.DataFrame(summary_data)

    # Create table
    ax.axis('off')
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['#E8F1F2'] * len(df.columns),
                    cellColours=[[COLORS['siamese'] if row['Type'] == 'Siamese'
                                else COLORS['traditional'] for row in df.itertuples()]
                               for _ in range(len(df))])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    ax.set_title('Benchmark Results Summary\n(Mean ± Std across 3 seeds)',
                fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function"""
    results_path = os.path.join('outputs', 'benchmarks', 'all', 'summary_all_seeds.json')
    output_dir = os.path.join('outputs', 'benchmarks', 'visualizations')

    # Backward-compatible path resolution
    if not os.path.exists(results_path):
        alt_results_path = os.path.join('bench_results_all', 'summary_all_seeds.json')
        if os.path.exists(alt_results_path):
            results_path = alt_results_path

    if not os.path.isdir(output_dir):
        alt_output_dir = os.path.join('bench_results_all', 'visualizations')
        # Ensure the alt directory exists if we switch to it
        if os.path.exists(alt_output_dir):
            output_dir = alt_output_dir

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    results = load_results(results_path)
    # Normalize seed key types (JSON keys -> str) to ints to avoid KeyError
    try:
        if isinstance(results.get('per_seed', None), dict):
            results['per_seed'] = {int(k): v for k, v in results['per_seed'].items()}
        if isinstance(results.get('per_seed_test', None), dict):
            results['per_seed_test'] = {int(k): v for k, v in results['per_seed_test'].items()}
        if isinstance(results.get('seeds', None), list):
            results['seeds'] = [int(s) for s in results['seeds']]
    except Exception:
        pass
    augment_results_with_seed_files(results, results_path)

    print("Generating advanced visualization charts...")

    # 1. Performance bubble chart
    create_performance_bubble_chart(results, f'{output_dir}/performance_bubble_chart.png')
    print("OK: Performance bubble chart generated")

    # 2. Radar chart
    create_radar_chart(results, f'{output_dir}/siamese_ablation_radar.png')
    print("OK: Siamese ablation radar chart generated")

    # Prepare subfolders
    box_dir = os.path.join(output_dir, 'boxplots')
    swarm_dir = os.path.join(output_dir, 'swarm')
    meanstd_dir = os.path.join(output_dir, 'meanstd')
    os.makedirs(box_dir, exist_ok=True)
    os.makedirs(swarm_dir, exist_ok=True)
    os.makedirs(meanstd_dir, exist_ok=True)

    # 3. RMSE-only Box plots (split-specific: valid & test)
    create_rmse_boxplot_baselines(results, f'{box_dir}/baseline_rmse_boxplot_valid.png', split='valid')
    print("OK: Baseline RMSE box plot (valid) generated")
    create_rmse_boxplot_baselines(results, f'{box_dir}/baseline_rmse_boxplot_test.png', split='test')
    print("OK: Baseline RMSE box plot (test) generated")
    create_rmse_boxplot_ablation(results, f'{box_dir}/ablation_rmse_boxplot_valid.png', split='valid')
    print("OK: Ablation RMSE box plot (valid) generated")
    create_rmse_boxplot_ablation(results, f'{box_dir}/ablation_rmse_boxplot_test.png', split='test')
    print("OK: Ablation RMSE box plot (test) generated")

    # 3a. R2-only Box plots
    create_r2_boxplot_baselines(results, f'{box_dir}/baseline_r2_boxplot.png')
    print("OK: Baseline R2 box plot generated")
    create_r2_boxplot_ablation(results, f'{box_dir}/ablation_r2_boxplot.png')
    print("OK: Ablation R2 box plot generated")

    # 3b. RMSE-only Swarm plots (alternative for small N)
    create_rmse_swarm_baselines(results, f'{swarm_dir}/baseline_rmse_swarm.png')
    print("OK: Baseline RMSE swarm plot generated")
    create_rmse_swarm_ablation(results, f'{swarm_dir}/ablation_rmse_swarm.png')
    print("OK: Ablation RMSE swarm plot generated")

    # 3c. Mean±Std point charts (sorted)
    create_rmse_meanstd_baselines(results, f'{meanstd_dir}/baseline_rmse_meanstd.png')
    print("OK: Baseline RMSE mean+std plot generated")
    create_rmse_meanstd_ablation(results, f'{meanstd_dir}/ablation_rmse_meanstd.png')
    print("OK: Ablation RMSE mean+std plot generated")

    # 3d. AUC-ROC Box plots (split-specific)
    create_auc_boxplot_baselines(results, f'{box_dir}/baseline_auc_boxplot_valid.png', split='valid')
    print("OK: Baseline AUC box plot (valid) generated")
    create_auc_boxplot_baselines(results, f'{box_dir}/baseline_auc_boxplot_test.png', split='test')
    print("OK: Baseline AUC box plot (test) generated")
    create_auc_boxplot_ablation(results, f'{box_dir}/ablation_auc_boxplot_valid.png', split='valid')
    print("OK: Ablation AUC box plot (valid) generated")
    create_auc_boxplot_ablation(results, f'{box_dir}/ablation_auc_boxplot_test.png', split='test')
    print("OK: Ablation AUC box plot (test) generated")

    # 3e. ROC curves (valid/test)
    roc_dir = os.path.join(output_dir, 'roc')
    # aggregate one figure per group (baselines / ablations) for valid & test
    plot_roc_aggregate_baselines(results, os.path.join(roc_dir, 'roc_baselines_valid.png'), split='valid')
    print("OK: Baselines ROC (valid) generated")
    plot_roc_aggregate_baselines(results, os.path.join(roc_dir, 'roc_baselines_test.png'), split='test')
    print("OK: Baselines ROC (test) generated")
    plot_roc_aggregate_ablation(results, os.path.join(roc_dir, 'roc_ablation_valid.png'), split='valid')
    print("OK: Ablation ROC (valid) generated")
    plot_roc_aggregate_ablation(results, os.path.join(roc_dir, 'roc_ablation_test.png'), split='test')
    print("OK: Ablation ROC (test) generated")

    # 4. Heatmap
    create_heatmap_comparison(results, f'{output_dir}/performance_heatmap.png')
    print("OK: Performance heatmap generated")

    # 5. Summary statistics table (temporarily skipped to avoid matplotlib table issues)
    # create_summary_stats(results, f'{output_dir}/summary_statistics.png')
    print("OK: Summary statistics table skipped (matplotlib table rendering issues)")

    print(f"\nAll charts saved to: {output_dir}/")
    print("Files included:")
    try:
        for root, _, files in os.walk(output_dir):
            for fn in sorted(files):
                if fn.lower().endswith('.png'):
                    rel = os.path.relpath(os.path.join(root, fn), output_dir)
                    print(f"- {rel}")
    except Exception:
        pass

if __name__ == '__main__':
    main()
