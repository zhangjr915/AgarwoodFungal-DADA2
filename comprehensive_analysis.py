#!/usr/bin/env python3
"""
Comprehensive Fungal ITS Community Analysis
Nature-level figures for fungal OTU data
All text in English, no Chinese characters
"""

import os
import warnings
import itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
import networkx as nx

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = '/home/zhhq/.openclaw/workspace-coder/FungalAnalysis'
FIG_DIR = os.path.join(BASE_DIR, 'figures')
RES_DIR = os.path.join(BASE_DIR, 'results')
SCRIPT_DIR = os.path.join(BASE_DIR, 'scripts')

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# Nature-style settings
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'grid.linewidth': 0.3,
    'lines.linewidth': 1.0,
})

# Color palette (color-blind friendly)
COLORS = {
    'A_a': '#4E79A7',
    'A_b': '#F28E2B',
    'A_c': '#59A14F',
    'B': '#E15759',
}
COLOR_LIST = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759']

# Sample grouping
SAMPLES = ['A_a_1', 'A_a_2', 'A_b_1', 'A_b_2', 'A_c_1', 'A_c_2', 'B_a', 'B_b']
GROUP_MAP = {
    'A_a_1': 'A_a', 'A_a_2': 'A_a',
    'A_b_1': 'A_b', 'A_b_2': 'A_b',
    'A_c_1': 'A_c', 'A_c_2': 'A_c',
    'B_a': 'B', 'B_b': 'B',
}
SUPERGROUP_MAP = {
    'A_a': 'A', 'A_b': 'A', 'A_c': 'A', 'B': 'B',
}

# ============================================================
# Data Loading
# ============================================================
print("Loading OTU table...")
otu_df = pd.read_csv(os.path.join(RES_DIR, 'otu_table.csv'), index_col=0)
sample_cols = [c for c in otu_df.columns if c in SAMPLES]
otu_df = otu_df[sample_cols]

# Relative abundance
rel_abund = otu_df.div(otu_df.sum(axis=0), axis=1) * 100

# Summary statistics storage
summary_stats = {}

print(f"Loaded {len(otu_df)} OTUs across {len(sample_cols)} samples")


# ============================================================
# Helper Functions
# ============================================================
def get_group_colors(samples):
    return [COLORS[GROUP_MAP[s]] for s in samples]

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")

def sig_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    else: return 'ns'


# ============================================================
# 1. Alpha Diversity
# ============================================================
print("\n[1/12] Alpha Diversity...")

def shannon(x):
    x = x[x > 0]
    p = x / x.sum()
    return -np.sum(p * np.log(p))

def simpson(x):
    x = x[x > 0]
    p = x / x.sum()
    return 1 - np.sum(p ** 2)

def chao1(x):
    x = x[x > 0]
    f1 = np.sum(x == 1)
    f2 = np.sum(x == 2)
    S_obs = len(x)
    if f2 == 0:
        return S_obs + f1 * (f1 - 1) / 2.0
    return S_obs + f1**2 / (2.0 * f2 + 1e-10)

def observed_otus(x):
    return np.sum(x > 0)

def pielou(x):
    H = shannon(x)
    S = observed_otus(x)
    if S <= 1:
        return 0
    return H / np.log(S)

alpha_metrics = {
    'Shannon': shannon,
    'Simpson': simpson,
    'Chao1': chao1,
    'Observed OTUs': observed_otus,
    "Pielou's Evenness": pielou,
}

alpha_df = pd.DataFrame(index=SAMPLES)
for metric_name, metric_fn in alpha_metrics.items():
    alpha_df[metric_name] = otu_df[SAMPLES].apply(lambda col: metric_fn(col.values))

alpha_df['Group'] = [GROUP_MAP[s] for s in alpha_df.index]

# Kruskal-Wallis tests
kw_results = {}
for metric_name in alpha_metrics:
    groups_data = []
    for g in ['A_a', 'A_b', 'A_c', 'B']:
        g_samples = alpha_df[alpha_df['Group'] == g][metric_name].values
        groups_data.append(g_samples)
    stat, p = stats.kruskal(*groups_data)
    kw_results[metric_name] = {'H': stat, 'p': p}

# Plot alpha diversity - 2x3 panel
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
axes = axes.flatten()

for idx, metric_name in enumerate(alpha_metrics):
    ax = axes[idx]
    data_plot = [alpha_df[alpha_df['Group'] == g][metric_name].values for g in ['A_a', 'A_b', 'A_c', 'B']]
    
    bp = ax.boxplot(data_plot, patch_artist=True, widths=0.6,
                    boxprops=dict(linewidth=0.8),
                    medianprops=dict(color='black', linewidth=1),
                    whiskerprops=dict(linewidth=0.6),
                    capprops=dict(linewidth=0.6),
                    flierprops=dict(markersize=3, markeredgecolor='gray'))
    
    for patch, color in zip(bp['boxes'], COLOR_LIST):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Scatter individual points
    for i, g in enumerate(['A_a', 'A_b', 'A_c', 'B']):
        vals = alpha_df[alpha_df['Group'] == g][metric_name].values
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                  color=COLOR_LIST[i], s=15, zorder=5, edgecolors='black', linewidths=0.3)
    
    ax.set_xticklabels(['A_a', 'A_b', 'A_c', 'B'])
    ax.set_title(metric_name)
    ax.set_ylabel(metric_name)
    
    # Add KW p-value
    p = kw_results[metric_name]['p']
    ax.text(0.95, 0.95, f'p = {p:.4f}' if p >= 0.001 else f'p < 0.001',
            transform=ax.transAxes, ha='right', va='top', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

# 6th panel: legend
axes[5].axis('off')
legend_patches = [mpatches.Patch(color=c, label=l, alpha=0.7) for c, l in zip(COLOR_LIST, ['A_a (Aquilaria)', 'A_b (Source 2)', 'A_c (Source 3)', 'B (Atractylodes)'])]
axes[5].legend(handles=legend_patches, loc='center', title='Group', frameon=True, fontsize=8)

fig.suptitle('Alpha Diversity Indices', fontsize=10, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, 'fig_alpha_diversity.png')

# Save alpha diversity stats
alpha_df.to_csv(os.path.join(RES_DIR, 'alpha_diversity.csv'))
summary_stats['alpha_kruskal_wallis'] = kw_results


# ============================================================
# 2. Rarefaction Curves
# ============================================================
print("\n[2/12] Rarefaction Curves...")

np.random.seed(42)

def rarefaction_curve(counts, step=500, max_depth=None):
    counts = counts.astype(int)
    total = counts.sum()
    if max_depth is None:
        max_depth = total
    depths = list(range(step, min(max_depth, total) + 1, step))
    if total not in depths:
        depths.append(total)
    
    otus_observed = []
    pool = np.repeat(np.arange(len(counts)), counts)
    
    for d in depths:
        np.random.shuffle(pool)
        subsample = pool[:d]
        otus_observed.append(len(np.unique(subsample)))
    
    return depths, otus_observed

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0))

for sample in SAMPLES:
    depths, otus = rarefaction_curve(otu_df[sample].values, step=1000)
    group = GROUP_MAP[sample]
    color = COLORS[group]
    ax.plot(depths, otus, color=color, alpha=0.8, linewidth=1.0,
            label=sample if sample in ['A_a_1', 'A_b_1', 'A_c_1', 'B_a'] else None)

# Add legend with group colors
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLOR_LIST, ['A_a', 'A_b', 'A_c', 'B'])]
ax.legend(handles=legend_patches, loc='lower right', frameon=True, fontsize=7)
ax.set_xlabel('Sequencing Depth')
ax.set_ylabel('Observed OTUs')
ax.set_title('Rarefaction Curves')
ax.grid(True, alpha=0.3)

save_fig(fig, 'fig_rarefaction_curve.png')


# ============================================================
# 3. Beta Diversity - PCoA
# ============================================================
print("\n[3/12] Beta Diversity - PCoA (Bray-Curtis)...")

def bray_curtis_distance(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    return np.sum(np.abs(x - y)) / np.sum(x + y)

# Compute Bray-Curtis distance matrix
n = len(SAMPLES)
bc_mat = np.zeros((n, n))
data_matrix = otu_df[SAMPLES].values.T

for i in range(n):
    for j in range(i + 1, n):
        d = bray_curtis_distance(data_matrix[i], data_matrix[j])
        bc_mat[i, j] = d
        bc_mat[j, i] = d

# PCoA via eigendecomposition
n_samples = n
H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
A = -0.5 * bc_mat ** 2
G = H @ A @ H
eigenvalues, eigenvectors = np.linalg.eigh(G)

# Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Take positive eigenvalues only
pos_mask = eigenvalues > 0
eigenvalues_pos = eigenvalues[pos_mask]
eigenvectors_pos = eigenvectors[:, pos_mask]

# Coordinates
coords = eigenvectors_pos[:, :2] * np.sqrt(eigenvalues_pos[:2])

# Variance explained
var_explained = eigenvalues_pos / eigenvalues_pos.sum() * 100

# PERMANOVA
def permanova(distance_mat, groups, permutations=999):
    n = len(groups)
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    
    # Overall mean distance
    ss_total = np.sum(distance_mat ** 2) / (2 * n)
    
    # Within-group SS
    ss_within = 0
    for g in unique_groups:
        idx_g = np.where(groups == g)[0]
        n_g = len(idx_g)
        if n_g < 2:
            continue
        for i in range(n_g):
            for j in range(i + 1, n_g):
                ss_within += distance_mat[idx_g[i], idx_g[j]] ** 2 / (2 * n)
    
    ss_between = ss_total - ss_within
    f_stat = (ss_between / (len(unique_groups) - 1)) / (ss_within / (n - len(unique_groups)))
    
    # Permutation test
    f_perms = []
    for _ in range(permutations):
        perm_groups = np.random.permutation(groups)
        ss_w_perm = 0
        for g in unique_groups:
            idx_g = np.where(perm_groups == g)[0]
            n_g = len(idx_g)
            if n_g < 2:
                continue
            for i in range(n_g):
                for j in range(i + 1, n_g):
                    ss_w_perm += distance_mat[idx_g[i], idx_g[j]] ** 2 / (2 * n)
        ss_b_perm = ss_total - ss_w_perm
        f_perm = (ss_b_perm / (len(unique_groups) - 1)) / (ss_w_perm / (n - len(unique_groups)))
        f_perms.append(f_perm)
    
    p_value = np.mean(np.array(f_perms) >= f_stat)
    return f_stat, p_value

groups_arr = [GROUP_MAP[s] for s in SAMPLES]
f_stat, p_perm = permanova(bc_mat, groups_arr, permutations=999)

# Plot PCoA
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.8))

for i, sample in enumerate(SAMPLES):
    group = GROUP_MAP[sample]
    color = COLORS[group]
    ax.scatter(coords[i, 0], coords[i, 1], c=color, s=60, edgecolors='black',
              linewidths=0.5, zorder=5)
    ax.annotate(sample, (coords[i, 0], coords[i, 1]), fontsize=5,
               xytext=(5, 5), textcoords='offset points', alpha=0.7)

# 95% confidence ellipses
for g in ['A_a', 'A_b', 'A_c', 'B']:
    idx_g = [i for i, s in enumerate(SAMPLES) if GROUP_MAP[s] == g]
    if len(idx_g) >= 2:
        from matplotlib.patches import Ellipse
        x_g = coords[idx_g, 0]
        y_g = coords[idx_g, 1]
        mean_x, mean_y = np.mean(x_g), np.mean(y_g)
        cov = np.cov(x_g, y_g)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        
        # 95% confidence ellipse (using chi-squared quantile)
        chi2_val = stats.chi2.ppf(0.95, 2)
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                         angle=angle, fill=False, edgecolor=COLORS[g],
                         linewidth=1.5, linestyle='--', alpha=0.7)
        ax.add_patch(ellipse)

ax.set_xlabel(f'PCoA1 ({var_explained[0]:.1f}% variance explained)')
ax.set_ylabel(f'PCoA2 ({var_explained[1]:.1f}% variance explained)')
ax.set_title('Principal Coordinates Analysis (Bray-Curtis)')

# Legend
legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLOR_LIST, ['A_a', 'A_b', 'A_c', 'B'])]
ax.legend(handles=legend_patches, loc='best', frameon=True, fontsize=7)

# PERMANOVA annotation
ax.text(0.02, 0.98, f'PERMANOVA\nF = {f_stat:.2f}, p = {p_perm:.3f}',
        transform=ax.transAxes, va='top', ha='left', fontsize=6,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

ax.grid(True, alpha=0.2)
save_fig(fig, 'fig_pcoa.png')

summary_stats['pcoa_var_explained'] = {'PC1': var_explained[0], 'PC2': var_explained[1]}
summary_stats['permanova'] = {'F': f_stat, 'p': p_perm}


# ============================================================
# 4. Beta Diversity - NMDS
# ============================================================
print("\n[4/12] Beta Diversity - NMDS (Bray-Curtis)...")

# NMDS using sklearn MDS on Bray-Curtis
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

# Use Bray-Curtis distance matrix
dist_matrix = squareform(bc_mat)  # condensed form
bc_square = bc_mat

# NMDS (MDS with metric=False)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
          normalized_stress=False, max_iter=300, eps=1e-6)
nmds_coords = mds.fit_transform(bc_square)
stress = mds.stress_

# Stress as proportion
total_dissim = np.sum(bc_square[np.triu_indices(n, k=1)] ** 2)
stress_prop = np.sqrt(stress / total_dissim) if total_dissim > 0 else 0

fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.8))

for i, sample in enumerate(SAMPLES):
    group = GROUP_MAP[sample]
    color = COLORS[group]
    ax.scatter(nmds_coords[i, 0], nmds_coords[i, 1], c=color, s=60,
              edgecolors='black', linewidths=0.5, zorder=5)
    ax.annotate(sample, (nmds_coords[i, 0], nmds_coords[i, 1]), fontsize=5,
               xytext=(5, 5), textcoords='offset points', alpha=0.7)

# Ellipses
for g in ['A_a', 'A_b', 'A_c', 'B']:
    idx_g = [i for i, s in enumerate(SAMPLES) if GROUP_MAP[s] == g]
    if len(idx_g) >= 2:
        x_g = nmds_coords[idx_g, 0]
        y_g = nmds_coords[idx_g, 1]
        mean_x, mean_y = np.mean(x_g), np.mean(y_g)
        cov = np.cov(x_g, y_g)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        chi2_val = stats.chi2.ppf(0.95, 2)
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                         angle=angle, fill=False, edgecolor=COLORS[g],
                         linewidth=1.5, linestyle='--', alpha=0.7)
        ax.add_patch(ellipse)

ax.set_xlabel('NMDS1')
ax.set_ylabel('NMDS2')
ax.set_title(f'Non-metric Multidimensional Scaling (Bray-Curtis)\nStress = {stress_prop:.3f}')

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(COLOR_LIST, ['A_a', 'A_b', 'A_c', 'B'])]
ax.legend(handles=legend_patches, loc='best', frameon=True, fontsize=7)

ax.grid(True, alpha=0.2)
save_fig(fig, 'fig_nmds.png')

summary_stats['nmds_stress'] = stress_prop


# ============================================================
# 5. Taxonomic Composition Stacked Bar (Top 20 OTUs)
# ============================================================
print("\n[5/12] Taxonomic Composition Stacked Bar...")

# No taxonomy, use OTU IDs (top 20)
otu_totals = otu_df[SAMPLES].sum(axis=1).sort_values(ascending=False)
top20_otus = otu_totals.head(20).index.tolist()

# Relative abundance of top 20
rel_top20 = rel_abund.loc[top20_otus, SAMPLES].copy()
rel_others = 100 - rel_top20.sum(axis=0)

# Build plot data
plot_data = rel_top20.copy()
plot_data.loc['Others'] = rel_others

# Color for stacked bars
stack_colors = sns.color_palette('tab20', 20) + ['#D3D3D3']

fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.0))

bottom = np.zeros(len(SAMPLES))
x = np.arange(len(SAMPLES))

for i, otu in enumerate(plot_data.index):
    vals = plot_data.loc[otu].values
    ax.bar(x, vals, bottom=bottom, color=stack_colors[i % len(stack_colors)],
           edgecolor='white', linewidth=0.3, label=otu, width=0.7)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(SAMPLES, rotation=45, ha='right')
ax.set_ylabel('Relative Abundance (%)')
ax.set_title('OTU Composition (Top 20)')

# Add group separators
ax.axvline(1.5, color='black', linewidth=0.5, linestyle=':')
ax.axvline(3.5, color='black', linewidth=0.5, linestyle=':')
ax.axvline(5.5, color='black', linewidth=0.5, linestyle=':')

# Compact legend - outside
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=5,
          ncol=1, frameon=True, title='OTU ID')

fig.tight_layout()
save_fig(fig, 'fig_taxa_composition.png')


# ============================================================
# 6. Heatmap (Top 30 OTUs)
# ============================================================
print("\n[6/12] Heatmap (Top 30 OTUs)...")

otu_totals = otu_df[SAMPLES].sum(axis=1).sort_values(ascending=False)
top30_otus = otu_totals.head(30).index.tolist()

heatmap_data = rel_abund.loc[top30_otus, SAMPLES]

# Column order: A_a, A_b, A_c, B
col_order = ['A_a_1', 'A_a_2', 'A_b_1', 'A_b_2', 'A_c_1', 'A_c_2', 'B_a', 'B_b']
heatmap_data = heatmap_data[col_order]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 7.0))

# Row clustering
row_linkage = linkage(heatmap_data.values, method='average', metric='euclidean')
row_dendro = dendrogram(row_linkage, no_plot=True)
row_order = row_dendro['leaves']
heatmap_data_sorted = heatmap_data.iloc[row_order]

sns.heatmap(heatmap_data_sorted, ax=ax, cmap='RdYlBu_r',
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Relative Abundance (%)', 'shrink': 0.5},
            xticklabels=True, yticklabels=True)

ax.set_xlabel('Sample')
ax.set_ylabel('OTU ID')
ax.set_title('Top 30 OTUs Heatmap')
ax.tick_params(axis='y', labelsize=6)

save_fig(fig, 'fig_heatmap.png')


# ============================================================
# 7. Venn/UpSet Plot
# ============================================================
print("\n[7/12] UpSet Plot (Shared/Unique OTUs)...")

# Determine OTU presence/absence per group (present if > 0 in at least one sample)
groups_for_venn = ['A_a', 'A_b', 'A_c', 'B']
group_otus = {}
for g in groups_for_venn:
    g_samples = [s for s in SAMPLES if GROUP_MAP[s] == g]
    # OTU present if found in any sample of the group
    present = otu_df[g_samples].sum(axis=1) > 0
    group_otus[g] = set(otu_df.index[present])

# All intersections
all_intersections = {}
for r in range(1, len(groups_for_venn) + 1):
    for combo in itertools.combinations(groups_for_venn, r):
        combo_set = set(combo)
        # OTUs in ALL groups of the combo and NOT in any other group
        if r == len(groups_for_venn):
            shared = set.intersection(*[group_otus[g] for g in combo])
            all_intersections[combo] = shared
        else:
            others = set(groups_for_venn) - combo_set
            shared = set.intersection(*[group_otus[g] for g in combo])
            exclusive = shared - set.union(*[group_otus[g] for g in others]) if others else shared
            all_intersections[combo] = exclusive

# Sort intersections by size
sorted_intersections = sorted(all_intersections.items(), key=lambda x: len(x[1]), reverse=True)

# Take top 15 intersections for display
top_intersections = sorted_intersections[:15]

fig, (ax_bar, ax_matrix) = plt.subplots(2, 1, figsize=(7.2, 4.5),
                                          gridspec_kw={'height_ratios': [2, 1]})

labels = []
counts = []
mat_data = []

for combo, otus in top_intersections:
    label = ' ∩ '.join(combo)
    labels.append(label)
    counts.append(len(otus))
    row = [1 if g in combo else 0 for g in groups_for_venn]
    mat_data.append(row)

# Bar chart
colors_bar = ['#4E79A7' if sum(r) == 1 else '#999999' for r in mat_data]
ax_bar.barh(range(len(counts)), counts, color=colors_bar, edgecolor='black', linewidth=0.3)
ax_bar.set_xlabel('Number of OTUs')
ax_bar.set_yticks(range(len(labels)))
ax_bar.set_yticklabels(labels, fontsize=6)
ax_bar.invert_yaxis()
ax_bar.set_title('Shared and Unique OTUs (UpSet-style)')

# Matrix
mat_array = np.array(mat_data)
for i in range(len(mat_data)):
    for j in range(len(groups_for_venn)):
        if mat_array[i, j] == 1:
            ax_matrix.plot(j, i, 'o', color=COLORS[groups_for_venn[j]], markersize=6)
        else:
            ax_matrix.plot(j, i, 'o', color='lightgray', markersize=4, alpha=0.3)

# Connect dots in same row
for i in range(len(mat_data)):
    ones = [j for j in range(len(groups_for_venn)) if mat_array[i, j] == 1]
    if len(ones) > 1:
        ax_matrix.plot([min(ones), max(ones)], [i, i], color='black', linewidth=1)

ax_matrix.set_xticks(range(len(groups_for_venn)))
ax_matrix.set_xticklabels(groups_for_venn)
ax_matrix.set_yticks([])
ax_matrix.invert_yaxis()
ax_matrix.set_xlim(-0.5, len(groups_for_venn) - 0.5)

fig.tight_layout()
save_fig(fig, 'fig_venn.png')

# Venn summary stats
venn_summary = {str(k): len(v) for k, v in all_intersections.items()}
summary_stats['venn'] = venn_summary


# ============================================================
# 8. Differential Analysis - Volcano Plot
# ============================================================
print("\n[8/12] Differential Analysis (Volcano Plot)...")

# A vs B comparison
a_samples = [s for s in SAMPLES if SUPERGROUP_MAP[GROUP_MAP[s]] == 'A']
b_samples = [s for s in SAMPLES if GROUP_MAP[s] == 'B']

# Filter: OTUs present in at least 2 samples
otu_present = (otu_df[SAMPLES] > 0).sum(axis=1) >= 2
otu_filtered = otu_df[SAMPLES].loc[otu_present]

log2fc_list = []
pval_list = []
otu_ids_list = []

for otu in otu_filtered.index:
    a_vals = otu_filtered.loc[otu, a_samples].values.astype(float)
    b_vals = otu_filtered.loc[otu, b_samples].values.astype(float)
    
    mean_a = np.mean(a_vals) + 1
    mean_b = np.mean(b_vals) + 1
    log2fc = np.log2(mean_a / mean_b)
    
    try:
        stat, pval = stats.ranksums(a_vals, b_vals)
    except:
        pval = 1.0
    
    log2fc_list.append(log2fc)
    pval_list.append(pval)
    otu_ids_list.append(otu)

volcano_df = pd.DataFrame({
    'OTU': otu_ids_list,
    'log2FC': log2fc_list,
    'pvalue': pval_list,
})
volcano_df['neg_log10_p'] = -np.log10(volcano_df['pvalue'].clip(lower=1e-10))

# Significance
volcano_df['significant'] = (volcano_df['pvalue'] < 0.05) & (abs(volcano_df['log2FC']) > 1)

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

# Non-significant
ns = volcano_df[~volcano_df['significant']]
ax.scatter(ns['log2FC'], ns['neg_log10_p'], c='gray', s=10, alpha=0.4, label='Not significant')

# Significant - enriched in A
sig_a = volcano_df[volcano_df['significant'] & (volcano_df['log2FC'] > 0)]
ax.scatter(sig_a['log2FC'], sig_a['neg_log10_p'], c='#4E79A7', s=20, alpha=0.8, label='Enriched in A')

# Significant - enriched in B
sig_b = volcano_df[volcano_df['significant'] & (volcano_df['log2FC'] < 0)]
ax.scatter(sig_b['log2FC'], sig_b['neg_log10_p'], c='#E15759', s=20, alpha=0.8, label='Enriched in B')

# Label top significant OTUs
top_sig = volcano_df[volcano_df['significant']].nlargest(5, 'neg_log10_p')
for _, row in top_sig.iterrows():
    ax.annotate(row['OTU'], (row['log2FC'], row['neg_log10_p']),
               fontsize=5, alpha=0.8, xytext=(5, 5), textcoords='offset points')

# Threshold lines
ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(-1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax.axvline(1, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

ax.set_xlabel('log2(Fold Change) [A / B]')
ax.set_ylabel('-log10(p-value)')
ax.set_title('Volcano Plot: A vs B (Wilcoxon rank-sum test)')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.2)

save_fig(fig, 'fig_differential.png')

summary_stats['volcano_A_vs_B'] = {
    'total_OTUs_tested': len(volcano_df),
    'significant_enriched_A': len(sig_a),
    'significant_enriched_B': len(sig_b),
}


# ============================================================
# 9. Random Forest Classifier
# ============================================================
print("\n[9/12] Random Forest Classifier...")

# Classify A vs B
X = otu_df[SAMPLES].T.values
y = np.array(['A' if SUPERGROUP_MAP[GROUP_MAP[s]] == 'A' else 'B' for s in SAMPLES])

# Feature selection: use OTUs with total abundance > 50
otu_filter = otu_df[SAMPLES].sum(axis=1) > 50
selected_otus = otu_df[SAMPLES].loc[otu_filter].index.tolist()
X = otu_df.loc[selected_otus, SAMPLES].T.values

rf = RandomForestClassifier(n_estimators=500, random_state=42, max_depth=3)
cv = StratifiedKFold(n_splits=min(4, len(np.unique(y))), shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')

rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=selected_otus).sort_values(ascending=False)
top20_importance = importances.head(20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

# Feature importance bar plot
colors_imp = []
for otu in top20_importance.index:
    a_mean = otu_df.loc[otu, a_samples].mean()
    b_mean = otu_df.loc[otu, b_samples].mean()
    colors_imp.append('#4E79A7' if a_mean > b_mean else '#E15759')

ax1.barh(range(len(top20_importance)), top20_importance.values, color=colors_imp,
         edgecolor='black', linewidth=0.3)
ax1.set_yticks(range(len(top20_importance)))
ax1.set_yticklabels(top20_importance.index, fontsize=5)
ax1.set_xlabel('Feature Importance')
ax1.set_title('Top 20 Discriminative OTUs')
ax1.invert_yaxis()

# CV accuracy box
ax2.boxplot([cv_scores], widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='black', linewidth=1.5))
ax2.scatter([1] * len(cv_scores), cv_scores, c='navy', s=30, zorder=5)
ax2.set_xticklabels(['A vs B'])
ax2.set_ylabel('Cross-validation Accuracy')
ax2.set_title(f'Random Forest Accuracy\nMean = {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}')
ax2.set_ylim(0, 1.1)
ax2.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

fig.tight_layout()
save_fig(fig, 'fig_random_forest.png')

summary_stats['random_forest'] = {
    'mean_accuracy': cv_scores.mean(),
    'std_accuracy': cv_scores.std(),
    'cv_scores': cv_scores.tolist(),
    'top10_features': top20_importance.head(10).to_dict(),
}


# ============================================================
# 10. Co-occurrence Network
# ============================================================
print("\n[10/12] Co-occurrence Network...")

otu_totals = otu_df[SAMPLES].sum(axis=1).sort_values(ascending=False)
top50_otus = otu_totals.head(50).index.tolist()
net_data = rel_abund.loc[top50_otus, SAMPLES]

# Spearman correlation
corr_matrix = np.zeros((len(top50_otus), len(top50_otus)))
pval_matrix = np.ones((len(top50_otus), len(top50_otus)))

for i in range(len(top50_otus)):
    for j in range(i + 1, len(top50_otus)):
        r, p = spearmanr(net_data.iloc[i].values, net_data.iloc[j].values)
        corr_matrix[i, j] = r
        corr_matrix[j, i] = r
        pval_matrix[i, j] = p
        pval_matrix[j, i] = p

# FDR correction
# Manual FDR (Benjamini-Hochberg) correction
pvals_flat = pval_matrix[np.triu_indices(len(top50_otus), k=1)]
# Benjamini-Hochberg
def bh_fdr(pvals):
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    fdr = np.zeros(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_pvals[i] * n / (i + 1)
    # Enforce monotonicity (from largest rank)
    sorted_fdr = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        sorted_fdr[i] = min(sorted_fdr[i], sorted_fdr[i + 1])
    # Cap at 1
    sorted_fdr = np.minimum(sorted_fdr, 1.0)
    result = np.zeros(n)
    for i in range(n):
        result[sorted_idx[i]] = sorted_fdr[i]
    return result

pvals_fdr = bh_fdr(pvals_flat)
pval_fdr_matrix = np.ones_like(pval_matrix)
idx = 0
for i in range(len(top50_otus)):
    for j in range(i + 1, len(top50_otus)):
        pval_fdr_matrix[i, j] = pvals_fdr[idx]
        pval_fdr_matrix[j, i] = pvals_fdr[idx]
        idx += 1

# Build network
G = nx.Graph()
for otu in top50_otus:
    G.add_node(otu)

edges = []
for i in range(len(top50_otus)):
    for j in range(i + 1, len(top50_otus)):
        if abs(corr_matrix[i, j]) > 0.6 and pval_fdr_matrix[i, j] < 0.05:
            G.add_edge(top50_otus[i], top50_otus[j],
                      weight=corr_matrix[i, j], sign='pos' if corr_matrix[i, j] > 0 else 'neg')
            edges.append((top50_otus[i], top50_otus[j], corr_matrix[i, j]))

# Node attributes
for otu in top50_otus:
    a_mean = rel_abund.loc[otu, a_samples].mean()
    b_mean = rel_abund.loc[otu, b_samples].mean()
    G.nodes[otu]['abundance'] = otu_totals[otu]
    G.nodes[otu]['preference'] = 'A' if a_mean > b_mean else 'B'

# Draw network
fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.0))

if len(G.edges()) > 0:
    pos = nx.spring_layout(G, k=2, seed=42, iterations=100)
    
    # Node sizes proportional to abundance
    node_sizes = [G.nodes[n]['abundance'] / otu_totals.max() * 300 + 50 for n in G.nodes()]
    
    # Node colors by group preference
    node_colors = [COLORS['A_a'] if G.nodes[n]['preference'] == 'A' else COLORS['B'] for n in G.nodes()]
    
    # Edge colors
    pos_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 'pos']
    neg_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] == 'neg']
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
                          edgecolors='black', linewidths=0.5, alpha=0.8)
    
    if pos_edges:
        nx.draw_networkx_edges(G, pos, edgelist=pos_edges, ax=ax, edge_color='#E15759',
                              alpha=0.5, width=1.0)
    if neg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=neg_edges, ax=ax, edge_color='#4E79A7',
                              alpha=0.5, width=1.0, style='dashed')
    
    # Labels for high-degree nodes
    degree_dict = dict(G.degree())
    high_deg = [n for n, d in degree_dict.items() if d >= 3]
    labels = {n: n for n in high_deg}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=5, font_color='black')
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['A_a'], markersize=8, label='A-preferred'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['B'], markersize=8, label='B-preferred'),
        Line2D([0], [0], color='#E15759', linewidth=1.5, label='Positive corr.'),
        Line2D([0], [0], color='#4E79A7', linewidth=1.5, linestyle='dashed', label='Negative corr.'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=7, frameon=True)

ax.set_title(f'Co-occurrence Network (|r| > 0.6, FDR < 0.05)\n{len(G.nodes())} nodes, {len(G.edges())} edges')
ax.axis('off')

save_fig(fig, 'fig_network.png')

summary_stats['network'] = {
    'nodes': len(G.nodes()),
    'edges': len(G.edges()),
    'positive_edges': len(pos_edges),
    'negative_edges': len(neg_edges),
}


# ============================================================
# 11. Distance Heatmap
# ============================================================
print("\n[11/12] Sample Distance Heatmap...")

bc_df = pd.DataFrame(bc_mat, index=SAMPLES, columns=SAMPLES)

# Cluster ordering
linkage_mat = linkage(squareform(bc_mat), method='average')
dendro = dendrogram(linkage_mat, no_plot=True)
cluster_order = [SAMPLES[i] for i in dendro['leaves']]

bc_df_clustered = bc_df.loc[cluster_order, cluster_order]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

mask = np.triu(np.ones_like(bc_df_clustered, dtype=bool), k=0)
sns.heatmap(bc_df_clustered, ax=ax, cmap='RdYlBu_r', annot=True, fmt='.3f',
            annot_kws={'size': 6}, linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Bray-Curtis Distance', 'shrink': 0.6},
            vmin=0, vmax=1)

ax.set_xlabel('Sample')
ax.set_ylabel('Sample')
ax.set_title('Bray-Curtis Distance Matrix')
ax.tick_params(axis='both', labelsize=7)

save_fig(fig, 'fig_distance_heatmap.png')


# ============================================================
# 12. Summary Statistics Table
# ============================================================
print("\n[12/12] Summary Statistics Table...")

# Build comprehensive summary
summary_rows = []

# Basic stats
summary_rows.append({'Metric': 'Total OTUs', 'Value': len(otu_df)})
summary_rows.append({'Metric': 'Total Samples', 'Value': len(SAMPLES)})

for s in SAMPLES:
    summary_rows.append({'Metric': f'{s} reads', 'Value': int(otu_df[s].sum())})
    summary_rows.append({'Metric': f'{s} OTUs', 'Value': int((otu_df[s] > 0).sum())})

# Alpha diversity summary
for metric_name in alpha_metrics:
    for g in ['A_a', 'A_b', 'A_c', 'B']:
        vals = alpha_df[alpha_df['Group'] == g][metric_name]
        summary_rows.append({
            'Metric': f'{metric_name} ({g}) mean +/- SD',
            'Value': f'{vals.mean():.3f} +/- {vals.std():.3f}'
        })
    summary_rows.append({
        'Metric': f'{metric_name} Kruskal-Wallis p-value',
        'Value': f'{kw_results[metric_name]["p"]:.4f}'
    })

# Beta diversity
summary_rows.append({'Metric': 'PCoA PC1 variance (%)', 'Value': f'{var_explained[0]:.2f}'})
summary_rows.append({'Metric': 'PCoA PC2 variance (%)', 'Value': f'{var_explained[1]:.2f}'})
summary_rows.append({'Metric': 'PERMANOVA F statistic', 'Value': f'{f_stat:.3f}'})
summary_rows.append({'Metric': 'PERMANOVA p-value', 'Value': f'{p_perm:.4f}'})
summary_rows.append({'Metric': 'NMDS stress', 'Value': f'{stress_prop:.4f}'})

# Network
summary_rows.append({'Metric': 'Network nodes', 'Value': summary_stats['network']['nodes']})
summary_rows.append({'Metric': 'Network edges', 'Value': summary_stats['network']['edges']})

# Random forest
summary_rows.append({'Metric': 'RF accuracy (mean +/- SD)', 'Value': f'{cv_scores.mean():.3f} +/- {cv_scores.std():.3f}'})

# Volcano
summary_rows.append({'Metric': 'Differential OTUs A-enriched', 'Value': summary_stats['volcano_A_vs_B']['significant_enriched_A']})
summary_rows.append({'Metric': 'Differential OTUs B-enriched', 'Value': summary_stats['volcano_A_vs_B']['significant_enriched_B']})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(RES_DIR, 'summary_statistics.csv'), index=False)
print(f"  Saved: {os.path.join(RES_DIR, 'summary_statistics.csv')}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nFigures saved in: {FIG_DIR}")
print(f"Results saved in: {RES_DIR}")
print(f"Scripts saved in: {SCRIPT_DIR}")
print("\nGenerated figures:")
for f in sorted(os.listdir(FIG_DIR)):
    if f.endswith('.png'):
        print(f"  - {f}")
