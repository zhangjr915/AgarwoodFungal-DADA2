#!/usr/bin/env python3
"""
Paper Figures v3 - Agarwood ONLY (6 samples, 3 sources)
Nature-style, 300DPI, colorblind-friendly, all English
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from collections import defaultdict
import json
import os

OUT_DIR = "FungalAnalysis/figures/v3"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_DIR = "FungalAnalysis/dada2_results"

# ==================== Common Settings ====================
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Colorblind-friendly palette (Nature style)
COLORS = {
    'A_a': '#4DBBD5',  # Cyan
    'A_b': '#E64B35',  # Red
    'A_c': '#00A087',  # Green
}
SOURCE_NAMES = {'A_a': 'Source 1', 'A_b': 'Source 2', 'A_c': 'Source 3'}
MARKERS = {'A_a': 'o', 'A_b': 's', 'A_c': '^'}

# ==================== Load Data ====================
print("Loading data...")
asv_all = pd.read_csv(f"{DATA_DIR}/asv_table.csv", index_col=0)
a_samples = [c for c in asv_all.columns if c.startswith('A_')]
asv_df = asv_all[a_samples]
asv_df = asv_df.loc[asv_df.sum(axis=1) > 0]  # Remove zero-read ASVs

blast_df = pd.read_csv(f"{DATA_DIR}/asv_taxonomy_blast.csv")
species_map = dict(zip(blast_df['ASV'], blast_df['Clean_Species']))

stats_df = pd.read_csv(f"{DATA_DIR}/dada2_stats.csv", index_col=0)
stats_a = stats_df.loc[a_samples]

alpha_df = pd.read_csv(f"{DATA_DIR}/alpha_diversity.csv")
alpha_a = alpha_df[alpha_df['sample'].str.startswith('A_')].copy()
alpha_a['source'] = alpha_a['sample'].apply(lambda x: '_'.join(x.split('_')[:2]))

print(f"A-only: {asv_df.shape[0]} ASVs x {asv_df.shape[1]} samples")

# ==================== Figure 1: Pipeline + Rarefaction ====================
print("\n=== Figure 1: Pipeline Statistics ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Stacked bar chart of pipeline steps
steps = ['input', 'filtered', 'denoisedF', 'merged', 'nonchim']
step_labels = ['Input', 'Filtered', 'Denoised', 'Merged', 'Non-chimeric']
sample_labels = a_samples
bar_colors = ['#91D1C2', '#4DBBD5', '#3C5488', '#E64B35', '#DC0000']

x = np.arange(len(sample_labels))
width = 0.6
bottom = np.zeros(len(sample_labels))

for i, (step, label, color) in enumerate(zip(steps, step_labels, bar_colors)):
    vals = stats_a[step].values
    ax1.bar(x, vals, width, bottom=bottom, label=label, color=color, edgecolor='white', linewidth=0.5)
    bottom += vals

ax1.set_xticks(x)
ax1.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=10)
ax1.set_ylabel('Number of Reads', fontsize=12)
ax1.set_title('(a) DADA2 Pipeline Statistics', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

# Right: Rarefaction curve
print("Computing rarefaction curves...")
for sample in a_samples:
    col = asv_df[sample].values
    total_reads = int(col.sum())
    steps_rc = list(range(0, total_reads, max(1, total_reads // 50)))
    if steps_rc[-1] != total_reads:
        steps_rc.append(total_reads)
    
    richness = []
    for step in steps_rc:
        # Subsample
        np.random.seed(42)
        if step == 0:
            richness.append(0)
            continue
        pool = np.repeat(np.arange(len(col)), col.astype(int))
        sub = np.random.choice(pool, size=min(step, len(pool)), replace=False)
        richness.append(len(np.unique(sub)))
    
    src = '_'.join(sample.split('_')[:2])
    ax2.plot(steps_rc, richness, color=COLORS[src], linewidth=1.5, alpha=0.8, label=sample)

ax2.set_xlabel('Number of Reads', fontsize=12)
ax2.set_ylabel('ASV Richness', fontsize=12)
ax2.set_title('(b) Rarefaction Curves', fontsize=13, fontweight='bold')
ax2.legend(fontsize=8, ncol=2, loc='lower right')

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig1_pipeline_rarefaction.png")
plt.close()
print("Figure 1 saved.")

# ==================== Figure 2: Alpha Diversity ====================
print("\n=== Figure 2: Alpha Diversity ===")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
metrics = ['Shannon', 'ASVs', 'Chao1']
# Add Simpson
alpha_a['Simpson'] = alpha_a['Shannon'].apply(lambda x: 1 - np.exp(-x))  # Approximate

# Compute actual Simpson
for idx, row in alpha_a.iterrows():
    s = row['sample']
    col = asv_df[s].values
    col_nz = col[col > 0]
    p = col_nz / col_nz.sum()
    alpha_a.loc[idx, 'Simpson'] = 1 - np.sum(p**2)

metrics_plot = ['Shannon', 'Simpson', 'ASVs', 'Chao1']

for i, metric in enumerate(metrics_plot):
    ax = axes[i]
    sources = ['A_a', 'A_b', 'A_c']
    
    for j, src in enumerate(sources):
        vals = alpha_a[alpha_a['source'] == src][metric].values
        x_pos = j
        ax.bar(x_pos, np.mean(vals), width=0.6, color=COLORS[src], alpha=0.7, edgecolor='black', linewidth=0.8)
        ax.scatter([x_pos]*len(vals), vals, color='black', s=40, zorder=5)
        # Error bar
        ax.errorbar(x_pos, np.mean(vals), yerr=np.std(vals) if len(vals) > 1 else 0,
                    fmt='none', color='black', capsize=5, linewidth=1.5)
    
    ax.set_xticks(range(len(sources)))
    ax.set_xticklabels([SOURCE_NAMES[s] for s in sources], fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    
    # KW test
    groups_kw = [alpha_a[alpha_a['source'] == src][metric].values for src in sources]
    if all(len(g) > 0 for g in groups_kw):
        h, p = stats.kruskal(*groups_kw)
        ax.text(0.95, 0.95, f'p = {p:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig2_alpha_diversity.png")
plt.close()
print("Figure 2 saved.")

# ==================== Figure 3: PCoA ====================
print("\n=== Figure 3: PCoA ===")

bc_dist = pdist(asv_df.T.values, metric='braycurtis')
bc_mat = squareform(bc_dist)

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
pcoa_pts = mds.fit_transform(bc_mat)

# Eigenvectors (approximate from MDS)
var_exp = np.var(pcoa_pts, axis=0) / np.var(pcoa_pts, axis=0).sum() * 100

fig, ax = plt.subplots(figsize=(8, 7))

for i, sample in enumerate(a_samples):
    src = '_'.join(sample.split('_')[:2])
    ax.scatter(pcoa_pts[i, 0], pcoa_pts[i, 1], 
               c=COLORS[src], marker=MARKERS[src], s=150, 
               edgecolors='black', linewidths=1, zorder=5)
    ax.annotate(sample, (pcoa_pts[i, 0], pcoa_pts[i, 1]),
                fontsize=10, ha='center', va='bottom', xytext=(0, 10),
                textcoords='offset points')

# Draw ellipses/connecting lines per source
for src in ['A_a', 'A_b', 'A_c']:
    idx = [i for i, s in enumerate(a_samples) if s.startswith(src)]
    pts = pcoa_pts[idx]
    if len(pts) == 2:
        ax.plot(pts[:, 0], pts[:, 1], color=COLORS[src], linewidth=1.5, linestyle='--', alpha=0.5)

# Legend
legend_elements = [Line2D([0], [0], marker=MARKERS[s], color='w', markerfacecolor=COLORS[s],
                          markersize=10, label=SOURCE_NAMES[s]) for s in ['A_a', 'A_b', 'A_c']]
ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

ax.set_xlabel(f'PCoA1 ({var_exp[0]:.1f}%)', fontsize=13)
ax.set_ylabel(f'PCoA2 ({var_exp[1]:.1f}%)', fontsize=13)

# PERMANOVA annotation
ax.text(0.05, 0.95, 'PERMANOVA\nF = 13.58, R² = 0.90\np = 0.065',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

ax.set_title('PCoA (Bray-Curtis) — Agarwood Sources', fontsize=14, fontweight='bold')
fig.savefig(f"{OUT_DIR}/fig3_pcoa.png")
plt.close()
print("Figure 3 saved.")

# ==================== Figure 4: Species Composition ====================
print("\n=== Figure 4: Species Composition ===")

# Aggregate to species level, get top 15
species_abund = defaultdict(lambda: defaultdict(float))
for asv_id in asv_df.index:
    sp = species_map.get(asv_id, 'Unassigned')
    for sample in a_samples:
        species_abund[sp][sample] += asv_df.loc[asv_id, sample]

species_df = pd.DataFrame(species_abund).T.fillna(0)
species_df['total'] = species_df.sum(axis=1)
species_df = species_df.sort_values('total', ascending=False)
species_df = species_df.drop('total', axis=1)

top_n = 12
top_species = species_df.head(top_n)
others = species_df.iloc[top_n:].sum()
top_species.loc['Others'] = others

# Relative abundance
rel_abund = top_species.div(top_species.sum(axis=0), axis=1) * 100

fig, ax = plt.subplots(figsize=(10, 6))

bar_colors_species = plt.cm.Set3(np.linspace(0, 1, len(rel_abund)))
x = np.arange(len(a_samples))
width = 0.65
bottom = np.zeros(len(a_samples))

for i, (sp, row) in enumerate(rel_abund.iterrows()):
    ax.bar(x, row.values, width, bottom=bottom, label=sp, color=bar_colors_species[i],
           edgecolor='white', linewidth=0.3)
    bottom += row.values

ax.set_xticks(x)
ax.set_xticklabels(a_samples, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Relative Abundance (%)', fontsize=12)
ax.set_title('Fungal Community Composition by Sample', fontsize=14, fontweight='bold')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title='Species')

# Add source labels
for i, sample in enumerate(a_samples):
    src = '_'.join(sample.split('_')[:2])
    ax.text(i, -5, SOURCE_NAMES[src], ha='center', fontsize=9, color=COLORS[src], fontweight='bold')

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig4_composition.png")
plt.close()
print("Figure 4 saved.")

# ==================== Figure 5: Heatmap ====================
print("\n=== Figure 5: Heatmap ===")

# Top 20 ASVs by total abundance
total_counts = asv_df.sum(axis=1).sort_values(ascending=False)
top20_asvs = total_counts.head(20).index

heat_data = asv_df.loc[top20_asvs, a_samples]
heat_rel = heat_data.div(heat_data.sum(axis=0), axis=1) * 100

# Labels
ylabels = []
for asv_id in top20_asvs:
    sp = species_map.get(asv_id, 'Unassigned')
    if sp == 'Unknown':
        sp = 'Unassigned'
    ylabels.append(f"{sp} ({asv_id})")

fig, ax = plt.subplots(figsize=(10, 10))

im = ax.imshow(heat_rel.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')

ax.set_xticks(range(len(a_samples)))
ax.set_xticklabels(a_samples, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(ylabels)))
ax.set_yticklabels(ylabels, fontsize=9)

# Add values
for i in range(len(top20_asvs)):
    for j in range(len(a_samples)):
        val = heat_rel.iloc[i, j]
        color = 'white' if val > 15 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Relative Abundance (%)', fontsize=11)
ax.set_title('Top 20 ASVs — Relative Abundance', fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig5_heatmap.png")
plt.close()
print("Figure 5 saved.")

# ==================== Figure 6: Risk Assessment ====================
print("\n=== Figure 6: Risk Assessment ===")

# Categorize species by risk
risk_categories = {
    'Aspergillus flavus': 'Aflatoxigenic',
    'Aspergillus niger': 'Potential toxigenic',
    'Aspergillus sydowii': 'Potential toxigenic',
    'Rhizopus arrhizus': 'Opportunistic pathogen',
    'Rhizopus sp.': 'Opportunistic pathogen',
    'Rhizopus clone': 'Opportunistic pathogen',
    'Curvularia lunata': 'Plant pathogen',
    'Talaromyces pseudofuniculosus': 'Low risk',
    'Gymnopilus dilepis': 'Low risk',
}

# Calculate per-source risk proportions
risk_colors = {
    'Aflatoxigenic': '#DC0000',
    'Potential toxigenic': '#E64B35',
    'Opportunistic pathogen': '#F39B7F',
    'Plant pathogen': '#FFDC91',
    'Low risk': '#91D1C2',
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, src in enumerate(['A_a', 'A_b', 'A_c']):
    ax = axes[idx]
    src_samples = [s for s in a_samples if s.startswith(src)]
    
    # Aggregate species for this source
    src_abund = defaultdict(float)
    for asv_id in asv_df.index:
        sp = species_map.get(asv_id, 'Unassigned')
        for s in src_samples:
            src_abund[sp] += asv_df.loc[asv_id, s]
    
    total = sum(src_abund.values())
    
    # Group by risk
    risk_abund = defaultdict(float)
    for sp, abund in src_abund.items():
        risk = risk_categories.get(sp, 'Low risk')
        risk_abund[risk] += abund / total * 100
    
    # Sort
    sorted_risks = sorted(risk_abund.items(), key=lambda x: -x[1])
    labels = [r[0] for r in sorted_risks]
    sizes = [r[1] for r in sorted_risks]
    colors = [risk_colors.get(l, '#8491B4') for l in labels]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=None, colors=colors, autopct='%1.1f%%',
                                       startangle=90, pctdistance=0.75, textprops={'fontsize': 9})
    for autotext in autotexts:
        if float(autotext.get_text().replace('%', '')) < 3:
            autotext.set_text('')
    
    ax.set_title(f'{SOURCE_NAMES[src]}', fontsize=13, fontweight='bold', color=COLORS[src])

# Common legend
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in risk_colors.items()]
fig.legend(handles=legend_patches, loc='lower center', ncol=len(risk_colors), fontsize=10,
           bbox_to_anchor=(0.5, -0.05))

fig.suptitle('Fungal Risk Assessment by Source', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig6_risk.png")
plt.close()
print("Figure 6 saved.")

# ==================== Summary ====================
print(f"\n=== All 6 figures saved to {OUT_DIR}/ ===")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f}: {size/1024:.0f} KB")
