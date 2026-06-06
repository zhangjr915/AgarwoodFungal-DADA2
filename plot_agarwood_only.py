#!/usr/bin/env python3
"""
沉香内生真菌多样性分析 —— 仅 A 组（沉香）6 个样本
排除 B 组（苍术）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

AGARWOOD = ["A_a_1","A_a_2","A_b_1","A_b_2","A_c_1","A_c_2"]
COLS_GROUP = {"Source A":"#E64B35","Source B":"#4DBBD5","Source C":"#00A087"}
SOURCE_MAP = {"A_a_1":"Source A","A_a_2":"Source A",
              "A_b_1":"Source B","A_b_2":"Source B",
              "A_c_1":"Source C","A_c_2":"Source C"}

OUT = "FungalAnalysis/figures/v2"
os.makedirs(OUT, exist_ok=True)

def get_source(sample):
    return SOURCE_MAP.get(sample, "Unknown")

# ============================================================
# Load data
# ============================================================
stats = pd.read_csv("FungalAnalysis/dada2_results/dada2_stats.csv", index_col=0)
stats = stats.loc[AGARWOOD]

alpha = pd.read_csv("FungalAnalysis/dada2_results/alpha_diversity.csv")
alpha = alpha[alpha["group"]=="Agarwood"].copy()
alpha["Source"] = alpha["source"].map({"A_a":"Source A","A_b":"Source B","A_c":"Source C"})
alpha["Source_cat"] = pd.Categorical(alpha["Source"], categories=["Source A","Source B","Source C"])
alpha = alpha.sort_values("Source_cat")

top20 = pd.read_csv("FungalAnalysis/dada2_results/top20_ASVs.csv")
tax = pd.read_csv("FungalAnalysis/dada2_results/asv_taxonomy_blast.csv")

asv_tab = pd.read_csv("FungalAnalysis/dada2_results/asv_table.csv", index_col=0)
asv_tab = asv_tab[AGARWOOD]

pct_cols = [f"{s}_pct" for s in AGARWOOD]

# ============================================================
# Fig 1 — DADA2 Pipeline Stages
# ============================================================
print(">>> Fig 1: DADA2 pipeline bar chart")
stages = ["input","filtered","denoisedF","merged","nonchim"]
stage_labels = ["Raw Reads","Filtered","Denoised","Merged","Non-chimeric"]
colors_stages = sns.color_palette("Set2", 5)

fig, ax = plt.subplots(figsize=(10,6))
x = np.arange(len(AGARWOOD))
w = 0.15
for i, (st, lab) in enumerate(zip(stages, stage_labels)):
    vals = stats[st].values
    ax.bar(x + i*w, vals, w, label=lab, color=colors_stages[i], edgecolor='black', linewidth=0.3)
ax.set_xticks(x + 2*w)
ax.set_xticklabels(AGARWOOD)
ax.set_ylabel("Number of Reads")
ax.set_title("DADA2 Processing Pipeline — Agarwood Samples")
ax.legend(title="Stage", loc='upper right')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f'{int(x):,}'))
sns.despine()
fig.savefig(f"{OUT}/fig1_dada2_pipeline.png")
plt.close()

# Fig 1b — Retention rate
print(">>> Fig 1b: Retention rate")
retention = (stats["nonchim"]/stats["input"]*100).values
fig, ax = plt.subplots(figsize=(8,5))
bar_colors = [COLS_GROUP[get_source(s)] for s in AGARWOOD]
bars = ax.bar(AGARWOOD, retention, color=bar_colors, edgecolor='black', linewidth=0.3, width=0.6)
for b, v in zip(bars, retention):
    ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
ax.set_ylim(0,100)
ax.set_ylabel("Retention Rate (%)")
ax.set_title("Read Retention Rate After DADA2 Pipeline")
sns.despine()
fig.savefig(f"{OUT}/fig1b_retention_rate.png")
plt.close()

# ============================================================
# Fig 2 — Alpha diversity boxplots
# ============================================================
print(">>> Fig 2: Alpha diversity")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metrics = [("Shannon","Shannon Index"), ("Simpson","Simpson Index"),
           ("Chao1","Chao1 Richness"), ("ASVs","Observed ASVs")]
source_order = ["Source A","Source B","Source C"]

for ax, (metric, ylabel) in zip(axes.flat, metrics):
    bp = sns.boxplot(data=alpha, x="Source_cat", y=metric, order=source_order,
                     palette=COLS_GROUP, width=0.5, ax=ax, fliersize=0)
    sns.stripplot(data=alpha, x="Source_cat", y=metric, order=source_order,
                  color="black", size=6, jitter=0.15, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.set_xticklabels(["Source A\n(A_a)","Source B\n(A_b)","Source C\n(A_c)"])

fig.suptitle("Alpha Diversity — Agarwood Endophytic Fungi", fontsize=16, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_alpha_diversity.png")
plt.close()

# ============================================================
# Fig 3 — PCoA (Bray-Curtis)
# ============================================================
print(">>> Fig 3: PCoA Bray-Curtis")
from sklearn.decomposition import PCA as skPCA

# Bray-Curtis via scipy
asv_t = asv_tab.T.values  # samples x ASVs
# Relative abundance for Bray-Curtis
asv_rel = asv_t / asv_t.sum(axis=1, keepdims=True)

# Manual Bray-Curtis distance matrix
n = asv_rel.shape[0]
bc = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        bc[i,j] = np.sum(np.abs(asv_rel[i] - asv_rel[j])) / np.sum(asv_rel[i] + asv_rel[j])

# PCoA via eigendecomposition of centered distance matrix
n_samples = bc.shape[0]
H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
A = -0.5 * bc**2
G = H @ A @ H
eigvals, eigvecs = np.linalg.eigh(G)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Keep positive eigenvalues only
pos = eigvals > 0
eigvals_pos = eigvals[pos]
eigvecs_pos = eigvecs[pos]

coords = eigvecs_pos[:2].T * np.sqrt(eigvals_pos[:2])
var_exp = eigvals_pos[:2] / eigvals_pos.sum() * 100

pcoa_df = pd.DataFrame({
    "Sample": AGARWOOD,
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "Source": [get_source(s) for s in AGARWOOD]
})

fig, ax = plt.subplots(figsize=(8, 7))
for src in source_order:
    sub = pcoa_df[pcoa_df["Source"]==src]
    ax.scatter(sub["PC1"], sub["PC2"], c=COLS_GROUP[src], s=120,
               edgecolors='black', linewidth=1, label=src, zorder=5)
    for _, row in sub.iterrows():
        ax.annotate(row["Sample"], (row["PC1"], row["PC2"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight='bold')

# Ellipses
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
for src in source_order:
    sub = pcoa_df[pcoa_df["Source"]==src]
    if len(sub) >= 2:
        cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
        w = max(abs(sub["PC1"].max() - sub["PC1"].min()) * 1.2, 0.01)
        h = max(abs(sub["PC2"].max() - sub["PC2"].min()) * 1.2, 0.01)
        ell = Ellipse((cx, cy), w, h, alpha=0.15, color=COLS_GROUP[src], linewidth=2, linestyle='--')
        ax.add_patch(ell)

ax.set_xlabel(f"PCoA1 ({var_exp[0]:.1f}%)")
ax.set_ylabel(f"PCoA2 ({var_exp[1]:.1f}%)")
ax.set_title("PCoA (Bray-Curtis) — Agarwood Fungal Communities")
ax.legend(title="Source")
ax.axhline(0, color='grey', linestyle=':', linewidth=0.5)
ax.axvline(0, color='grey', linestyle=':', linewidth=0.5)
sns.despine()
fig.savefig(f"{OUT}/fig3_pcoa_braycurtis.png")
plt.close()

# ============================================================
# Fig 3b — PCoA (Jaccard)
# ============================================================
print(">>> Fig 3b: PCoA Jaccard")
# Jaccard: presence/absence
asv_pa = (asv_t > 0).astype(float)
jac = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a_and_b = np.sum(np.minimum(asv_pa[i], asv_pa[j]))
        a_or_b = np.sum(np.maximum(asv_pa[i], asv_pa[j]))
        jac[i,j] = 1 - a_and_b / a_or_b if a_or_b > 0 else 0

A_j = -0.5 * jac**2
G_j = H @ A_j @ H
eigvals_j, eigvecs_j = np.linalg.eigh(G_j)
idx_j = np.argsort(eigvals_j)[::-1]
eigvals_j = eigvals_j[idx_j]; eigvecs_j = eigvecs_j[:, idx_j]
pos_j = eigvals_j > 0
eigvals_j_pos = eigvals_j[pos_j]; eigvecs_j_pos = eigvecs_j[pos_j]
coords_j = eigvecs_j_pos[:2].T * np.sqrt(eigvals_j_pos[:2])
var_exp_j = eigvals_j_pos[:2] / eigvals_j_pos.sum() * 100

pcoa_j_df = pd.DataFrame({
    "Sample": AGARWOOD, "PC1": coords_j[:,0], "PC2": coords_j[:,1],
    "Source": [get_source(s) for s in AGARWOOD]
})

fig, ax = plt.subplots(figsize=(8, 7))
for src in source_order:
    sub = pcoa_j_df[pcoa_j_df["Source"]==src]
    ax.scatter(sub["PC1"], sub["PC2"], c=COLS_GROUP[src], s=120,
               edgecolors='black', linewidth=1, label=src, zorder=5)
    for _, row in sub.iterrows():
        ax.annotate(row["Sample"], (row["PC1"], row["PC2"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight='bold')
    if len(sub) >= 2:
        cx, cy = sub["PC1"].mean(), sub["PC2"].mean()
        w = max(abs(sub["PC1"].max()-sub["PC1"].min())*1.2, 0.01)
        h = max(abs(sub["PC2"].max()-sub["PC2"].min())*1.2, 0.01)
        ell = Ellipse((cx,cy), w, h, alpha=0.15, color=COLS_GROUP[src], linewidth=2, linestyle='--')
        ax.add_patch(ell)

ax.set_xlabel(f"PCoA1 ({var_exp_j[0]:.1f}%)")
ax.set_ylabel(f"PCoA2 ({var_exp_j[1]:.1f}%)")
ax.set_title("PCoA (Jaccard) — Agarwood Fungal Communities")
ax.legend(title="Source")
ax.axhline(0, color='grey', linestyle=':', linewidth=0.5)
ax.axvline(0, color='grey', linestyle=':', linewidth=0.5)
sns.despine()
fig.savefig(f"{OUT}/fig3b_pcoa_jaccard.png")
plt.close()

# ============================================================
# Fig 4 — Species Composition Stacked Bar
# ============================================================
print(">>> Fig 4: Species composition")
comp = top20[["ASV"] + pct_cols].copy()
comp = comp.merge(tax[["ASV","Clean_Species"]], on="ASV", how="left")

# Merge same species
comp_agg = comp.groupby("Clean_Species")[pct_cols].sum().reset_index()
comp_agg["Total"] = comp_agg[pct_cols].sum(axis=1)
comp_agg = comp_agg.sort_values("Total", ascending=False)

# Top 10 + Other
top10 = comp_agg.head(10).copy()
other = comp_agg.iloc[10:][pct_cols].sum()
other_row = pd.DataFrame({"Clean_Species":["Other"],
                           **{c:[other[c]] for c in pct_cols},
                           "Total":[other.sum()]})
top10 = pd.concat([top10, other_row], ignore_index=True)

bar_df = top10.melt(id_vars=["Clean_Species","Total"],
                     value_vars=pct_cols, var_name="Sample", value_name="Abundance")
bar_df["Sample"] = bar_df["Sample"].str.replace("_pct","")

species_order = top10.sort_values("Total", ascending=True)["Clean_Species"].tolist()
n_sp = len(species_order)
species_colors = sns.color_palette("Set3", n_sp)
color_map = dict(zip(species_order, species_colors))

fig, ax = plt.subplots(figsize=(10, 6))
bottom = np.zeros(len(AGARWOOD))
for sp in species_order:
    vals = bar_df[bar_df["Clean_Species"]==sp].set_index("Sample").reindex(AGARWOOD)["Abundance"].values
    ax.bar(AGARWOOD, vals, bottom=bottom, label=sp, color=color_map[sp],
           edgecolor='black', linewidth=0.2, width=0.65)
    bottom += vals

ax.set_ylabel("Relative Abundance (%)")
ax.set_title("Fungal Community Composition — Agarwood Samples")
ax.legend(title="Species", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
sns.despine()
fig.savefig(f"{OUT}/fig4_species_composition.png")
plt.close()

# ============================================================
# Fig 5 — Heatmap Top 20 ASVs
# ============================================================
print(">>> Fig 5: Heatmap")
heat_mat = top20[pct_cols].copy()
heat_mat.index = top20["ASV"]
heat_mat.columns = AGARWOOD

# Add species to row labels
species_labels = tax.set_index("ASV").loc[top20["ASV"], "Clean_Species"].values
heat_mat.index = [f"{a} ({s[:20]})" for a, s in zip(top20["ASV"], species_labels)]

# Z-score by row
heat_z = heat_mat.sub(heat_mat.mean(axis=1), axis=0).div(heat_mat.std(axis=1), axis=0)

# Column colors
col_colors = pd.Series([COLS_GROUP[get_source(s)] for s in AGARWOOD], index=AGARWOOD)

g = sns.clustermap(heat_z, cmap="RdBu_r", center=0,
                    col_colors=col_colors.values,
                    figsize=(10, 9),
                    linewidths=0.5,
                    cbar_kws={"label":"Z-score"})
g.fig.suptitle("Heatmap of Top 20 ASVs — Agarwood Samples", y=1.02, fontsize=14, fontweight='bold')

# Legend for column colors
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=COLS_GROUP[s], label=s) for s in source_order]
g.ax_heatmap.legend(handles=legend_elements, title="Source",
                     bbox_to_anchor=(1.35, 1), loc='upper left')

g.savefig(f"{OUT}/fig5_heatmap_top20.png")
plt.close()

# ============================================================
# Fig 6 — Top 10 species by source (grouped bar)
# ============================================================
print(">>> Fig 6: Top 10 species by source")
top10_sp = comp_agg.head(10)["Clean_Species"].tolist()

src_means = []
for src_name, src_samples in [("Source A", ["A_a_1","A_a_2"]),
                                ("Source B", ["A_b_1","A_b_2"]),
                                ("Source C", ["A_c_1","A_c_2"])]:
    sp_cols = [f"{s}_pct" for s in src_samples]
    means = comp_agg.set_index("Clean_Species").loc[top10_sp, sp_cols].mean(axis=1)
    for sp, val in zip(top10_sp, means):
        src_means.append({"Species": sp, "Source": src_name, "Abundance": val})

bar6 = pd.DataFrame(src_means)
bar6["Species"] = pd.Categorical(bar6["Species"], categories=top10_sp, ordered=True)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=bar6, x="Species", y="Abundance", hue="Source",
            palette=COLS_GROUP, ax=ax, edgecolor='black', linewidth=0.3)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
ax.set_ylabel("Mean Relative Abundance (%)")
ax.set_title("Top 10 Fungal Species — Mean Abundance by Source")
ax.legend(title="Source")
sns.despine()
fig.savefig(f"{OUT}/fig6_top10_species_by_source.png")
plt.close()

# ============================================================
# Fig 7 — Bray-Curtis distance heatmap
# ============================================================
print(">>> Fig 7: BC distance heatmap")
fig, ax = plt.subplots(figsize=(7, 6))
mask = np.triu(np.ones_like(bc, dtype=bool), k=1)
sns.heatmap(bc, mask=mask, annot=True, fmt='.3f',
            xticklabels=AGARWOOD, yticklabels=AGARWOOD,
            cmap="YlOrRd", ax=ax, square=True,
            cbar_kws={"label":"Bray-Curtis Dissimilarity", "shrink": 0.8})
ax.set_title("Bray-Curtis Dissimilarity — Agarwood Samples")
fig.savefig(f"{OUT}/fig7_braycurtis_distance.png")
plt.close()

# ============================================================
# Fig 8 — Rarefaction curves
# ============================================================
print(">>> Fig 8: Rarefaction curves")
fig, ax = plt.subplots(figsize=(8, 6))
np.random.seed(42)
for i, sample in enumerate(AGARWOOD):
    counts = asv_tab[sample].values.astype(int)
    total = counts.sum()
    depths = np.linspace(1, total, min(100, total), dtype=int)
    richness = []
    for d in depths:
        # Probabilistic rarefaction
        probs = counts / counts.sum()
        sampled = np.random.multinomial(d, probs)
        richness.append(np.sum(sampled > 0))
    ax.plot(depths, richness, color=COLS_GROUP[get_source(sample)],
            linewidth=1.5, label=f"{sample} ({get_source(sample)})")

ax.set_xlabel("Sequencing Depth (Reads)")
ax.set_ylabel("Observed ASVs")
ax.set_title("Rarefaction Curves — Agarwood Samples")
ax.legend(title="Sample", fontsize=9)
sns.despine()
fig.savefig(f"{OUT}/fig8_rarefaction_curve.png")
plt.close()

# ============================================================
# Fig 9 — Genus pie charts (3 panels)
# ============================================================
print(">>> Fig 9: Genus pie charts")
comp_agg["Genus"] = comp_agg["Clean_Species"].apply(lambda x: x.split()[0])
genus_agg = comp_agg.groupby("Genus")[pct_cols].sum().reset_index()
genus_agg["Total"] = genus_agg[pct_cols].sum(axis=1)
genus_agg = genus_agg.sort_values("Total", ascending=False)

top_genera = genus_agg.head(8)["Genus"].tolist()

pie_colors = sns.color_palette("Set2", 9)

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for idx, (src_name, src_samples) in enumerate([("Source A (A_a)", ["A_a_1","A_a_2"]),
                                                  ("Source B (A_b)", ["A_b_1","A_b_2"]),
                                                  ("Source C (A_c)", ["A_c_1","A_c_2"])]):
    sp_cols = [f"{s}_pct" for s in src_samples]
    vals = genus_agg.set_index("Genus").loc[:, sp_cols].mean(axis=1)

    top_vals = vals[vals.index.isin(top_genera)].sort_values(ascending=False)
    other = vals[~vals.index.isin(top_genera)].sum()
    if other > 0:
        top_vals["Other"] = other

    colors_p = pie_colors[:len(top_vals)]
    wedges, texts, autotexts = axes[idx].pie(
        top_vals.values, labels=top_vals.index, autopct='%1.1f%%',
        colors=colors_p, startangle=90, pctdistance=0.82,
        textprops={'fontsize': 8})
    for t in autotexts:
        t.set_fontsize(7)
    axes[idx].set_title(src_name, fontsize=14, fontweight='bold')

fig.suptitle("Genus-Level Composition — Agarwood Endophytic Fungi", fontsize=16, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_genus_pie_charts.png")
plt.close()

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("沉香（Agarwood）6 样本分析总结")
print("="*60)
print(f"\n样本列表: {AGARWOOD}")
print(f"\n--- DADA2 保留率 ---")
for s in AGARWOOD:
    ret = stats.loc[s, "nonchim"] / stats.loc[s, "input"] * 100
    print(f"  {s}: {ret:.1f}% ({int(stats.loc[s,'nonchim']):,} / {int(stats.loc[s,'input']):,})")

print(f"\n--- Alpha 多样性 ---")
for _, row in alpha.iterrows():
    print(f"  {row['sample']}: Shannon={row['Shannon']:.3f}, Simpson={row['Simpson']:.3f}, "
          f"Chao1={row['Chao1']:.0f}, ASVs={row['ASVs']}")

print(f"\n--- 优势物种 (Top 5) ---")
for _, row in comp_agg.head(5).iterrows():
    print(f"  {row['Clean_Species']}: {row['Total']/comp_agg['Total'].sum()*100:.1f}%")

print(f"\n所有图表已保存至: {OUT}/")
print("完成！")
