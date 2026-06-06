#!/usr/bin/env python3
"""
Generate correct taxonomy from BLAST cache and update figures with species names
"""
import json, csv, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

RESULTS_DIR = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/results"
FIGURES_DIR = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/figures"
SCRIPTS_DIR = "/home/zhhq/.openclaw/workspace-coder/FungalAnalysis/scripts"

# === Corrected taxonomy based on BLAST results ===
# Taxonomy manually corrected to proper fungal classification
TAXONOMY = {
    "OTU_0001": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Talaromyces","Species":"Talaromyces pseudofuniculosus"},
    "OTU_0002": {"Kingdom":"Fungi","Phylum":"Mucoromycota","Class":"Mucoromycetes","Order":"Mucorales","Family":"Rhizopodaceae","Genus":"Rhizopus","Species":"Rhizopus arrhizus"},
    "OTU_0003": {"Kingdom":"Fungi","Phylum":"Mucoromycota","Class":"Mucoromycetes","Order":"Mucorales","Family":"Rhizopodaceae","Genus":"Rhizopus","Species":"Rhizopus sp."},
    "OTU_0004": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Aspergillus","Species":"Aspergillus tubingensis"},
    "OTU_0005": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Sordariomycetes","Order":"Hypocreales","Family":"Hypocreaceae","Genus":"Trichoderma","Species":"Trichoderma asperellum"},
    "OTU_0006": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Dothideomycetes","Order":"Pleosporales","Family":"Pleosporaceae","Genus":"Curvularia","Species":"Curvularia lunata"},
    "OTU_0007": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Talaromyces","Species":"Talaromyces verruculosus"},
    "OTU_0008": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Aspergillus","Species":"Aspergillus sydowii"},
    "OTU_0009": {"Kingdom":"Fungi","Phylum":"Basidiomycota","Class":"Agaricomycetes","Order":"Agaricales","Family":"Cortinariaceae","Genus":"Gymnopilus","Species":"Gymnopilus lepidotus"},
    "OTU_0010": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Aspergillus","Species":"Aspergillus sp."},
    "OTU_0011": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Aspergillus","Species":"Aspergillus flavus"},
    "OTU_0012": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Dothideomycetes","Order":"Pleosporales","Family":"Pleosporaceae","Genus":"Curvularia","Species":"Curvularia lunata"},
    "OTU_0013": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Dothideomycetes","Order":"Pleosporales","Family":"Pleosporaceae","Genus":"Curvularia","Species":"Curvularia lunata"},
    "OTU_0014": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Aspergillus","Species":"Aspergillus sp."},
    "OTU_0015": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Talaromyces","Species":"Talaromyces verruculosus"},
    "OTU_0016": {"Kingdom":"Fungi","Phylum":"Ascomycota","Class":"Eurotiomycetes","Order":"Eurotiales","Family":"Trichocomaceae","Genus":"Talaromyces","Species":"Talaromyces verruculosus"},
}

SAMPLE_GROUPS = {
    "A_a_1": "A_a", "A_a_2": "A_a",
    "A_b_1": "A_b", "A_b_2": "A_b",
    "A_c_1": "A_c", "A_c_2": "A_c",
    "B_a": "B", "B_b": "B"
}

GROUP_COLORS = {"A_a": "#4E79A7", "A_b": "#F28E2B", "A_c": "#59A14F", "B": "#E15759"}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

def main():
    # Load OTU table
    otu_df = pd.read_csv(os.path.join(RESULTS_DIR, "otu_table.csv"), index_col=0)
    samples = [c for c in otu_df.columns if c != 'Total']
    
    # Write corrected taxonomy
    tax_levels = ["Kingdom","Phylum","Class","Order","Family","Genus","Species"]
    with open(os.path.join(RESULTS_DIR, "taxonomy.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["OTU_ID"] + tax_levels)
        for i in range(1, max(int(x.replace('OTU_','')) for x in otu_df.index) + 1):
            otu_id = f"OTU_{i:04d}"
            if otu_id in TAXONOMY:
                tax = TAXONOMY[otu_id]
                w.writerow([otu_id] + [tax.get(l, 'unidentified') for l in tax_levels])
            else:
                w.writerow([otu_id] + ['Fungi','unidentified','unidentified','unidentified','unidentified','unidentified','unidentified'])
    
    print(f"Taxonomy written for {len(otu_df)} OTUs")
    
    # === Figure: Species composition stacked bar (top 16 annotated + others) ===
    print("Generating species composition figure...")
    
    # Get top OTUs and their species names
    otu_totals = otu_df[samples].sum(axis=1).sort_values(ascending=False)
    top_n = 20
    
    # Merge OTUs with same species
    species_abund = defaultdict(lambda: defaultdict(float))
    annotated = 0
    for otu_id in otu_totals.index[:top_n]:
        if otu_id in TAXONOMY:
            sp = TAXONOMY[otu_id]["Species"]
            # Italicize genus
            genus = TAXONOMY[otu_id]["Genus"]
            label = f"{genus} sp." if "sp." in sp else sp
        else:
            label = otu_id
            annotated += 1
        for s in samples:
            species_abund[label][s] += otu_df.loc[otu_id, s] if otu_id in otu_df.index else 0
    
    # Add "Others"
    top_otu_ids = list(otu_totals.index[:top_n])
    for s in samples:
        other = otu_df.loc[~otu_df.index.isin(top_otu_ids), s].sum() if s in otu_df.columns else 0
        species_abund["Others"][s] += other
    
    # Build relative abundance dataframe
    species_df = pd.DataFrame(species_abund, index=samples).T
    species_df = species_df.div(species_df.sum(axis=0), axis=1) * 100
    
    # Sort by abundance
    mean_vals = species_df.mean(axis=1)
    species_df = species_df.loc[mean_vals.sort_values(ascending=False).index]
    
    # Color palette (ColorBrewer Set3 + extra)
    n_species = len(species_df)
    cmap = plt.cm.get_cmap('tab20', max(20, n_species))
    colors = [cmap(i) for i in range(n_species)]
    
    fig, ax = plt.subplots(figsize=(7.2, 4))
    species_df.T.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='none', width=0.8)
    
    # Add group labels
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=7)
    
    # Add group brackets
    group_ranges = {"A_a": (0,1), "A_b": (2,3), "A_c": (4,5), "B": (6,7)}
    for grp, (start, end) in group_ranges.items():
        label = {"A_a": "Agarwood A", "A_b": "Agarwood B", "A_c": "Agarwood C", "B": "Atractylodes"}[grp]
        mid = (start + end) / 2
        ax.annotate(label, xy=(mid, -8), ha='center', fontsize=7, 
                    fontweight='bold', color=GROUP_COLORS[grp])
    
    ax.set_ylabel("Relative Abundance (%)")
    ax.set_xlabel("")
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=5, ncol=1, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_species_composition.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig_species_composition.png")
    
    # === Figure: Phylum-level composition ===
    print("Generating phylum composition figure...")
    
    phylum_abund = defaultdict(lambda: defaultdict(float))
    for otu_id in otu_df.index:
        if otu_id in TAXONOMY:
            ph = TAXONOMY[otu_id]["Phylum"]
        else:
            ph = "unidentified"
        for s in samples:
            phylum_abund[ph][s] += otu_df.loc[otu_id, s]
    
    phylum_df = pd.DataFrame(phylum_abund, index=samples).T
    phylum_df = phylum_df.div(phylum_df.sum(axis=0), axis=1) * 100
    phylum_df['mean'] = phylum_df.mean(axis=1)
    phylum_df = phylum_df.sort_values('mean', ascending=False).drop(columns=['mean'])
    
    phylum_colors = {"Ascomycota": "#4E79A7", "Mucoromycota": "#F28E2B", 
                     "Basidiomycota": "#59A14F", "unidentified": "#BAB0AC"}
    
    fig, ax = plt.subplots(figsize=(7.2, 3))
    phylum_df.T.plot(kind='bar', stacked=True, ax=ax, 
                     color=[phylum_colors.get(p, "#BAB0AC") for p in phylum_df.index],
                     edgecolor='none', width=0.8)
    ax.set_xticklabels(samples, rotation=45, ha='right')
    ax.set_ylabel("Relative Abundance (%)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(os.path.join(FIGURES_DIR, "fig_phylum_composition.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig_phylum_composition.png")
    
    # === Figure: Genus-level heatmap with species annotations ===
    print("Generating genus heatmap with taxonomy...")
    
    top30 = otu_totals.index[:30]
    heat_data = otu_df.loc[top30, samples].copy()
    
    # Convert to relative abundance per sample
    sample_totals = heat_data.sum(axis=0)
    heat_rel = heat_data.div(sample_totals, axis=1) * 100
    
    # Replace OTU IDs with species names
    new_labels = []
    for oid in top30:
        if oid in TAXONOMY:
            new_labels.append(TAXONOMY[oid]["Species"])
        else:
            new_labels.append(oid)
    heat_rel.index = new_labels
    
    # Cluster rows
    if heat_rel.shape[0] > 2:
        row_linkage = linkage(pdist(heat_rel.values, metric='euclidean'), method='average')
        row_dendro = dendrogram(row_linkage, no_plot=True)
        row_order = row_dendro['leaves']
        heat_rel = heat_rel.iloc[row_order]
    
    fig, ax = plt.subplots(figsize=(7.2, 8))
    sns.heatmap(heat_rel, cmap='YlOrRd', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Relative Abundance (%)', 'shrink': 0.5},
                xticklabels=True, yticklabels=True)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=5, style='italic')
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.savefig(os.path.join(FIGURES_DIR, "fig_heatmap_annotated.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig_heatmap_annotated.png")
    
    # === Figure: Pathogen/toxigenic risk assessment ===
    print("Generating risk assessment figure...")
    
    # Known toxigenic/pathogenic genera
    risk_genera = {
        "Aspergillus": "Toxigenic (aflatoxin/ochratoxin)",
        "Rhizopus": "Opportunistic pathogen",
        "Curvularia": "Plant pathogen / Opportunistic",
        "Talaromyces": "Potential toxigenic",
    }
    
    risk_data = defaultdict(lambda: defaultdict(float))
    for otu_id in otu_df.index:
        if otu_id in TAXONOMY:
            genus = TAXONOMY[otu_id]["Genus"]
            if genus in risk_genera:
                for s in samples:
                    risk_data[genus][s] += otu_df.loc[otu_id, s]
    
    if risk_data:
        risk_df = pd.DataFrame(risk_data, index=samples).T
        risk_rel = risk_df.div(risk_df.sum(axis=0), axis=1) * 100
        
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        risk_rel.T.plot(kind='bar', stacked=True, ax=ax,
                       color=['#E15759', '#F28E2B', '#76B7B2', '#59A14F'],
                       edgecolor='none', width=0.8)
        ax.set_xticklabels(samples, rotation=45, ha='right')
        ax.set_ylabel("Relative Abundance (%)")
        ax.set_title("Potential Toxigenic/Pathogenic Fungi", fontsize=10)
        
        # Add risk labels in legend
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [f"{l} ({risk_genera[l]})" for l in labels]
        ax.legend(handles, new_labels, frameon=False, fontsize=5, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.savefig(os.path.join(FIGURES_DIR, "fig_risk_assessment.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved fig_risk_assessment.png")
    
    # === Summary table of top 16 species ===
    print("Generating species summary...")
    with open(os.path.join(RESULTS_DIR, "species_summary.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["OTU_ID", "Species", "Phylum", "Class", "Total_Reads", 
                     "A_a_mean%", "A_b_mean%", "A_c_mean%", "B_mean%", "Risk"])
        for oid in otu_totals.index[:16]:
            if oid in TAXONOMY:
                tax = TAXONOMY[oid]
                species = tax["Species"]
                phylum = tax["Phylum"]
                cls = tax["Class"]
                genus = tax["Genus"]
                risk = risk_genera.get(genus, "Low")
            else:
                species = oid
                phylum = "?"
                cls = "?"
                risk = "?"
            
            total = otu_df.loc[oid, 'Total'] if 'Total' in otu_df.columns else otu_df.loc[oid].sum()
            
            # Group means
            a_a = otu_df.loc[oid, ['A_a_1','A_a_2']].mean()
            a_b = otu_df.loc[oid, ['A_b_1','A_b_2']].mean()
            a_c = otu_df.loc[oid, ['A_c_1','A_c_2']].mean()
            b_mean = otu_df.loc[oid, ['B_a','B_b']].mean()
            all_total = otu_df.loc[oid, samples].sum()
            
            w.writerow([oid, species, phylum, cls, int(total),
                       f"{a_a/all_total*100:.1f}", f"{a_b/all_total*100:.1f}",
                       f"{a_c/all_total*100:.1f}", f"{b_mean/all_total*100:.1f}", risk])
    
    print("  Saved species_summary.csv")
    print("\nAll taxonomy figures generated!")

if __name__ == "__main__":
    main()
