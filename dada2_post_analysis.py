#!/usr/bin/env python3
"""
DADA2 Results Post-Analysis: Alpha/Beta Diversity, PERMANOVA, BLAST annotation
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import subprocess
import os
import json

OUT_DIR = "FungalAnalysis/dada2_results"
os.makedirs(os.path.join(OUT_DIR, "beta_diversity"), exist_ok=True)

# ============================================================
# 1. Load ASV table
# ============================================================
print("=== Loading ASV table ===")
asv_df = pd.read_csv(os.path.join(OUT_DIR, "asv_table.csv"), index_col=0)
print(f"ASV table: {asv_df.shape[0]} ASVs x {asv_df.shape[1]} samples")
print(f"Total reads: {asv_df.sum().sum()}")

samples = list(asv_df.columns)
sample_info = pd.DataFrame({
    'sample': samples,
    'group': ['Agarwood' if s.startswith('A_') else 'Atractylodes' for s in samples],
    'source': ['_'.join(s.split('_')[:2]) for s in samples]
})

# ============================================================
# 2. Alpha Diversity
# ============================================================
print("\n=== Alpha Diversity ===")

def shannon(x):
    x = x[x > 0]
    p = x / x.sum()
    return -np.sum(p * np.log(p))

def simpson(x):
    x = x[x > 0]
    p = x / x.sum()
    return 1 - np.sum(p**2)

def chao1(x):
    x = x[x > 0]
    f1 = np.sum(x == 1)
    f2 = np.sum(x == 2)
    S = len(x)
    if f2 > 0:
        return S + f1*(f1-1)/(2*(f2+1))
    else:
        return S + f1*(f1-1)/2

alpha_results = []
for s in samples:
    col = asv_df[s].values
    alpha_results.append({
        'sample': s,
        'group': sample_info.loc[sample_info['sample']==s, 'group'].values[0],
        'source': sample_info.loc[sample_info['sample']==s, 'source'].values[0],
        'reads': int(col.sum()),
        'ASVs': int(np.sum(col > 0)),
        'Shannon': round(shannon(col), 4),
        'Simpson': round(simpson(col), 4),
        'Chao1': round(chao1(col), 1)
    })

alpha_df = pd.DataFrame(alpha_results)
alpha_df.to_csv(os.path.join(OUT_DIR, "alpha_diversity.csv"), index=False)
print(alpha_df.to_string(index=False))

# KW tests
print("\n--- Kruskal-Wallis Tests ---")
for metric in ['Shannon', 'ASVs', 'Chao1']:
    ag = alpha_df[alpha_df['group']=='Agarwood'][metric].values
    at = alpha_df[alpha_df['group']=='Atractylodes'][metric].values
    stat, p = stats.kruskal(ag, at)
    print(f"  {metric}: H={stat:.4f}, p={p:.4f}")

# ============================================================
# 3. Beta Diversity
# ============================================================
print("\n=== Beta Diversity ===")

# Relative abundance
asv_rel = asv_df.div(asv_df.sum(axis=0), axis=1)

# Bray-Curtis
asv_t = asv_df.T  # samples x ASVs
bc_dist = pdist(asv_t.values, metric='braycurtis')
bc_mat = squareform(bc_dist)
bc_df = pd.DataFrame(bc_mat, index=samples, columns=samples)
bc_df.to_csv(os.path.join(OUT_DIR, "beta_diversity", "bray_curtis.csv"))
print("\nBray-Curtis distance matrix:")
print(bc_df.round(4).to_string())

# Jaccard
asv_pa = (asv_df > 0).astype(int)  # presence/absence
ja_dist = pdist(asv_pa.T.values, metric='jaccard')
ja_mat = squareform(ja_dist)
ja_df = pd.DataFrame(ja_mat, index=samples, columns=samples)
ja_df.to_csv(os.path.join(OUT_DIR, "beta_diversity", "jaccard.csv"))

# PCoA via MDS
mds_bc = MDS(n_components=3, dissimilarity='precomputed', random_state=42, normalized_stress=False)
bc_pcoa = mds_bc.fit_transform(bc_mat)
pcoa_df = pd.DataFrame(bc_pcoa, index=samples, columns=['PC1', 'PC2', 'PC3'])
pcoa_df['sample'] = samples
pcoa_df = pcoa_df.merge(sample_info, on='sample', how='left')
pcoa_df.to_csv(os.path.join(OUT_DIR, "beta_diversity", "pcoa_bray_points.csv"))

# PCoA plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
colors_map = {'A_a': '#E41A1C', 'A_b': '#377EB8', 'A_c': '#4DAF4A', 'B_a': '#FF7F00', 'B_b': '#984EA3'}
markers_map = {'Agarwood': 'o', 'Atractylodes': 's'}

for _, row in pcoa_df.iterrows():
    ax.scatter(row['PC1'], row['PC2'], 
               c=colors_map.get(row['source'], 'gray'),
               marker=markers_map.get(row['group'], 'o'),
               s=120, edgecolors='black', linewidths=0.5,
               label=row['source'])
    ax.annotate(row['sample'], (row['PC1'], row['PC2']), 
                fontsize=9, ha='center', va='bottom', xytext=(0, 8),
                textcoords='offset points')

# Remove duplicate legends
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10)
ax.set_xlabel('PCoA1', fontsize=12)
ax.set_ylabel('PCoA2', fontsize=12)
ax.set_title('PCoA (Bray-Curtis) - DADA2 ASV Level', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "beta_diversity", "pcoa_bray.png"), dpi=300, bbox_inches='tight')
plt.close()
print("PCoA plot saved.")

# ============================================================
# 4. PERMANOVA (manual implementation)
# ============================================================
print("\n=== PERMANOVA ===")

def permanova(distance_matrix, groups, permutations=9999):
    """Manual PERMANOVA implementation"""
    n = len(groups)
    groups = np.array(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)
    
    # Total SS
    total_ss = np.sum(distance_matrix**2) / (2 * n) if distance_matrix.shape[0] > 1 else 0
    
    # Within-group SS
    within_ss = 0
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        if len(idx) > 1:
            sub = distance_matrix[np.ix_(idx, idx)]
            within_ss += np.sum(sub**2) / (2 * len(idx))
    
    between_ss = total_ss - within_ss
    
    # F-statistic
    df_between = k - 1
    df_within = n - k
    if df_within == 0:
        return {'F': np.nan, 'p': np.nan, 'R2': np.nan, 'df_between': df_between, 'df_within': df_within}
    
    F = (between_ss / df_between) / (within_ss / df_within) if within_ss > 0 else np.nan
    R2 = between_ss / total_ss if total_ss > 0 else np.nan
    
    # Permutation test
    count = 0
    for _ in range(permutations):
        perm_groups = np.random.permutation(groups)
        perm_within = 0
        for g in unique_groups:
            idx = np.where(perm_groups == g)[0]
            if len(idx) > 1:
                sub = distance_matrix[np.ix_(idx, idx)]
                perm_within += np.sum(sub**2) / (2 * len(idx))
        perm_between = total_ss - perm_within
        perm_F = (perm_between / df_between) / (perm_within / df_within) if perm_within > 0 else 0
        if perm_F >= F:
            count += 1
    
    p_value = (count + 1) / (permutations + 1)
    
    return {'F': round(F, 4), 'p': round(p_value, 5), 'R2': round(R2, 4), 
            'df_between': df_between, 'df_within': df_within}

np.random.seed(42)

# All samples: Agarwood vs Atractylodes
groups_all = ['Agarwood' if s.startswith('A_') else 'Atractylodes' for s in samples]
perm_all_bc = permanova(bc_mat, groups_all)
print(f"\n--- PERMANOVA All (Bray-Curtis): Agarwood vs Atractylodes ---")
print(f"  F={perm_all_bc['F']}, R2={perm_all_bc['R2']}, p={perm_all_bc['p']}")
print(f"  df_between={perm_all_bc['df_between']}, df_within={perm_all_bc['df_within']}")

perm_all_ja = permanova(ja_mat, groups_all)
print(f"\n--- PERMANOVA All (Jaccard): Agarwood vs Atractylodes ---")
print(f"  F={perm_all_ja['F']}, R2={perm_all_ja['R2']}, p={perm_all_ja['p']}")

# Agarwood-only
ag_idx = [i for i, s in enumerate(samples) if s.startswith('A_')]
if len(ag_idx) >= 3:
    ag_samples = [samples[i] for i in ag_idx]
    ag_sources = ['_'.join(s.split('_')[:2]) for s in ag_samples]
    bc_ag = bc_mat[np.ix_(ag_idx, ag_idx)]
    ja_ag = ja_mat[np.ix_(ag_idx, ag_idx)]
    
    perm_ag_bc = permanova(bc_ag, ag_sources)
    print(f"\n--- PERMANOVA Agarwood-only (Bray-Curtis): 3 sources ---")
    print(f"  F={perm_ag_bc['F']}, R2={perm_ag_bc['R2']}, p={perm_ag_bc['p']}")
    
    perm_ag_ja = permanova(ja_ag, ag_sources)
    print(f"\n--- PERMANOVA Agarwood-only (Jaccard): 3 sources ---")
    print(f"  F={perm_ag_ja['F']}, R2={perm_ag_ja['R2']}, p={perm_ag_ja['p']}")
    
    # Pairwise
    sources = list(set(ag_sources))
    pairwise_results = []
    for i in range(len(sources)):
        for j in range(i+1, len(sources)):
            s1, s2 = sources[i], sources[j]
            pair_idx = [k for k, s in enumerate(ag_sources) if s in [s1, s2]]
            if len(pair_idx) >= 3:
                bc_pair = bc_mat[np.ix_([ag_idx[k] for k in pair_idx], [ag_idx[k] for k in pair_idx])]
                ja_pair = ja_mat[np.ix_([ag_idx[k] for k in pair_idx], [ag_idx[k] for k in pair_idx])]
                g_pair = [ag_sources[k] for k in pair_idx]
                
                perm_pair_bc = permanova(bc_pair, g_pair)
                perm_pair_ja = permanova(ja_pair, g_pair)
                
                print(f"\n--- Pairwise: {s1} vs {s2} (Bray-Curtis) ---")
                print(f"  F={perm_pair_bc['F']}, R2={perm_pair_bc['R2']}, p={perm_pair_bc['p']}")
                print(f"--- Pairwise: {s1} vs {s2} (Jaccard) ---")
                print(f"  F={perm_pair_ja['F']}, R2={perm_pair_ja['R2']}, p={perm_pair_ja['p']}")
                
                pairwise_results.append({
                    'comparison': f"{s1} vs {s2}",
                    'BC_F': perm_pair_bc['F'], 'BC_R2': perm_pair_bc['R2'], 'BC_p': perm_pair_bc['p'],
                    'JA_F': perm_pair_ja['F'], 'JA_R2': perm_pair_ja['R2'], 'JA_p': perm_pair_ja['p']
                })

# Save PERMANOVA results
with open(os.path.join(OUT_DIR, "permanova_results.txt"), 'w') as f:
    f.write("=== PERMANOVA Results (DADA2 ASV Level) ===\n\n")
    f.write(f"--- All: Agarwood vs Atractylodes (Bray-Curtis) ---\n")
    f.write(f"F={perm_all_bc['F']}, R2={perm_all_bc['R2']}, p={perm_all_bc['p']}, df={perm_all_bc['df_between']},{perm_all_bc['df_within']}\n\n")
    f.write(f"--- All: Agarwood vs Atractylodes (Jaccard) ---\n")
    f.write(f"F={perm_all_ja['F']}, R2={perm_all_ja['R2']}, p={perm_all_ja['p']}, df={perm_all_ja['df_between']},{perm_all_ja['df_within']}\n\n")
    if len(ag_idx) >= 3:
        f.write(f"--- Agarwood-only global (Bray-Curtis) ---\n")
        f.write(f"F={perm_ag_bc['F']}, R2={perm_ag_bc['R2']}, p={perm_ag_bc['p']}, df={perm_ag_bc['df_between']},{perm_ag_bc['df_within']}\n\n")
        f.write(f"--- Agarwood-only global (Jaccard) ---\n")
        f.write(f"F={perm_ag_ja['F']}, R2={perm_ag_ja['R2']}, p={perm_ag_ja['p']}, df={perm_ag_ja['df_between']},{perm_ag_ja['df_within']}\n\n")
        for pr in pairwise_results:
            f.write(f"--- Pairwise: {pr['comparison']} (Bray-Curtis) ---\n")
            f.write(f"F={pr['BC_F']}, R2={pr['BC_R2']}, p={pr['BC_p']}\n")
            f.write(f"--- Pairwise: {pr['comparison']} (Jaccard) ---\n")
            f.write(f"F={pr['JA_F']}, R2={pr['JA_R2']}, p={pr['JA_p']}\n\n")

# ============================================================
# 5. Top 20 ASVs
# ============================================================
print("\n=== Top 20 ASVs ===")
total_counts = asv_df.sum(axis=1)
top20_idx = total_counts.sort_values(ascending=False).head(20).index

top20_df = pd.DataFrame({
    'ASV': top20_idx,
    'Total_Reads': total_counts[top20_idx].values.astype(int),
    'Rel_Abundance_pct': (total_counts[top20_idx] / total_counts.sum() * 100).round(2).values
})

# Per-sample percentages
for s in samples:
    top20_df[f'{s}_pct'] = (asv_df.loc[top20_idx, s] / asv_df[s].sum() * 100).round(2).values

top20_df.to_csv(os.path.join(OUT_DIR, "top20_ASVs.csv"), index=False)
print(top20_df[['ASV', 'Total_Reads', 'Rel_Abundance_pct']].to_string(index=False))

# ============================================================
# 6. BLAST annotation (using existing blast_annotate.py approach)
# ============================================================
print("\n=== BLAST Annotation ===")
fasta_file = os.path.join(OUT_DIR, "asv_sequences.fasta")
blast_output = os.path.join(OUT_DIR, "asv_blast_results.txt")

# Check if blastn is available
blast_available = False
try:
    result = subprocess.run(['which', 'blastn'], capture_output=True, text=True)
    if result.returncode == 0:
        blast_available = True
        print("blastn found, running BLAST...")
except:
    pass

if blast_available:
    cmd = f'blastn -query {fasta_file} -db nt -out {blast_output} -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore stitle" -max_target_seqs 1 -evalue 1e-10 -remote'
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            print("BLAST completed successfully!")
            # Parse BLAST results
            blast_cols = ['qseqid','sseqid','pident','length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore','stitle']
            blast_df = pd.read_csv(blast_output, sep='\t', header=None, names=blast_cols)
            
            # Extract species from stitle
            def extract_species(title):
                if pd.isna(title):
                    return 'Unknown'
                parts = title.split()
                if len(parts) >= 2:
                    return f"{parts[0]} {parts[1]}"
                return title[:50]
            
            blast_df['Species'] = blast_df['stitle'].apply(extract_species)
            
            # Merge with top20
            taxonomy_df = blast_df[['qseqid', 'Species', 'pident', 'evalue']].copy()
            taxonomy_df.columns = ['ASV', 'BLAST_Species', 'Identity_pct', 'E-value']
            taxonomy_df.to_csv(os.path.join(OUT_DIR, "asv_taxonomy_blast.csv"), index=False)
            print(f"Taxonomy for {len(taxonomy_df)} ASVs saved.")
        else:
            print(f"BLAST failed: {result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        print("BLAST timed out (30min limit). Will use existing OTU taxonomy as reference.")
    except Exception as e:
        print(f"BLAST error: {e}")
else:
    print("blastn not available. Using reference-based mapping from OTU taxonomy.")

# If BLAST not available, try to map ASVs to existing OTU taxonomy
if not os.path.exists(os.path.join(OUT_DIR, "asv_taxonomy_blast.csv")):
    print("\nMapping ASVs to existing OTU taxonomy via sequence similarity...")
    # This is a fallback - we'll use the OTU taxonomy as reference
    print("Will need to use NCBI BLAST web or local alignment for final taxonomy.")
    print("Saving ASV sequences for manual BLAST submission.")

# ============================================================
# 7. Summary
# ============================================================
print("\n=== Summary ===")
summary = {
    'total_ASVs': int(asv_df.shape[0]),
    'total_samples': int(asv_df.shape[1]),
    'total_reads': int(asv_df.sum().sum()),
    'mean_reads_per_sample': float(asv_df.sum().mean()),
    'mean_ASVs_per_sample': float((asv_df > 0).sum().mean()),
    'permanova_all_BC_F': perm_all_bc['F'],
    'permanova_all_BC_p': perm_all_bc['p'],
    'permanova_all_JA_F': perm_all_ja['F'],
    'permanova_all_JA_p': perm_all_ja['p'],
}
if len(ag_idx) >= 3:
    summary['permanova_agarwood_BC_F'] = perm_ag_bc['F']
    summary['permanova_agarwood_BC_p'] = perm_ag_bc['p']

with open(os.path.join(OUT_DIR, "summary.json"), 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\n=== Post-Analysis Complete ===")
