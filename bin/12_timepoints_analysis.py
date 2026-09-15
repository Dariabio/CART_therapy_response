'''
2) Unsupervised clustering и DEG
-показать heatmap групп product, d7, d28 c Unsupervised clustering 
- показать heatmap групп product, d7, d28 c DEG, где сравнение идет в отношении группы product
3) Молекулярные пути в КАРах
-Проанализировать GO_Biological process (молекулярные пути) в product, d7, d28 (вне зависимости от ответчиков и неответчиков) - представить в виде heatmap, barplot или как будет более visually impactful
'''

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp


import argparse

# Arguments

parser = argparse.ArgumentParser(
    description="Unsupervised clustering and timepoints comparisons"
)

parser.add_argument(
    "--expr_all_vst",
    required=True
) # expr_all_vst

parser.add_argument(
    "--counts",
    required=True
) # raw counts


parser.add_argument(
    "--metadata",
    required=True
) # meta_analysis.tsv

parser.add_argument(
    "--output_dir",
    required=True
)

parser.add_argument(
    "--figures",
    required=True
)

args = parser.parse_args()

# Output directory

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.figures, exist_ok=True)

# Load data vst and all meta data
vst_df = pd.read_csv(
    args.expr_all_vst,
    sep="\t",
    index_col=0
)

meta = pd.read_csv(
    args.metadata,
    sep="\t",
    index_col=0
)

# TIMEPOINT ORDER

timepoint_order = [
    "product",
    "d7 after",
    "d28 after"
]


category_palette = {
    "product": "#4C72B0",
    "d7 after": "#DD8452",
    "d28 after": "#55A868"
}

# MATCH VST AND METADATA SAMPLES

common_samples = [
    sample
    for sample in meta.index
    if sample in vst_df.index
]

meta = meta.loc[common_samples].copy()
vst_df = vst_df.loc[common_samples].copy()

assert list(meta.index) == list(vst_df.index)

# 1. Unsupervised clustering

# filter top genes with high variance
gene_variances = vst_df.var(axis=0)
top_500_genes = gene_variances.sort_values(ascending=False).head(500).index
vst_filtered = vst_df[top_500_genes]

# columns samples rows genes
heatmap_data = vst_filtered.T

# Mapping colours and samples
col_colors = meta['timepoint'].map(category_palette)

# CLUSTERMAP

# z_score=1 scaling by genes 

g = sns.clustermap(
    heatmap_data, 
    metric="correlation", 
    method="ward", 
    cmap="vlag",      
    z_score=1,        
    col_colors=col_colors, 
    figsize=(12, 10),
    yticklabels=False,
    xticklabels=True
)

# legends
legend_elements = [Patch(facecolor=color, label=label) for label, color in category_palette.items()]

plt.legend(
        handles=legend_elements, 
        title="Timepoints", 
        bbox_to_anchor=(1.2, 1), 
        loc='upper left'
    )

g.fig.suptitle('Unsupervised clustering: Top 500 variable genes', y=1.02, fontsize=14)

# 6. Saving
heatmap_save_path = os.path.join(args.figures, "unsupervised_clustering_topgenes.png")
plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=300)
plt.close()

# 2. DE analysis by timepoints (vs product) and clustering 

counts = pd.read_csv(
    args.counts,
    sep="\t",
    index_col=0
)

# Match samples

common_samples = [
        sample
        for sample in meta.index
        if sample in counts.index
    ]


meta_tp = meta.loc[
        common_samples
    ].copy()

counts_tp = counts.loc[
        common_samples].copy()

# Make sure order is identical

assert list(meta_tp.index) == list(
        counts_tp.index)

# Remove genes with zero counts

keep_genes = (
        counts_tp.sum(axis=0) > 0
    )

counts_tp = counts_tp.loc[
        :,
        keep_genes.values
    ]

# Metadata factors

meta_tp["timepoint"] = (
        meta_tp["timepoint"]
        .astype(str)
    )


meta_tp["batch_nexp"] = (
        meta_tp["batch_nexp"]
        .astype(str)
    )

# Make timepoint categorical
meta_tp["timepoint"] = pd.Categorical(
    meta_tp["timepoint"],
    categories=timepoint_order,
    ordered=True
)

# Design

design_formula = (
f"~ batch_nexp + timepoint"
    )


# DESeq2

dds = DeseqDataSet(
        counts=counts_tp,
        metadata=meta_tp,
        design=design_formula
    )

dds.deseq2()

# Contrasts

comparisons = {
    "d7_vs_product": {
        "contrast": ["timepoint", "d7 after", "product"],
        "tp_name": "d7_after_vs_product"
    },
    "d28_vs_product": {
        "contrast": ["timepoint", "d28 after", "product"],
        "tp_name": "d28_after_vs_product"
    }
}

for comp_id, info in comparisons.items():
    
    # Stats per comparison
    ds = DeseqStats(
        dds,
        contrast=info["contrast"]
    )
    
    # p-values and fold changes
    ds.summary()
    results = ds.results_df.copy()

    # PATHWS
    results_file = os.path.join(args.output_dir, f"DESeq2_{info['tp_name']}.tsv")
    deg_file = os.path.join(args.output_dir, f"DEGs_{info['tp_name']}.tsv")
    results.to_csv(results_file, sep="\t")
    print(f"All results saved to {results_file}")
    
    # Filter DEGs
    deg = results[
        (results["padj"] < 0.05) & 
        (results["log2FoldChange"].abs() > 1)
    ].copy()
    deg.to_csv(deg_file, sep="\t")

# 3. Heatmap witg degs genes

# Merge DEGs

# 1. Unique DEGs
deg_genes = set()

for info in comparisons.values():
    deg_file = os.path.join(args.output_dir, f"DEGs_{info['tp_name']}.tsv")
    if os.path.exists(deg_file):
        df_deg = pd.read_csv(deg_file, sep="\t", index_col=0)
        deg_genes.update(df_deg.index)

deg_genes = list(deg_genes)
print(f"Total unique DEGs found across both comparisons: {len(deg_genes)}")

if len(deg_genes) == 0:
    print("Warning: No DEGs found. Heatmap cannot be generated.")
else:
    # 2. Filter vst
    vst_deg = vst_df[deg_genes].T

    # 3. Sort samples: product -> d7 -> d28
    meta_tp['timepoint'] = pd.Categorical(
        meta_tp['timepoint'], 
        categories=['product', 'd7 after', 'd28 after'], 
        ordered=True
    )
    # Sort
    sorted_samples = meta_tp.sort_values('timepoint').index
    vst_deg_sorted = vst_deg[sorted_samples]

    # 4. Update colours
    col_colors_sorted = meta_tp.loc[sorted_samples, 'timepoint'].map(category_palette)

    # 5. Clustermap
    g_deg = sns.clustermap(
        vst_deg_sorted, 
        metric="correlation", 
        method="ward", 
        cmap="vlag",       
        z_score=0,        
        col_colors=col_colors_sorted, 
        col_cluster=False, 
        row_cluster=True,  
        figsize=(12, 10),
        yticklabels=False, 
        xticklabels=True   
    )

    # Add legend
    legend_elements = [Patch(facecolor=color, label=label) for label, color in category_palette.items()]
    plt.legend(
        handles=legend_elements, 
        title="Timepoints", 
        bbox_to_anchor=(1.2, 1), 
        loc='upper left'
    )

    g_deg.fig.suptitle(f'DEG Expression Heatmap ({len(deg_genes)} genes)', y=1.02, fontsize=14)

    # 6. Saving
    heatmap_save_path = os.path.join(args.figures, "deg_heatmap_chronological.png")
    plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"DEG heatmap successfully saved to {heatmap_save_path}")

# 4. GO pathways

# Run GSEA

gsea_timepoints = [
    "d7_after_vs_product",
    "d28_after_vs_product"
]


for tp_name in gsea_timepoints:

    input_file = os.path.join(
        args.output_dir,
        f"DESeq2_{tp_name}.tsv"
    )


    df = pd.read_csv(
        input_file,
        sep="\t",
        index_col=0
    )

    df.index.name = "gene"

    df = df.reset_index()

    df = df[
        ["gene", "log2FoldChange"]
    ].dropna()

    df = df.drop_duplicates(
        subset="gene", keep='first'
    )

    # Use log2FoldChange as metric
    #We add micro-noise and sort (from the highest LFC among responders to the lowest)
    np.random.seed(42)
    
    df["rank_metric"] = (
        df["log2FoldChange"]
        +
        np.random.normal(
            0,
            1e-9,
            len(df)
        )
    )

    df = df.sort_values(
        "rank_metric",
        ascending=False)


    ranked = df[["gene", "rank_metric"]]


    outdir = os.path.join(
        args.output_dir,
        "global_gsea",
        tp_name
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )

    # public db
    public_sets = ['GO_Biological_Process_2021']
            
    for db in public_sets:
        pr = gp.prerank(
        rnk = ranked,
        gene_sets=db,  # database
        outdir=outdir,
        permutation_num=1000,
        min_size=15,   # for big db higher
        max_size=500,
        seed=42, verbose=True)

    # Save all results
 

        all_results = pr.res2d.copy()


        all_results_file = os.path.join(
            outdir,
            f"all_results_{db}_{tp_name}.tsv"
        )


        all_results.to_csv(
            all_results_file,
            sep="\t",
            index=False
        )
                
    # Filter significant (FDR < 0.05)
        significant_terms = pr.res2d[pr.res2d['FDR q-val'] < 0.05]
        significant_terms.to_csv(
        os.path.join(outdir, f"significant_results_{db}_{tp_name}.tsv"), 
        sep="\t", index=False)


