import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import gseapy as gp
from matplotlib.colors import TwoSlopeNorm, Normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score




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
).T


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
    "product": "#6ECFFF",
    "d7 after": "#FFC400",
    "d28 after": "#CB3CFF"
}

aggressiv_palette = {'aggressive': "#7C20C7FF", 
                     'indolent': "#8E9B88"}

bone_marrow_palette = {'bone_marrow_positive': "#FFC3E6", 
               'bone_marrow_negative': "#FFF785"}

lesion_palette = {'widespread': "#B83E09", 
                  'localized': "#8BE8EB"}

response_palette = {'responder': "#00FF33", 
                  'nonresponder': "#FF0303"}



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

def unsup_cluster(n=50, labelsize=10):
    # filter top genes with high variance
    gene_variances = vst_df.var(axis=0)
    top_100_genes = gene_variances.sort_values(ascending=False).head(n).index
    vst_filtered = vst_df[top_100_genes]

    # columns samples rows genes
    heatmap_data = vst_filtered.T

# Mapping colours and samples
#col_colors = meta['timepoint'].map(category_palette)

    col_colors = pd.DataFrame({
        'Timepoint': meta['timepoint'].map(category_palette),
        'Aggressiveness': meta['aggressiveness'].map(aggressiv_palette),
        'Bone marrow': meta['bone_marrow'].map(bone_marrow_palette),
        'Lesion': meta['lesion'].map(lesion_palette),
        'Response':meta['response'].map(response_palette)
    })
    col_colors = col_colors.loc[heatmap_data.columns]
# CLUSTERMAP

# z_score=1 scaling by genes 

    g = sns.clustermap(
    heatmap_data, 
    metric="correlation", 
    method="average", 
    cmap="vlag",      
    z_score=1,        
    col_colors=col_colors, 
    figsize=(14, 12),
    yticklabels=True,
    xticklabels=False,
    linewidths=0,
    dendrogram_ratio=(0.12, 0.08),
    colors_ratio=0.025,
    cbar_pos=(0.03, 0.82, 0.02, 0.12)
)


    def make_legend_handles(palette, title):
        handles = [
        Patch(
            facecolor=color,
            edgecolor='none',
            label=str(label)
        )
        for label, color in palette.items()
    ]

        return handles


# Timepoint
    timepoint_handles = make_legend_handles(
    category_palette,
    "Timepoint"
)

# Aggressiveness
    aggressiveness_handles = make_legend_handles(
    aggressiv_palette,
    "Aggressiveness"
)

# Bone marrow
    bone_marrow_handles = make_legend_handles(
    bone_marrow_palette,
    "Bone marrow"
)

# Lesion
    lesion_handles = make_legend_handles(
    lesion_palette,
    "Lesion"
)
# Response
    response_handles = make_legend_handles(
    response_palette,
    "Response"
)


    legend_kwargs = dict(
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    loc="upper left"
)


    g.fig.legend(
    handles=timepoint_handles,
    title="Timepoint",
    bbox_to_anchor=(1.1, 1.0),
    bbox_transform=g.ax_heatmap.transAxes, # use 0x/oy heatmap!
    **legend_kwargs
)

    g.fig.legend(
    handles=aggressiveness_handles,
    title="Aggressiveness",
    bbox_to_anchor=(1.1, 0.8),
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

    g.fig.legend(
    handles=bone_marrow_handles,
    title="Bone marrow",
    bbox_to_anchor=(1.1, 0.6),
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

    g.fig.legend(
    handles=lesion_handles,
    title="Lesion",
    bbox_to_anchor=(1.1, 0.4),
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

    g.fig.legend(
    handles=response_handles,
    title="Response",
    bbox_to_anchor=(1.1, 0.2),
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
    )

    g.tick_params(axis='y', labelsize=labelsize) 

    g.fig.suptitle(
    f"Unsupervised clustering: Top {n} variable genes",
    fontsize=16,
    fontweight="bold",
    x=0.48,
    y=0.98
)

    heatmap_save_path = os.path.join(
    args.figures,
    f"unsupervised_clustering_{n}_genes.png"
)

    g.savefig(
    heatmap_save_path,
    bbox_inches="tight",
    dpi=300
)

    plt.close(g.fig)


unsup_cluster(n=50, labelsize=10)
unsup_cluster(n=100, labelsize=8)

# 2. Clustering specific genes

# Definitive list of MHC and antigen presentation pathway genes
mhc_i_genes = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP", "CALR", "CNX", "PDIA3"]
mhc_ii_genes = ["HLA-DRA", "HLA-DRB1", "HLA-DRB3", "HLA-DRB4", "HLA-DRB5", 
                "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1", "CD74", "HLA-DMA", "HLA-DMB"]
antigen_presentation_genes = ["CIITA", "RFX5", "RFXANK", "RFXAP", "ERAP1", "ERAP2", "PSMB8", "PSMB9", "NLEDC1"]

# Full supervised target feature set
target_genes = list(dict.fromkeys(
    mhc_i_genes
    + mhc_ii_genes
    + antigen_presentation_genes
))

available_genes = [g for g in target_genes if g in vst_df.columns]

X_supervised = vst_df[available_genes].copy()

# StandardScaler: mean 0 and diseprsion 1.
scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_supervised
)

X_scaled = pd.DataFrame(
    X_scaled,
    index=X_supervised.index,
    columns=X_supervised.columns
)

# 5. Clusterization and validation

best_n_clusters = 2
best_score = -1
best_labels = None

for n in range(2, 6):

    clusterer = AgglomerativeClustering(
        n_clusters=n,
        linkage="ward"
    )

    labels = clusterer.fit_predict(
        X_scaled
    )

    score = silhouette_score(
        X_scaled,
        labels
    )

    print(
        f"Clusters = {n} | "
        f"Silhouette = {score:.3f}"
    )

    if score > best_score:
        best_score = score
        best_n_clusters = n
        best_labels = labels

for n in range(2, 6):
    clusterer = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = clusterer.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_score = score
        best_n_clusters = n
        best_labels = labels


# Cluster labels


cluster_labels = pd.Series(
    best_labels,
    index=X_supervised.index,
    name="Cluster"
)

print("\nCluster sizes:")
print(
    cluster_labels.value_counts()
    .sort_index()
)
# Order samples by cluster


sample_order = (
    cluster_labels
    .sort_values()
    .index
)

X_supervised_ordered = X_supervised.loc[
    sample_order
]
 
# Cluster colors


cluster_palette = sns.color_palette(
    "Set2",
    best_n_clusters
)

sample_colors = (
    cluster_labels
    .loc[sample_order]
    .map(
        lambda x: cluster_palette[x]
    )
)

# Annotation dataframe
col_colors2 = pd.DataFrame({
    "Timepoint": meta.loc[sample_order, "timepoint"].map(category_palette),
    "Aggressiveness": meta.loc[sample_order, "aggressiveness"].map(aggressiv_palette),
    "Bone marrow": meta.loc[sample_order, "bone_marrow"].map(bone_marrow_palette),
    "Lesion": meta.loc[sample_order, "lesion"].map(lesion_palette),
    "Response": meta.loc[sample_order, "response"].map(response_palette),
    "Cluster": sample_colors
})


heatmap_data = (
    X_supervised_ordered
    .T
)

# Make absolutely sure that annotation order
# is identical to heatmap sample order
col_colors2 = col_colors2.loc[heatmap_data.columns]

g = sns.clustermap(
    heatmap_data,

    # expression colors
    cmap="vlag",
    center=0,

    # z-score each gene across samples
    z_score=0,

    # IMPORTANT:
    # samples are already ordered according
    # to the clusters identified above
    row_cluster=False,
    col_cluster=False,

    # cluster colors above samples
    col_colors=col_colors2,

    # visualization
    figsize=(14, 10),
    yticklabels=True,
    xticklabels=False,

    linewidths=0,

    dendrogram_ratio=(0.12, 0.03),
    colors_ratio=0.025,
    cbar_pos=(0.03, 0.82, 0.02, 0.12)
)

available_mhc_i = [
    gene for gene in mhc_i_genes
    if gene in vst_df.columns
]

available_mhc_ii = [
    gene for gene in mhc_ii_genes
    if gene in vst_df.columns
]

available_antigen = [
    gene for gene in antigen_presentation_genes
    if gene in vst_df.columns
]

available_genes = (
    available_mhc_i
    + available_mhc_ii
    + available_antigen
)


n_mhc_i = len(available_mhc_i)
n_mhc_ii = len(available_mhc_ii)
# Horizontal lines between:
# MHC-I / MHC-II / antigen presentation

g.ax_heatmap.axhline(
    n_mhc_i,
    color="black",
    linewidth=1.5
)

g.ax_heatmap.axhline(
    n_mhc_i + n_mhc_ii,
    color="black",
    linewidth=1.5
)

def make_legend_handles(palette, title):
        handles = [
        Patch(
            facecolor=color,
            edgecolor='none',
            label=str(label)
        )
        for label, color in palette.items()
    ]

        return handles


# Timepoint
timepoint_handles = make_legend_handles(
    category_palette,
    "Timepoint"
)

# Aggressiveness
aggressiveness_handles = make_legend_handles(
    aggressiv_palette,
    "Aggressiveness"
)

# Bone marrow
bone_marrow_handles = make_legend_handles(
    bone_marrow_palette,
    "Bone marrow"
)

# Lesion
lesion_handles = make_legend_handles(
    lesion_palette,
    "Lesion"
)

# Response
response_handles = make_legend_handles(
    response_palette,
    "Response"
)

legend_kwargs = dict(
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    loc="upper left"
)

g.fig.legend(
    handles=timepoint_handles,
    title="Timepoint",
    bbox_to_anchor=(1.1, 1.0),
    bbox_transform=g.ax_heatmap.transAxes, 
    **legend_kwargs
)

g.fig.legend(
    handles=aggressiveness_handles,
    title="Aggressiveness",
    bbox_to_anchor=(1.1, 0.8), 
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.fig.legend(
    handles=bone_marrow_handles,
    title="Bone marrow",
    bbox_to_anchor=(1.1, 0.6), 
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.fig.legend(
    handles=lesion_handles,
    title="Lesion",
    bbox_to_anchor=(1.1, 0.4), 
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.fig.legend(
    handles=response_handles,
    title="Response",
    bbox_to_anchor=(1.1, 0.2), 
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
    )

g.tick_params(axis='y', labelsize=10) 

g.fig.suptitle(
    "Supervised clustering based on "
    "MHC and antigen-presentation genes",
    y=1.02,
    fontsize=15,
    fontweight="bold"
)


heatmap_save_path = os.path.join(
    args.figures,
    "supervised_MHC_antigen_presentation_clustering.png"
)

g.savefig(
    heatmap_save_path,
    bbox_inches="tight",
    dpi=300
)

plt.close(g.fig)


# 3. DE analysis by timepoints (vs product) and clustering 

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
        (results["log2FoldChange"].abs() > 1.5)
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
    deg_genes = [
    gene for gene in deg_genes
    if gene in vst_df.columns
]
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
    # Annotation dataframe
    col_colors_sorted = pd.DataFrame({
    "Timepoint": meta_tp.loc[sorted_samples, "timepoint"].map(category_palette),
    "Aggressiveness": meta_tp.loc[sorted_samples, "aggressiveness"].map(aggressiv_palette),
    "Bone marrow": meta_tp.loc[sorted_samples, "bone_marrow"].map(bone_marrow_palette),
    "Lesion": meta_tp.loc[sorted_samples, "lesion"].map(lesion_palette),
    "Response": meta.loc[sorted_samples, "response"].map(response_palette),

})
    #col_colors_sorted = meta_tp.loc[sorted_samples, 'timepoint'].map(category_palette)


    # 5. Clustermap
    g_deg = sns.clustermap(
        vst_deg_sorted, 
        metric="correlation", 
        method="average", 
        cmap="vlag",
        center=0,   
        z_score=0,        
        col_colors=col_colors_sorted, 
        col_cluster=False, 
        row_cluster=True,
        dendrogram_ratio=(0.08, 0.02),
        figsize=(16, 40),
        yticklabels=False, 
        xticklabels=True,
        cbar_pos=(0.01, 0.8, 0.01, 0.15),
        colors_ratio=(0.05, 0.005) 
    )
    g_deg.ax_cbar.yaxis.set_ticks_position("left")

    legend_kwargs = dict(
    frameon=False,
    fontsize=15,
    title_fontsize=20,
    loc="upper left"
)
    # Add legend
    g_deg.fig.legend(
    handles=timepoint_handles,
    title="Timepoint",
    bbox_to_anchor=(1.1, 0.95),
    bbox_transform=g_deg.ax_heatmap.transAxes,
    **legend_kwargs
)

    g_deg.fig.legend(
    handles=aggressiveness_handles,
    title="Aggressiveness",
    bbox_to_anchor=(1.1, 0.85), 
    bbox_transform=g_deg.ax_heatmap.transAxes,
    **legend_kwargs
)

    g_deg.fig.legend(
    handles=bone_marrow_handles,
    title="Bone marrow",
    bbox_to_anchor=(1.1, 0.75), 
    bbox_transform=g_deg.ax_heatmap.transAxes,
    **legend_kwargs
)

    g_deg.fig.legend(
    handles=lesion_handles,
    title="Lesion",
    bbox_to_anchor=(1.1, 0.65), 
    bbox_transform=g_deg.ax_heatmap.transAxes,
    **legend_kwargs
)
    g_deg.fig.legend(
    handles=response_handles,
    title="Response",
    bbox_to_anchor=(1.1, 0.55), 
    bbox_transform=g_deg.ax_heatmap.transAxes,
    **legend_kwargs
    )

    g_deg.fig.suptitle(f'DEG Expression Heatmap ({len(deg_genes)} genes)', y=1.02, fontsize=25)

    # 6. Saving
    heatmap_save_path = os.path.join(args.figures, "deg_heatmap_chronological.png")
    plt.savefig(heatmap_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"DEG heatmap successfully saved to {heatmap_save_path}")


# 4. TOP DEGs FOR EACH TIMEPOINT COMPARISON

top_n = 20

for comp_id, info in comparisons.items():

    tp_name = info["tp_name"]

    deg_file = os.path.join(
        args.output_dir,
        f"DEGs_{tp_name}.tsv"
    )

    if not os.path.exists(deg_file):
        print(f"DEG file not found: {deg_file}")
        continue

    # Load DEGs
    deg = pd.read_csv(
        deg_file,
        sep="\t",
        index_col=0
    )

    # Make sure numeric
    deg["padj"] = pd.to_numeric(
        deg["padj"],
        errors="coerce"
    )

    deg["log2FoldChange"] = pd.to_numeric(
        deg["log2FoldChange"],
        errors="coerce"
    )

    deg = deg.dropna(
        subset=["padj", "log2FoldChange"]
    )

  
    # Select top 20 genes
    # Here genes are ranked by absolute log2FC
  
    top_genes = (
        deg
        .sort_values(
            "log2FoldChange",
            key=lambda x: x.abs(),
            ascending=False
        )
        .head(top_n)
        .index
        .tolist()
    )

    # Keep only genes present in VST
    top_genes = [
        gene
        for gene in top_genes
        if gene in vst_df.columns
    ]

    print(
        f"{tp_name}: selected {len(top_genes)} top DEGs"
    )

    if len(top_genes) == 0:
        print(
            f"No matching genes in VST for {tp_name}"
        )
        continue

    
    # Select samples for this comparison
 

    if tp_name == "d7_after_vs_product":

        selected_timepoints = [
            "product",
            "d7 after"
        ]

    elif tp_name == "d28_after_vs_product":

        selected_timepoints = [
            "product",
            "d28 after"
        ]

    else:
        continue

    selected_samples = meta[
        meta["timepoint"].isin(selected_timepoints)
    ].index

    selected_samples = [
        sample
        for sample in selected_samples
        if sample in vst_df.index
    ]

    meta_heatmap = meta.loc[
        selected_samples
    ].copy()

    # Sort samples chronologically
    meta_heatmap["timepoint"] = pd.Categorical(
        meta_heatmap["timepoint"],
        categories=selected_timepoints,
        ordered=True
    )

    sorted_samples = (
        meta_heatmap
        .sort_values("timepoint")
        .index
    )

 
    # Prepare heatmap matrix


    heatmap_data = (
        vst_df
        .loc[
            sorted_samples,
            top_genes
        ]
        .T
    )

    # Add arrows according to direction of change
    gene_labels = []

    for gene in top_genes:

        log2fc = deg.loc[
            gene,
            "log2FoldChange"
        ]

        if log2fc > 0:
            gene_labels.append(
                f"{gene} ↑"
            )
        else:
            gene_labels.append(
                f"{gene} ↓"
            )

    heatmap_data.index = gene_labels

  
    # Column colors
    

    col_colors = (
        meta_heatmap
        .loc[
            sorted_samples,
            "timepoint"
        ]
        .map(category_palette)
    )

    # Plot

    # Clean heatmap matrix before clustering

    # Convert everything to numeric
    heatmap_data = heatmap_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace infinite values
    heatmap_data = heatmap_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        f"{tp_name}: matrix before cleaning = "
        f"{heatmap_data.shape}"
    )

    print(
        f"{tp_name}: NaN before cleaning = "
        f"{heatmap_data.isna().sum().sum()}"
    )

    # Remove genes containing NaN
    heatmap_data = heatmap_data.dropna(
        axis=0,
        how="any"
    )

    # Remove genes with zero variance
    row_variance = heatmap_data.var(
        axis=1,
        ddof=1
    )

    heatmap_data = heatmap_data.loc[
        row_variance > 1e-10
    ]

    print(
        f"{tp_name}: matrix after cleaning = "
        f"{heatmap_data.shape}"
    )

    if heatmap_data.shape[0] < 2:
        print(
            f"{tp_name}: fewer than 2 valid genes "
            f"remain after cleaning. Skipping heatmap."
        )
        continue

   
    # Explicit z-score by gene
    

    heatmap_z = heatmap_data.sub(
        heatmap_data.mean(axis=1),
        axis=0
    ).div(
        heatmap_data.std(axis=1, ddof=1),
        axis=0
    )

    # Final safety check
    if not np.isfinite(
        heatmap_z.to_numpy()
    ).all():

        print(
            f"{tp_name}: non-finite values detected "
            f"after z-score. Skipping heatmap."
        )
        continue

    # Plot

    g_top = sns.clustermap(
        heatmap_z,
        metric="correlation",
        method="average",
        cmap="vlag",
        z_score=None,
        col_colors=col_colors.loc[
            heatmap_z.columns
        ],
        col_cluster=False,
        row_cluster=True,
        figsize=(14, 9),
        yticklabels=True,
        xticklabels=True
    )

    # Legend
    legend_elements = [
        Patch(
            facecolor=category_palette[tp],
            label=tp
        )
        for tp in selected_timepoints
    ]

    g_top.ax_heatmap.legend(
        handles=legend_elements,
        title="Timepoint",
        bbox_to_anchor=(1.25, 1),
        loc="upper left"
    )

    g_top.fig.suptitle(
        f"Top 20 DEGs: {tp_name.replace('_', ' ')}",
        y=1.02,
        fontsize=14
    )


    # Save
    

    heatmap_save_path = os.path.join(
        args.figures,
        f"DEG_heatmap_{tp_name}_top20.png"
    )

    g_top.savefig(
        heatmap_save_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close(g_top.fig)

    print(
        f"Top 20 DEG heatmap saved to: "
        f"{heatmap_save_path}"
    )




# 5. GSEA FOR TIMEPOINT COMPARISONS


gsea_timepoints = [
    "d7_after_vs_product",
    "d28_after_vs_product"
]

gsea_root = os.path.join(
    args.output_dir,
    "timepoint_gsea"
)

os.makedirs(
    gsea_root,
    exist_ok=True
)


for tp_name in gsea_timepoints:

    # --------------------------------------------------------
    # Load DESeq2 results
    # --------------------------------------------------------

    input_file = os.path.join(
        args.output_dir,
        f"DESeq2_{tp_name}.tsv"
    )

    if not os.path.exists(input_file):

        print(
            f"DESeq2 file not found: {input_file}"
        )

        continue


    df = pd.read_csv(
        input_file,
        sep="\t",
        index_col=0
    )


    # --------------------------------------------------------
    # Prepare ranking
    # --------------------------------------------------------

    df.index.name = "gene"

    df = df.reset_index()

    df = df[
        [
            "gene",
            "stat"
        ]
    ].dropna()


    # Remove duplicated genes
    df = df.drop_duplicates(
        subset="gene",
        keep="first"
    )


    # Remove empty gene names
    df = df[
        df["gene"].astype(str).str.strip() != ""
    ]


    # Numeric statistic
    df["stat"] = pd.to_numeric(
        df["stat"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["stat"]
    )


    # Sort genes by DESeq2 statistic
    ranked = (
        df[
            [
                "gene",
                "stat"
            ]
        ]
        .sort_values(
            "stat",
            ascending=False
        )
    )


    print(
        f"{tp_name}: "
        f"{len(ranked)} genes available for GSEA"
    )


    if len(ranked) < 20:

        print(
            f"Not enough genes for GSEA: {tp_name}"
        )

        continue



    # Output directory for this comparison


    outdir = os.path.join(
        gsea_root,
        tp_name
    )

    os.makedirs(
        outdir,
        exist_ok=True
    )


   
    # Run GSEA
   

    try:

        pr = gp.prerank(
            rnk=ranked,
            gene_sets="GO_Biological_Process_2021",
            outdir=outdir,
            permutation_num=1000,
            min_size=15,
            max_size=500,
            seed=42,
            verbose=True
        )

    except Exception as e:

        print(
            f"GSEA failed for {tp_name}: {e}"
        )

        continue



    # Save all GSEA results
   

    all_results = pr.res2d.copy()

    all_results_file = os.path.join(
        outdir,
        f"all_results_GO_Biological_Process_2021_{tp_name}.tsv"
    )

    all_results.to_csv(
        all_results_file,
        sep="\t",
        index=False
    )

    print(
        f"All GSEA results saved: "
        f"{all_results_file}"
    )


    # Check required columns
    

    required_columns = [
        "Term",
        "NES",
        "FDR q-val"
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in all_results.columns
    ]

    if missing_columns:

        print(
            f"Missing columns for {tp_name}: "
            f"{missing_columns}"
        )

        print(
            all_results.columns.tolist()
        )

        continue


    
    # Prepare results for plotting
 

    plot_df = all_results[
        [
            "Term",
            "NES",
            "FDR q-val"
        ]
    ].copy()


    plot_df["NES"] = pd.to_numeric(
        plot_df["NES"],
        errors="coerce"
    )

    plot_df["FDR q-val"] = pd.to_numeric(
        plot_df["FDR q-val"],
        errors="coerce"
    )


    plot_df = plot_df.dropna(
        subset=[
            "NES",
            "FDR q-val"
        ]
    )



    
    # Significant pathways

    significant = plot_df[
        plot_df["FDR q-val"] < 0.05
    ].copy()


    significant_file = os.path.join(
        outdir,
        f"significant_results_GO_Biological_Process_2021_{tp_name}.tsv"
    )

    significant.to_csv(
        significant_file,
        sep="\t",
        index=False
    )


    print(
        f"{tp_name}: "
        f"{len(significant)} significant pathways"
    )


    if significant.empty:

        print(
            f"No significant pathways for {tp_name}"
        )

        continue

    # POSITIVE NES

    positive = (
        significant[
            significant["NES"] > 0
        ]
        .sort_values(
            "FDR q-val"
        )
        .head(10)
        .copy()
    )


    if not positive.empty:

        positive["minus_log10_FDR"] = (
            -np.log10(
                positive["FDR q-val"].clip(
                    lower=1e-300
                )
            )
        )


        plt.figure(
            figsize=(10, 7)
        )


        sns.scatterplot(
            data=positive,
            x="NES",
            y="Term",
            size="minus_log10_FDR",
            hue="NES",
            palette="Reds",
            sizes=(50, 350),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6
        )


        plt.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.8
        )


        plt.title(
            f"GSEA positive NES — {tp_name}"
        )


        plt.xlabel(
            "Normalized Enrichment Score (NES)"
        )

        plt.ylabel("")


        plt.legend(
            bbox_to_anchor=(
                1.05,
                1
            ),
            loc="upper left"
        )


        plt.tight_layout()


        positive_file = os.path.join(
            outdir,
            f"GSEA_positive_NES_{tp_name}.png"
        )


        plt.savefig(
            positive_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


        print(
            f"Positive GSEA plot saved: "
            f"{positive_file}"
        )

    # NEGATIVE NES

    negative = (
        significant[
            significant["NES"] < 0
        ]
        .sort_values(
            "FDR q-val"
        )
        .head(10)
        .copy()
    )


    if not negative.empty:

        negative["minus_log10_FDR"] = (
            -np.log10(
                negative["FDR q-val"].clip(
                    lower=1e-300
                )
            )
        )


        plt.figure(
            figsize=(10, 7)
        )


        sns.scatterplot(
            data=negative,
            x="NES",
            y="Term",
            size="minus_log10_FDR",
            hue="NES",
            palette="Blues_r",
            sizes=(50, 350),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6
        )


        plt.axvline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.8
        )


        plt.title(
            f"GSEA negative NES — {tp_name}"
        )


        plt.xlabel(
            "Normalized Enrichment Score (NES)"
        )

        plt.ylabel("")


        plt.legend(
            bbox_to_anchor=(
                1.05,
                1
            ),
            loc="upper left"
        )


        plt.tight_layout()


        negative_file = os.path.join(
            outdir,
            f"GSEA_negative_NES_{tp_name}.png"
        )


        plt.savefig(
            negative_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

