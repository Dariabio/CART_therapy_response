import os
import pandas as pd
import gseapy as gp
from scipy.stats import zscore
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

project_dir = "/Users/darakuzmina/Desktop/CAR_T"
output_dir = os.path.join(project_dir, "data", "raw")


signatures = {
    "CD8_T": [
        "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2",
        "CD8A", "CD8B", "LCK", "MAL", "LTB"
    ],

    "CD4_T": [
        "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2",
        "CD4", "IL7R", "LTB", "MAL", "CCR7"
    ],

    "Regulatory_T_cell": [
        "CD3D", "CD3E", "TRBC1", "TRBC2",
        "FOXP3", "IL2RA", "CTLA4", "TIGIT",
        "IKZF2", "BATF"
    ],

    "NK_cell": [
        "NKG7", "GNLY", "KLRD1", "PRF1",
        "GZMB", "GZMH", "FCGR3A", "TRAC",
        "TYROBP", "XCL1", "XCL2"
    ],

    "B_cell": [
        "CD79A", "CD79B", "MS4A1", "CD37",
        "CD74", "HLA-DRA", "HLA-DPA1",
        "CD22", "BANK1", "CD19"
    ],

    "Plasma_cell": [
        "MZB1", "JCHAIN", "XBP1", "SDC1",
        "CD38", "IGHG1", "IGHG3", "IGKC",
        "DERL3", "SEC11C"
    ],

    "Monocyte": [
        "LYZ", "FCN1", "S100A8", "S100A9",
        "CTSS", "LGALS3", "CTSD", "FCER1G",
        "TYROBP", "LILRB1", "CTSC"
    ],

    "Macrophage": [
        "LYZ", "LST1", "TYROBP", "FCER1G",
        "LILRB1", "CTSD", "CTSS",
        "C1QA", "C1QB", "C1QC", "MSR1"
    ],

    "Dendritic_cell": [
        "FCER1A", "CD1C", "CLEC10A",
        "CST3", "HLA-DRA", "HLA-DPA1",
        "HLA-DPB1", "HLA-DQB1", "CD74",
        "CLEC12A"
    ],

    "Neutrophil": [
        "FCGR3B", "CSF3R", "CEACAM8",
        "S100A8", "S100A9", "FPR1",
        "NAMPT", "MNDA", "S100P", "FCGR3A"
    ]
}


# максимальное количество генов среди всех signatures
max_genes = max(
    len(genes)
    for genes in signatures.values()
)


# создаём DataFrame
df = pd.DataFrame({
    cell_type: genes + [None] * (max_genes - len(genes))
    for cell_type, genes in signatures.items()
})


# сохраняем
df.to_csv(os.path.join(output_dir,
    "PBMC_immune_signatures.csv"),
    index=False
)


expression = pd.read_csv(os.path.join(project_dir,
    "results", "03_vst_pca", "expr_all_vst.tsv"),
    sep="\t",
    index_col=0
)

meta = pd.read_csv(os.path.join(project_dir,
    "results", "02_prepare_metadata", "meta_analysis.tsv"),
    sep="\t",
    index_col=0
)

gene_sets = {}

for cell_type in df.columns:

    genes = (
       df[cell_type]
        .dropna()
        .astype(str)
        .tolist()
    )

    # оставить только гены,
    # которые есть в expression
    genes = [
        gene
        for gene in genes
        if gene in expression.index
    ]

    gene_sets[cell_type] = genes

    print(
        cell_type,
        len(genes),
        "genes found"
    )


ss = gp.ssgsea(
    data=expression,
    gene_sets=gene_sets,
    outdir= output_dir,
    permutation_num=0,
    min_size=3,
    max_size=500,
    seed=42
)


scores_raw = ss.res2d

scores_all = scores_raw.pivot_table(
    index="Name",
    columns="Term",
    values="NES"
)

# гарантируем числовой тип
scores_all = scores_all.apply(
    pd.to_numeric,
    errors="coerce"
)

# Z-score по каждому пациенту
heatmap = scores_all.apply(
    lambda x: zscore(x, nan_policy="omit"),
    axis=1
)

heatmap_data = pd.DataFrame(
    heatmap.tolist(),
    index=scores_all.index,
    columns=scores_all.columns
)

heatmap_data = heatmap_data.T


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

aggressiv_palette = {'aggressive': '#B6A1C7', 
                     'indolent': '#666262'}

bone_marrow_palette = {'bone_marrow_positive': '#FFC0CB', 
               'bone_marrow_negative': "#B6D69A"}

lesion_palette = {'widespread': '#FF692A', 
                  'localized': "#5F3737"}

col_colors = pd.DataFrame({
        'Timepoint': meta['timepoint'].map(category_palette),
        'Aggressiveness': meta['aggressiveness'].map(aggressiv_palette),
        'Bone marrow': meta['bone_marrow'].map(bone_marrow_palette),
        'Lesion': meta['lesion'].map(lesion_palette)
    })
col_colors = col_colors.loc[heatmap_data.columns]

# CLUSTERMAP

g = sns.clustermap(
    heatmap_data,
    cmap="vlag",
    center=0,
    col_colors=col_colors,
    figsize=(14, 12),
    xticklabels=True,
    yticklabels=True
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

legend_kwargs = dict(
    frameon=False,
    fontsize=11,
    title_fontsize=12,
    loc="upper left"
)


g.fig.legend(
    handles=timepoint_handles,
    title="Timepoint",
    bbox_to_anchor=(1.3, 1.0),
    bbox_transform=g.ax_heatmap.transAxes, # Используем оси heatmap!
    **legend_kwargs
)

g.fig.legend(
    handles=aggressiveness_handles,
    title="Aggressiveness",
    bbox_to_anchor=(1.3, 0.8), # Смещаем ниже
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.fig.legend(
    handles=bone_marrow_handles,
    title="Bone marrow",
    bbox_to_anchor=(1.3, 0.6), # Смещаем ниже
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.fig.legend(
    handles=lesion_handles,
    title="Lesion",
    bbox_to_anchor=(1.3, 0.4), # Смещаем ниже
    bbox_transform=g.ax_heatmap.transAxes,
    **legend_kwargs
)

g.tick_params(axis='y', labelsize=10) 

g.ax_heatmap.set_xlabel("Patient")
g.ax_heatmap.set_ylabel("Cell signature")

g.fig.suptitle(
    "Immune cell signatures — ssGSEA activity",
    y=1.02
)

g.savefig(
    os.path.join(
        output_dir,
        "GSEA_immune_signatures_clustermap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()