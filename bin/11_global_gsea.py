import os
import argparse
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description="ranked GSEA, GO/KEGG")

#Arguments
parser.add_argument(
    "--comparison",
    required=True
)

parser.add_argument(
    "--de_dir",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

args = parser.parse_args()


comparison = args.comparison


#Paths
GSEA_DIR = args.output_dir
DE_DIR = args.de_dir

os.makedirs(
    GSEA_DIR,
    exist_ok=True
)


# Run GSEA
# per each timepoint


timepoints = [
    "product",
    "d7",
    "d28"
]


# COLLECT GSEA RESULTS
results = {}



for short_name in timepoints:

    input_file = os.path.join(
        DE_DIR,
        f"DESeq2_{short_name}.tsv"
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
        GSEA_DIR,
        short_name
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

        all_results = pr.res2d.copy()

        all_results.to_csv(
            os.path.join(
                outdir,
                f"all_results_{db}_{short_name}.tsv"
            ),
            sep="\t",
            index=False
        )
            
        # Filter significant (FDR < 0.05)
        significant_terms = all_results[all_results['FDR q-val'] < 0.05]
        significant_terms.to_csv(
        os.path.join(outdir, f"significant_results_{db}_{short_name}.tsv"), 
        sep="\t", index=False)

        df2 = all_results[["Term", "NES", "FDR q-val"]].copy()

        df2 = df2.dropna(subset=["NES"])

        results[short_name] = df2

# BUILD COMMON HEATMAP

# NES matrix
nes_list = []

fdr_list = []


for timepoint in timepoints:

    if timepoint not in results:

        continue


    res = results[
        timepoint
    ].copy()


    res = res.set_index(
        "Term"
    )


    nes = pd.to_numeric(
        res["NES"],
        errors="coerce"
    )

    fdr = pd.to_numeric(
        res["FDR q-val"],
        errors="coerce"
    )


    nes_list.append(
        nes.rename(
            timepoint
        )
    )

    fdr_list.append(
        fdr.rename(
            timepoint
        )
    )

# Combine all pathways

nes_mat = pd.concat(
    nes_list,
    axis=1
)

fdr_mat = pd.concat(
    fdr_list,
    axis=1
)


# Make sure columns are in desired order
nes_mat = nes_mat.reindex(
    columns=timepoints
)

fdr_mat = fdr_mat.reindex(
    columns=timepoints
)


# Keep pathways significant in at least one timepoint
significant_any = (
    fdr_mat < 0.05
).any(axis=1)

nes_mat = nes_mat.loc[
    significant_any
]

fdr_mat = fdr_mat.loc[
    significant_any
]


# ORDER PATHWAYS

# Mean absolute NES across available timepoints.
# This is ONLY for visualization ordering.

nes_mat["mean_abs_NES"] = (
    nes_mat
    .abs()
    .mean(
        axis=1,
        skipna=True
    )
)

nes_mat = nes_mat.sort_values(
    "mean_abs_NES",
    ascending=False
)

nes_mat = nes_mat.drop(
    columns="mean_abs_NES"
)


fdr_mat = fdr_mat.reindex(
    nes_mat.index
)


# SIGNIFICANCE MARKERS


def fdr_to_stars(x):

    if pd.isna(x):
        return ""

    if x < 0.0001:
        return "***"

    elif x < 0.01:
        return "**"

    elif x < 0.05:
        return "*"

    return ""


significance = fdr_mat.apply(
    lambda col: col.map(fdr_to_stars)
)


g = sns.clustermap(
    nes_mat,
    cmap="RdBu_r",
    center=0,

    figsize=(
        10,
        max(
            8,
            len(nes_mat) * 0.30
        )
    ),

    col_cluster=False,
    row_cluster=False,

    linewidths=0.5,
    linecolor="white",

    annot=significance,
    fmt="",

    annot_kws={
        "fontsize": 9,
        "fontweight": "bold"
    },
    # пока оставляем colorbar
    cbar_pos=(0.03, 0.3, 0.01, 0.15)
)



g.ax_cbar.yaxis.set_ticks_position("left")

# Labels


g.ax_heatmap.set_xlabel(
        "Timepoint"
    )

g.ax_heatmap.set_ylabel(
        "Gene signature"
    )

g.ax_heatmap.set_title(
        f"Ranked GSEA — "
        f"{comparison} — "
        f"NES across timepoints - GO Biological pathways",
        pad=20
    )



output_file = os.path.join(
    GSEA_DIR,
    "Ranked_GO_GSEA_NES_heatmap.png"
)


g.savefig(
    output_file,
    dpi=300
)


plt.close(
    g.fig
)