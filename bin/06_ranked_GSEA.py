import os
import numpy as np
import argparse
import pandas as pd
import gseapy as gp

# Paths
project_dir = "/Users/darakuzmina/Desktop/CAR_T"
results_dir = os.path.join(project_dir, "results") 
dir_raw = os.path.join(project_dir, "data", "raw")
dir_proc = os.path.join(project_dir, "data", "processed")

#

parser = argparse.ArgumentParser()

parser.add_argument(
    "--comparison",
    required=True
)

args = parser.parse_args()

comparison = args.comparison

# 1. Build gene panels
panels = {}
# Panel 1
gp1 = pd.read_csv(
    os.path.join(
        dir_raw,
        "gene_panel_1.csv"
    )
)
for col in gp1.columns:

    genes = (
        gp1[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    genes = [
        g for g in genes
        if g != "nan"
    ]

    panels[
        f"P1_{col}"
    ] = genes


# Panel 2

gp2 = pd.read_csv(
    os.path.join(
       dir_raw,
        "gene_panel_2.csv"
    )
)

gp2["subpop"] = (
    gp2["Annotation"].astype(str)
    + "_"
    + gp2["Gene_Category"].astype(str)
)

for name, group in gp2.groupby(
    "subpop"
):

    genes = (
        group["Gene"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    panels[
        f"P2_{name}"
    ] = genes


# Panel 3

gp3 = pd.read_csv(
    os.path.join(
        dir_raw,
        "gene_panel_3.csv"
    )
)

for col in gp3.columns:

    genes = (
        gp3[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    panels[
        f"P3_{col}"
    ] = genes


# Panel 4

gp4 = pd.read_csv(
    os.path.join(
        dir_raw,
        "gene_panel_4.csv"
    )
)

for col in gp4.columns:

    genes = (
        gp4[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    panels[
        f"P4_{col}"
    ] = genes

# 2. Run GSEA

#Paths
GSEA_DIR = os.path.join(
    results_dir,
    comparison,
    "ranked_gsea"
)

os.makedirs(
    GSEA_DIR,
    exist_ok=True
)

DE_DIR = os.path.join(
    results_dir,
    comparison,
    "de_analysis"
)

# per each timepoint
for short_name in [
    "product",
    "d7",
    "d28"
]:

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


    gp.prerank(
        rnk=ranked,
        gene_sets=panels,
        outdir=outdir,
        permutation_num=1000,
        min_size=3,
        max_size=500,
        seed=42
    )


print(
    f"Panel GSEA completed: {comparison}"
)


