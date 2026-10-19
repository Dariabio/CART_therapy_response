import os
import numpy as np
import argparse
import pandas as pd
import gseapy as gp

parser = argparse.ArgumentParser()

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
    "--panel_dir",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

args = parser.parse_args()


comparison = args.comparison

os.makedirs(
    args.output_dir,
    exist_ok=True
)

#Paths
GSEA_DIR = args.output_dir

DE_DIR = args.de_dir

# 1. Build gene panels
panels = {}
# Panel 1
gp1 = pd.read_csv(
    os.path.join(
    args.panel_dir,
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
    args.panel_dir,
    "gene_panel_5.csv"
)
    )

for col in gp2.columns:

    genes = (
        gp2[col]
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
        f"P2_{col}"
    ] = genes


# Panel 3

gp3 = pd.read_csv(
    os.path.join(
    args.panel_dir,
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
    args.panel_dir,
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
