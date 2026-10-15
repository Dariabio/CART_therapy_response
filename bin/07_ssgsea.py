import os
import argparse
import pandas as pd
import gseapy as gp

parser = argparse.ArgumentParser()

parser.add_argument(
    "--expression",
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


os.makedirs(
    args.output_dir,
    exist_ok=True
)


# Load files
expression = pd.read_csv(args.expression, sep="\t", index_col=0)

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

'''
# Panel 2

gp2 = pd.read_csv(
    os.path.join(
    args.panel_dir,
    "gene_panel_2.csv"
)
    )

gp2["subpop"] = (
    gp2["Stage"].astype(str)
    + "_"
    + gp2["Annotation"].astype(str)
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
'''


# Panel 5
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

# Panel_2 preprossesing




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


# 3. Run ssGSEA
ss = gp.ssgsea(
    data=expression,
    gene_sets=panels,
    outdir= args.output_dir ,
    permutation_num=0, # for ssGSEA
    min_size=3,
    max_size=500,
    seed=42
)

# saved results to read

scores_raw = ss.res2d

scores_all = scores_raw.pivot_table(
    index="Name",
    columns="Term",
    values="NES"
)

scores_all.to_csv(os.path.join(args.output_dir, "scores_all.csv"))

print("ssGSEA completed")
