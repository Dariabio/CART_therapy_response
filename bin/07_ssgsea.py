import os
import pandas as pd
import gseapy as gp

# Paths

project_dir = "/Users/darakuzmina/Desktop/CAR_T"
results_dir = os.path.join(project_dir, "results") 
dir_raw = os.path.join(project_dir, "data", "raw")
dir_proc = os.path.join(project_dir, "data", "processed")

OUT_DIR = os.path.join(results_dir,"ssgsea")
os.makedirs(OUT_DIR, exist_ok=True)

# Load files
expression = pd.read_csv(os.path.join(dir_proc, "expr_all_vst.tsv"), sep="\t", index_col=0)

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

# 3. Run ssGSEA
ss = gp.ssgsea(
    data=expression,
    gene_sets=panels,
    outdir= OUT_DIR ,
    permutation_num=0,# for ssGSEA
    min_size=3,
    max_size=500,
    seed=42
)

# saved results to read

scores_raw = pd.read_csv(os.path.join(OUT_DIR, "gseapy.gene_set.ssgsea.report.csv"), sep=",")

# scores_wide: samples × subpopulations
scores_all = scores_raw.pivot_table(
    index="Name",
    columns="Term",
    values="NES"
)

scores_all.to_csv(os.path.join(OUT_DIR, "scores_all.csv"))

print("ssGSEA completed")

