import os
import argparse
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize


parser = argparse.ArgumentParser(description="ranked GSEA, GO/KEGG")

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


        # Prepare results for visualization

        required_columns = [
            "Term",
            "NES",
            "FDR q-val"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in all_results.columns
        ]

        if missing_columns:

            print(
                f"  -> Missing columns: "
                f"{missing_columns}"
            )

            print(
                "  -> Available columns:"
            )

            print(
                all_results.columns.tolist()
            )

            continue

        df = all_results[
            [
                "Term",
                "NES",
                "FDR q-val"
            ]
        ].copy()

        # Приводим числовые колонки к numeric

        df["NES"] = pd.to_numeric(
            df["NES"],
            errors="coerce"
        )

        df["FDR q-val"] = pd.to_numeric(
            df["FDR q-val"],
            errors="coerce"
        )

        df = df.dropna(
                subset=[
                    "NES",
                    "FDR q-val"
                ]
            )


# --------------------------------------------------------
# Select positive and negative pathways
# --------------------------------------------------------

df["NES"] = pd.to_numeric(
    df["NES"],
    errors="coerce"
)

df["FDR q-val"] = pd.to_numeric(
    df["FDR q-val"],
    errors="coerce"
)

df = df.dropna(
    subset=["NES", "FDR q-val"]
)


# ========================================================
# POSITIVE NES
# ========================================================

df_positive = (
    df[df["NES"] > 0]
    .sort_values("FDR q-val")
    .head(10)
    .copy()
)


if not df_positive.empty:

    df_positive["minus_log10_FDR"] = (
        -np.log10(
            df_positive["FDR q-val"] + 1e-300
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.scatterplot(
        data=df_positive,
        x="NES",
        y="Term",
        size="minus_log10_FDR",
        hue="NES",
        palette="RdBu_r",
        sizes=(40, 350),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    plt.title(
        f"Ranked GSEA — positive NES — {short_name}"
    )

    plt.xlabel(
        "Normalized Enrichment Score"
    )

    plt.ylabel("")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()

    output_file_positive = os.path.join(
        outdir,
        f"GSEA_dotplot_{short_name}_positive.png"
    )

    plt.savefig(
        output_file_positive,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Positive GSEA dotplot saved:",
        output_file_positive
    )


# ========================================================
# NEGATIVE NES
# ========================================================

df_negative = (
    df[df["NES"] < 0]
    .sort_values("FDR q-val")
    .head(10)
    .copy()
)


if not df_negative.empty:

    df_negative["minus_log10_FDR"] = (
        -np.log10(
            df_negative["FDR q-val"] + 1e-300
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.scatterplot(
        data=df_negative,
        x="NES",
        y="Term",
        size="minus_log10_FDR",
        hue="NES",
        palette="RdBu_r",
        sizes=(40, 350),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    plt.title(
        f"Ranked GSEA — negative NES — {short_name}"
    )

    plt.xlabel(
        "Normalized Enrichment Score"
    )

    plt.ylabel("")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()

    output_file_negative = os.path.join(
        outdir,
        f"GSEA_dotplot_{short_name}_negative.png"
    )

    plt.savefig(
        output_file_negative,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Negative GSEA dotplot saved:",
        output_file_negative
    )





