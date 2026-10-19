import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#  Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--comparison",
    required=True
)

parser.add_argument(
    "--de_dir",
    required=True
)

parser.add_argument(
    "--figures",
    required=True
)

args = parser.parse_args()

comparison = args.comparison

os.makedirs(
    args.figures,
    exist_ok=True
)


# Volcano plot function


def volcano_plot(
    de_results,
    comparison,
    comparison_label,
    timepoint,
    outdir,
    padj_cutoff=0.05,
    logfc_cutoff=1,
    top_n_labels=10
):

    df = de_results.copy()

    
    # Convert columns to numeric
    

    df["padj"] = pd.to_numeric(
        df["padj"],
        errors="coerce"
    )

    df["log2FoldChange"] = pd.to_numeric(
        df["log2FoldChange"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "padj",
            "log2FoldChange"
        ]
    )

    
    # -log10 adjusted p-value
    

    df["minus_log10_padj"] = (
        -np.log10(df["padj"] + 1e-300)
    )

   
    # Significance
    

    df["significance"] = "NS"

    df.loc[
        (df["padj"] < padj_cutoff) &
        (df["log2FoldChange"] > logfc_cutoff),
        "significance"
    ] = "Up"

    df.loc[
        (df["padj"] < padj_cutoff) &
        (df["log2FoldChange"] < -logfc_cutoff),
        "significance"
    ] = "Down"

  
    # Plot
    

    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        data=df,
        x="log2FoldChange",
        y="minus_log10_padj",
        hue="significance",
        palette={
            "Up": "red",
            "Down": "blue",
            "NS": "grey"
        },
        alpha=0.25,
        s=20,
        linewidth=0
    )

    
    # Labels for top genes
    

    top_genes = (
        df[
            df["significance"].isin(
                ["Up", "Down"]
            )
        ]
        .sort_values("padj")
        .head(top_n_labels)
    )

    x_offset = 0.08
    y_offset = 0.05

    label_bbox = dict(
        boxstyle="round,pad=0.2",
        fc="white",
        ec="none",
        alpha=0.55
    )

    for idx, row in top_genes.iterrows():

        gene_name = str(idx)

        x_pos = row["log2FoldChange"]
        y_pos = row["minus_log10_padj"]

        if x_pos > 0:

            plt.text(
                x_pos + x_offset,
                y_pos + y_offset,
                gene_name,
                fontsize=8,
                fontweight="bold",
                color="black",
                ha="left",
                va="center",
                bbox=label_bbox
            )

        else:

            plt.text(
                x_pos - x_offset,
                y_pos + y_offset,
                gene_name,
                fontsize=8,
                fontweight="bold",
                color="black",
                ha="right",
                va="center",
                bbox=label_bbox
            )

   
    # Threshold lines
    
    plt.axvline(
        logfc_cutoff,
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.axvline(
        -logfc_cutoff,
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.axhline(
        -np.log10(padj_cutoff),
        color="black",
        linestyle="--",
        linewidth=1
    )

    # Labels
    

    plt.xlabel(
        f"log2 Fold Change\n{comparison_label}"
    )

    plt.ylabel(
        "-log10 adjusted p-value"
    )

    plt.title(
        f"DESeq2 volcano plot: {comparison_label}\n{timepoint}"
    )

    plt.legend(title="")

    sns.despine()

    plt.tight_layout()

    
    # Save
    

    output_file = os.path.join(
        args.figures,
        f"Volcano_DESeq2_{comparison}_{timepoint}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()




# 5. Run for all timepoints


TIMEPOINTS = {
    "product": "DESeq2_product.tsv",
    "d7": "DESeq2_d7.tsv",
    "d28": "DESeq2_d28.tsv"
}


for timepoint, filename in TIMEPOINTS.items():

    results_file = os.path.join(
        args.de_dir,
        filename
    )

    if not os.path.exists(results_file):

        print(
            f"WARNING: file not found: {results_file}"
        )

        continue

    # save
    results = pd.read_csv(
        results_file,
        sep="\t",
        index_col=0
    )

    volcano_plot(
        de_results=results,
        comparison=comparison,
        comparison_label=comparison,
        timepoint=timepoint,
        outdir=args.figures
    )
