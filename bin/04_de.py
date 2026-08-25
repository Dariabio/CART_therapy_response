import os
import pandas as pd
import argparse

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


# ============================================================
# PATHS
# ============================================================

project_dir = "/Users/darakuzmina/Desktop/CAR_T"

dir_proc = os.path.join(
    project_dir,
    "data",
    "processed"
)


# ============================================================
# LOAD DATA
# ============================================================

counts = pd.read_csv(
    os.path.join(
        dir_proc,
        "counts.tsv"
    ),
    sep="\t",
    index_col=0
)

meta = pd.read_csv(
    os.path.join(
        dir_proc,
        "meta_analysis.tsv"
    ),
    sep="\t",
    index_col=0
)


print("Counts shape:", counts.shape)
print("Metadata shape:", meta.shape)


# ============================================================
# COMPARISON
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--comparison",
    required=True,
    choices=[
        "group_analysis_1",
        "response",
        "aggressiveness",
        "bone_marrow",
        "lesion"
    ]
)

args = parser.parse_args()

comparison = args.comparison

# Configurations
COMPARISONS = {

    "group_analysis_1": {
            "column": "group_analysis_1",
            "positive": "responder",
            "negative": "nonresponder"
        },

    "response": {
        "column": "response",
        "positive": "responder",
        "negative": "nonresponder"
    },

    "aggressiveness": {
        "column": "aggressiveness",
        "positive": "aggressive",
        "negative": "indolent"
    },

    "bone_marrow": {
        "column": "bone_marrow",
        "positive": "bone_marrow_positive",
        "negative": "bone_marrow_negative"
    },

    "lesion": {
        "column": "lesion",
        "positive": "widespread",
        "negative": "localized"
    }
}


config = COMPARISONS[comparison]

comparison_column = config["column"]
positive = config["positive"]
negative = config["negative"]


print("\n========================================")
print("COMPARISON")
print("========================================")

print("Comparison:", comparison)
print("Column:", comparison_column)
print("Positive:", positive)
print("Negative:", negative)


# ============================================================
# CHECK METADATA
# ============================================================

if comparison_column not in meta.columns:

    raise ValueError(
        f"Column '{comparison_column}' "
        f"not found in metadata."
    )


print("\nAvailable groups:")

print(
    meta[comparison_column]
    .value_counts(dropna=False)
)


# ============================================================
# OUTPUT
# ============================================================

results_dir = os.path.join(
    project_dir,
    "results",
    comparison,
    "de_analysis"
)

os.makedirs(
    results_dir,
    exist_ok=True
)


# ============================================================
# TIMEPOINTS
# ============================================================

TIMEPOINTS = {
    "product": "product",
    "d7": "d7 after",
    "d28": "d28 after"
}


# ============================================================
# DESEQ2
# ============================================================

for tp_name, tp_value in TIMEPOINTS.items():

    print("\n")
    print("========================================")
    print("TIMEPOINT:", tp_name)
    print("========================================")


    # --------------------------------------------------------
    # Select metadata
    # --------------------------------------------------------

    meta_tp = meta[
        (meta["timepoint"] == tp_value) &
        (
            meta[comparison_column].isin(
                [positive, negative]
            )
        )
    ].copy()


    # --------------------------------------------------------
    # Match samples
    # --------------------------------------------------------

    common_samples = [
        sample
        for sample in meta_tp.index
        if sample in counts.index
    ]


    meta_tp = meta_tp.loc[
        common_samples
    ].copy()

    counts_tp = counts.loc[
        common_samples
    ].copy()


    # --------------------------------------------------------
    # Check
    # --------------------------------------------------------

    print("\nGroups:")

    print(
        meta_tp[
            comparison_column
        ].value_counts()
    )

    print(
        "Samples:",
        len(meta_tp)
    )

    print(
        "Counts shape:",
        counts_tp.shape
    )


    # --------------------------------------------------------
    # Check both groups
    # --------------------------------------------------------

    if positive not in meta_tp[
        comparison_column
    ].values:

        print(
            f"SKIP {tp_name}: "
            f"no {positive}"
        )

        continue


    if negative not in meta_tp[
        comparison_column
    ].values:

        print(
            f"SKIP {tp_name}: "
            f"no {negative}"
        )

        continue


    # --------------------------------------------------------
    # Make sure order is identical
    # --------------------------------------------------------

    assert list(meta_tp.index) == list(
        counts_tp.index
    )


    # --------------------------------------------------------
    # Remove genes with zero counts
    # --------------------------------------------------------

    keep_genes = (
        counts_tp.sum(axis=0) > 0
    )

    counts_tp = counts_tp.loc[
        :,
        keep_genes.values
    ]


    print(
        "Genes after filtering:",
        counts_tp.shape[1]
    )


    # --------------------------------------------------------
    # Metadata factor
    # --------------------------------------------------------

    meta_tp[comparison_column] = (
        meta_tp[comparison_column]
        .astype(str)
    )


    meta_tp["batch_nexp"] = (
        meta_tp["batch_nexp"]
        .astype(str)
    )


    # --------------------------------------------------------
    # Design
    # --------------------------------------------------------

    design_formula = (
        f"~ batch_nexp + {comparison_column}"
    )

    print(
        "Design:",
        design_formula
    )


    # --------------------------------------------------------
    # DESeq2
    # --------------------------------------------------------

    print("\nRunning DESeq2...")


    dds = DeseqDataSet(
        counts=counts_tp,
        metadata=meta_tp,
        design=design_formula
    )

    dds.deseq2()


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    print(
        f"Contrast: "
        f"{positive} vs {negative}"
    )


    ds = DeseqStats(
        dds,
        contrast=[
            comparison_column,
            positive,
            negative
        ]
    )

    ds.summary()


    results = ds.results_df.copy()


    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------

    results_file = os.path.join(
        results_dir,
        f"DESeq2_{tp_name}.tsv"
    )

    results.to_csv(
        results_file,
        sep="\t"
    )


    print(
        "Saved:",
        results_file
    )


    # --------------------------------------------------------
    # DEG filtering
    # --------------------------------------------------------

    deg = results[
        (results["padj"] < 0.05) &
        (
            results["log2FoldChange"].abs() > 1
        )
    ].copy()


    deg_file = os.path.join(
        results_dir,
        f"DEGs_{tp_name}.tsv"
    )

    deg.to_csv(
        deg_file,
        sep="\t"
    )


    print(
        "Significant DEGs:",
        len(deg)
    )

    print(
        "Saved:",
        deg_file
    )


print("\n")
print("========================================")
print("ANALYSIS FINISHED")
print("========================================")
