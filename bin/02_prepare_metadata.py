import os
import argparse
import numpy as np
import pandas as pd


# Arguments


parser = argparse.ArgumentParser(
    description="Prepare count matrix and clinical metadata"
)

parser.add_argument(
    "--expression",
    required=True
)

parser.add_argument(
    "--clinical",
    required=True
)

parser.add_argument(
    "--clinical_new",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

args = parser.parse_args()

# Output directory

os.makedirs(
    args.output_dir,
    exist_ok=True
)


# Load data

expression = pd.read_csv(
    args.expression,
    index_col=0,
    sep="\t"
)

info_cart = pd.read_csv(
    args.clinical
)

info_new = pd.read_csv(
    args.clinical_new
)


# 1. Match old and new clinical data

clinical = info_cart.merge(info_new, on = "sample_id")
clinical["sample_id"] = clinical["sample_id"].str.replace(" ", "_") # _ in patient_id

# 2. Check matching between expression and clinical data


common_samples = sorted(
    set(expression.columns)
    &
    set(clinical["sample_id"])
)

if not common_samples:
    raise ValueError(
        "No common samples between expression and clinical data."
    )


expression = expression[
    common_samples
]

meta = (
    clinical[
        clinical["sample_id"]
        .isin(common_samples)
    ]
    .set_index("sample_id")
    .reindex(common_samples)
)


assert (
    list(expression.columns)
    ==
    list(meta.index)
)

# 2. Make category covariates
categorical_columns = [
    "batch_nexp",
    "batch_seq",
    "group_analysis_1",
    "group_analysis_2",
    "response",
    "aggressiveness",
    "bone_marrow",
    "lesion",
    "timepoint",
    "patient_id"
]

for col in categorical_columns:

    if col in meta.columns:
        meta[col] = (
            meta[col]
            .astype(str)
        )


# 4. Filter counts
#remove zero counts
counts = expression.copy()[(expression > 0).any(axis=1)].T  # samples × genes
# match indices
common = sorted(set(counts.index) & set(meta.index))
counts = counts.reindex(common)
meta = meta.reindex(common)
assert list(counts.index) == list(meta.index), "Indices don't match"

counts.to_csv(
    os.path.join(
        args.output_dir,
        "counts.tsv"
    ),
    sep="\t"
)

meta.to_csv(
    os.path.join(
        args.output_dir,
        "meta_analysis.tsv"
    ),
    sep="\t"
)



