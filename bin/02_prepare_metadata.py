import os
import numpy as np
import pandas as pd

# Path

project_dir = "/Users/darakuzmina/Desktop/CAR_T"
ann_dir = "/Users/darakuzmina/Desktop/crc_metastasis_analysis/data"

dir_raw = os.path.join(project_dir, "data", "raw")
dir_proc = os.path.join(project_dir, "data", "processed")

# 1. Match old and new clinical data

info_cart = pd.read_csv(os.path.join(dir_raw,"clinical_data.csv"))
info_new = pd.read_csv(os.path.join(dir_raw,"clinical_data_new.csv"))

clinical = info_cart.merge(info_new, on = "sample_id")
clinical["sample_id"] = clinical["sample_id"].str.replace(" ", "_") # _ in patient_id

# 2. Check matching between expression and clinical data

expression = pd.read_csv(os.path.join(dir_raw, "expression_raw_ann.tsv"), index_col=0, sep="\t")

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
        dir_proc,
        "counts.tsv"
    ),
    sep="\t"
)

meta.to_csv(
    os.path.join(
        dir_proc,
        "meta_analysis.tsv"
    ),
    sep="\t"
)



