import os
import glob

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/E-MTAB-12407"
)

RNA_DIR = os.path.join(
    BASE_DIR,
    "rna_norm"
)

ANNOTATION_DIR = os.path.join(
    BASE_DIR,
    "annotations"
)

SIGNATURE_FILE = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/13_signature/"
    "CAR_T_response_signature.tsv"
)

OUTPUT_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/rnaseq_emtab12407/degs_signature"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

# Используем только CART+ клетки
CART_LABEL = "CART+"

# Clinical endpoint
RESPONSE_COLUMN = "response_1month_2category"

# Patient identifier
PATIENT_COLUMN = "Sample_Name"

# Cell identifier в annotation
CELL_ID_COLUMN = None
# None = использовать первый столбец файла как cell ID

# ВАЖНО:
# твоя основная signature — responder_UP.
# Если в signature-файле другая структура, код остановится
# и покажет названия колонок.
SIGNATURE_DIRECTION = "responder_UP"


# ============================================================
# 1. READ SIGNATURE

signature = pd.read_csv(
    SIGNATURE_FILE,  sep="\t"
)


signature_genes = (
    signature.loc[
        signature["direction"] == "responder_UP",
        "gene"
    ]
    .astype(str)
    .str.strip()
    .tolist()
)



signature_genes = list(
    dict.fromkeys(signature_genes)
)


# ============================================================
# 2. FIND PATIENT FILES
# ============================================================

rna_files = sorted(
    glob.glob(
        os.path.join(
            RNA_DIR,
            "*_rnanorm.tsv"
        )
    )
)

print("Number of RNA files:", len(rna_files))

for f in rna_files:
    print(os.path.basename(f))

# 3. PROCESS ONE PATIENT AT A TIME

patient_expression = {}
patient_metadata = []

all_missing_signature = {}

for i, rna_file in enumerate(rna_files, start=1):

    filename = os.path.basename(rna_file)

    # patient prefix

    patient_prefix = filename.replace(
        "_rnanorm.tsv",
        ""
    )

    annotation_file = os.path.join(
        ANNOTATION_DIR,
        patient_prefix + "_annotations.tsv"
    )

    print("\n" + "=" * 70)
    print(
        f"[{i}/{len(rna_files)}] {patient_prefix}"
    )
    print("=" * 70)

    if not os.path.exists(annotation_file):

        print(
            "WARNING: annotation file not found:",
            annotation_file
        )

        continue


    # READ RNA
    # ========================================================

    rna = pd.read_csv(
        rna_file,
        sep="\t",
        index_col=0
    )

    # Make gene names strings
    rna.index = (
        rna.index
        .astype(str)
        .str.strip()
    )

    # Remove duplicated genes if any
    if rna.index.duplicated().any():

        print(
            "WARNING:",
            rna.index.duplicated().sum(),
            "duplicated gene names."
        )

        rna = rna[
            ~rna.index.duplicated(
                keep="first"
            )
        ]


    # ========================================================
    # READ ANNOTATION
    # ========================================================

    ann = pd.read_csv(
        annotation_file,
        sep="\t",
        index_col=0
    )

    ann.index = (
        ann.index
        .astype(str)
        .str.strip()
    )


    # ========================================================
    # BASIC CHECKS
    # ========================================================

    print(
        "RNA cells:",
        rna.shape[1]
    )

    print(
        "Annotated cells:",
        ann.shape[0]
    )

    common_cells = (
        rna.columns
        .astype(str)
        .str.strip()
        .intersection(ann.index)
    )

    print(
        "Common cells:",
        len(common_cells)
    )

    if len(common_cells) == 0:

        print(
            "ERROR: no common cell IDs."
        )

        continue


    # Keep only cells present in both files

    common_cells = list(common_cells)

    rna = rna.loc[
        :,
        common_cells
    ]

    ann = ann.loc[
        common_cells
    ]


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    response_values = (
        ann[RESPONSE_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print(
        "Response:",
        response_values
    )

    if len(response_values) != 1:

        raise ValueError(
            f"{patient_prefix}: "
            f"multiple response values found: "
            f"{response_values}"
        )

    response = response_values[0]


    # ========================================================
    # CHECK PATIENT
    # ========================================================

    patient_values = (
        ann[PATIENT_COLUMN]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print(
        "Patient:",
        patient_values
    )

    if len(patient_values) != 1:

        raise ValueError(
            f"{patient_prefix}: "
            f"multiple Sample_Name values found: "
            f"{patient_values}"
        )

    patient = patient_values[0]


    # ========================================================
    # SELECT CART+ CELLS
    # ========================================================

    cart_mask = (
        ann["CART.label"]
        .astype(str)
        .str.strip()
        .eq(CART_LABEL)
    )

    cart_cells = ann.index[
        cart_mask
    ]

    print(
        "CART+ cells:",
        len(cart_cells)
    )

    if len(cart_cells) == 0:

        print(
            "WARNING: no CART+ cells."
        )

        continue


    # ========================================================
    # CART+ EXPRESSION
    # ========================================================

    rna_cart = rna.loc[
        :,
        cart_cells
    ]


    # ========================================================
    # MEAN EXPRESSION PER PATIENT
    # ========================================================

    patient_expr = rna_cart.mean(
        axis=1
    )

    patient_expression[patient] = (
        patient_expr
    )


    # ========================================================
    # SIGNATURE AVAILABILITY
    # ========================================================

    available = [
        gene
        for gene in signature_genes
        if gene in patient_expr.index
    ]

    missing = [
        gene
        for gene in signature_genes
        if gene not in patient_expr.index
    ]

    all_missing_signature[patient] = missing

    print(
        "Signature genes found:",
        len(available),
        "/",
        len(signature_genes)
    )

    patient_metadata.append(
        {
            "patient": patient,
            "response": response,
            "n_total_cells": rna.shape[1],
            "n_CART_cells": len(cart_cells)
        }
    )


# ============================================================
# 4. PATIENT × GENE MATRIX
# ============================================================

print("\n" + "=" * 70)
print("BUILDING PATIENT-LEVEL EXPRESSION MATRIX")
print("=" * 70)

expression = pd.DataFrame(
    patient_expression
)

expression.index.name = "gene"

print(
    "Expression matrix:",
    expression.shape
)

print(
    "Patients:",
    expression.shape[1]
)

print(
    "Genes:",
    expression.shape[0]
)


# ============================================================
# SAVE PATIENT EXPRESSION
# ============================================================

expression.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "patient_level_expression_CARTplus.tsv"
    ),
    sep="\t"
)


# ============================================================
# 5. KEEP SIGNATURE GENES
# ============================================================

available_signature_genes = [
    gene
    for gene in signature_genes
    if gene in expression.index
]

missing_signature_genes = [
    gene
    for gene in signature_genes
    if gene not in expression.index
]

print("\n" + "=" * 70)
print("SIGNATURE")
print("=" * 70)

print(
    "Signature genes:",
    len(signature_genes)
)

print(
    "Available genes:",
    len(available_signature_genes)
)

print(
    "Missing genes:",
    len(missing_signature_genes)
)

if missing_signature_genes:

    print(
        "\nMissing genes:"
    )

    print(
        missing_signature_genes
    )


if len(available_signature_genes) == 0:

    raise ValueError(
        "None of the signature genes were found."
    )


# ============================================================
# 6. SIGNATURE EXPRESSION
# ============================================================

signature_expression = expression.loc[
    available_signature_genes
]


# ============================================================
# 7. GENE-WISE Z-SCORE ACROSS PATIENTS
# ============================================================

signature_z = (
    signature_expression
    .sub(
        signature_expression.mean(
            axis=1
        ),
        axis=0
    )
    .div(
        signature_expression.std(
            axis=1,
            ddof=0
        ).replace(0, np.nan),
        axis=0
    )
)


# Remove genes with zero variance

signature_z = signature_z.dropna(
    axis=0,
    how="all"
)

print(
    "\nGenes used for score:",
    signature_z.shape[0]
)


# ============================================================
# 8. PATIENT SIGNATURE SCORE
# ============================================================

signature_score = (
    signature_z.mean(
        axis=0
    )
)


# ============================================================
# 9. BUILD FINAL RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "patient": signature_score.index,
        "signature_score": signature_score.values
    }
)

metadata_df = pd.DataFrame(
    patient_metadata
)

results = results.merge(
    metadata_df,
    on="patient",
    how="left"
)

results = results.sort_values(
    "signature_score",
    ascending=False
)


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "patient_signature_scores.tsv"
)

results.to_csv(
    results_file,
    sep="\t",
    index=False
)

print(
    "\nSaved:",
    results_file
)


# ============================================================
# 11. RESPONSE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RESPONSE SUMMARY")
print("=" * 70)

print(
    results["response"].value_counts(
        dropna=False
    )
)

print(
    "\nSignature scores:"
)

print(
    results[
        [
            "patient",
            "response",
            "n_CART_cells",
            "signature_score"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 12. MANN-WHITNEY
# ============================================================

R = results.loc[
    results["response"] == "R",
    "signature_score"
].dropna()

NR = results.loc[
    results["response"] == "NR",
    "signature_score"
].dropna()

print("\n" + "=" * 70)
print("R vs NR")
print("=" * 70)

print(
    "R:",
    len(R)
)

print(
    "NR:",
    len(NR)
)

if len(R) > 0 and len(NR) > 0:

    U, p = mannwhitneyu(
        R,
        NR,
        alternative="two-sided"
    )

    # rank-biserial correlation
    rank_biserial = (
        2 * U / (len(R) * len(NR))
        - 1
    )

    print(
        "Mann-Whitney U:",
        U
    )

    print(
        "p-value:",
        p
    )

    print(
        "Rank-biserial effect:",
        rank_biserial
    )

else:

    print(
        "Cannot calculate R vs NR:"
        " one group is empty."
    )


# ============================================================
# 13. ROC / AUC
# ============================================================

print("\n" + "=" * 70)
print("ROC / AUC")
print("=" * 70)

roc_results = results[
    results["response"].isin(
        ["R", "NR"]
    )
].copy()

roc_results["y"] = (
    roc_results["response"]
    .map(
        {
            "NR": 0,
            "R": 1
        }
    )
)

if (
    roc_results["y"].nunique() == 2
):

    auc = roc_auc_score(
        roc_results["y"],
        roc_results["signature_score"]
    )

    fpr, tpr, thresholds = roc_curve(
        roc_results["y"],
        roc_results["signature_score"]
    )

    print(
        "AUC:",
        auc
    )

    roc_table = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        }
    )

    roc_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "ROC_curve.tsv"
        ),
        sep="\t",
        index=False
    )

else:

    print(
        "ROC cannot be calculated:"
        " only one response class."
    )


# ============================================================
# 14. SAVE MISSING SIGNATURE GENES
# ============================================================

missing_table = pd.DataFrame(
    {
        "gene": missing_signature_genes
    }
)

missing_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "missing_signature_genes.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 15. SAVE SIGNATURE Z-SCORES
# ============================================================

signature_z.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "signature_gene_zscores.tsv"
    ),
    sep="\t"
)

