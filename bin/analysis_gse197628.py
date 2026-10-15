import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)




# ============================================================
# PATHS
# ============================================================

H5AD = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/scRNAseq/CART_fulldataset_clustered.h5ad"
)


'''
OUTPUT_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/rnaseq_gse197628/degs_signature"
)
'''
OUTPUT_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/rnaseq_gse197628/elastic_net_signature"
)

'''
SIGNATURE_FILE = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/13_signature/"
    "CAR_T_response_signature.tsv"
)
'''


SIGNATURE_FILE = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/15_signature_validation/"
    "signature_elastic_net.csv"
)



os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# PARAMETERS

RANDOM_SEED = 42


# LOAD H5AD

print()
print("Loading h5ad in backed mode...")

adata = sc.read_h5ad(
    H5AD,
    backed="r"
)

print(
    "adata shape:",
    adata.shape
)

print(
    "adata.raw shape:",
    adata.raw.shape
)

print(adata.obs.columns.tolist())


timepoint_mask = (
    adata.obs["timepoint"].astype(str)
    == "Infusion"
)

cell_type_mask = (
    adata.obs["cell_type"].astype(str)
    == "Infusion T"
)


mask = (
    timepoint_mask
    & cell_type_mask
    & adata.obs["response"].isin(["R", "NR"])
)

obs = adata.obs.loc[mask].copy()

# LOAD SIGNATURE

signature = pd.read_csv(
    SIGNATURE_FILE
)


available_genes = (
    signature.loc[
        signature["direction"] == "responder_UP",
        "gene"
    ]
    .astype(str)
    .str.strip()
    .tolist()
)
'''



available_genes = (
    signature["gene"]
    .astype(str)
    .str.strip()
    .tolist()
)
'''

# remove duplicates while preserving order
available_genes = list(
    dict.fromkeys(
        available_genes
    )
)


# MATCH SIGNATURE GENES TO scRNA-seq
# Check
raw_genes = set(
    adata.raw.var_names
)

available_genes_sc = [
    gene
    for gene in available_genes
    if gene in raw_genes
]

missing_genes = [
    gene
    for gene in available_genes
    if gene not in raw_genes
]

print(
    "Signature genes in scRNA-seq:",
    len(available_genes_sc)
)

print(
    "Missing signature genes:",
    len(missing_genes)
)

if missing_genes:
    print(
        "Missing genes:",
        missing_genes
    )

# get genes in scRNA-seq
gene_indices = [
    adata.raw.var_names.get_loc(gene)
    for gene in available_genes_sc
]

# FAM49A -> CYRIA

selected_cell_indices = np.where(mask.values)[0]

X = adata.raw.X[
    selected_cell_indices,
    :
][:, gene_indices]

# ============================================================
# AGGREGATE CELLS TO SAMPLE LEVEL
# ============================================================

patient_ids = (
    obs["barcode"]
    .astype(str)
    .values
)

if hasattr(X, "toarray"):
    X_dense = X.toarray()
else:
    X_dense = np.asarray(X)

expression = pd.DataFrame(
    X_dense,
    columns=available_genes_sc,
    index=patient_ids
)

sample_expr = (
    expression
    .groupby(level=0)
    .mean()
)

print(
    "Sample-level expression:",
    sample_expr.shape
)

#cell counts per patient for sensitivity test

cell_counts = (
    obs["barcode"]
    .astype(str)
    .value_counts()
    .sort_index()
)

sample_metadata = (
    obs[
        [
            "barcode",
            "response",
            "product",
            "generic"
        ]
    ]
    .drop_duplicates("barcode")
    .set_index("barcode")
    .sort_index()
)

sample_metadata["n_cells"] = (
    cell_counts
)

# ============================================================
# CALCULATE SAMPLE-LEVEL SIGNATURE SCORE
# ============================================================

expression = sample_expr[
    available_genes_sc
].astype(float)

# Gene-wise Z-score across samples
gene_means = expression.mean(axis=0)
gene_stds = expression.std(axis=0, ddof=1)

z_expression = (
    expression - gene_means
).div(
    gene_stds.replace(0, np.nan),
    axis=1
)

# Mean Z-score across signature genes
signature_score = z_expression.mean(
    axis=1,
    skipna=True
)

print()
print("Signature score:")
print(signature_score)

# Description statistics
print()
print("Score summary:")
print(signature_score.describe())

sample_results = sample_metadata.copy()

sample_results["signature_score"] = (
    signature_score
)

print()
print(sample_results)

print()

print("Signature score by product and response:")

summary_product = (
    sample_results
    .groupby(
        ["product", "response"],
        observed=True
    )["signature_score"]
    .agg(
        ["count", "mean", "median", "std"]
    )
)

print(summary_product)

print()

print(
    sample_results[
        [
            "product",
            "response",
            "n_cells",
            "signature_score"
        ]
    ]
    .sort_values(
        ["product", "response", "signature_score"]
    )
)

# R vs NR WITHIN EACH PRODUCT

results = []
for product in ["Kymriah", "Yescarta"]:

    subset = sample_results[
        sample_results["product"] == product
    ].copy()

    scores_R = subset.loc[
        subset["response"] == "R",
        "signature_score"
    ].dropna()

    scores_NR = subset.loc[
        subset["response"] == "NR",
        "signature_score"
    ].dropna()

    u_stat, p_value = mannwhitneyu(
        scores_R,
        scores_NR,
        alternative="two-sided"
    )

    print()
    print("=" * 60)
    print(product)
    print("=" * 60)

    print("R n:", len(scores_R))
    print("NR n:", len(scores_NR))

    print("R median:", scores_R.median())
    print("NR median:", scores_NR.median())

    print("R mean:", scores_R.mean())
    print("NR mean:", scores_NR.mean())

    print("Mann-Whitney U:", u_stat)
    print("P-value:", p_value)

    n_R = len(scores_R)
    n_NR = len(scores_NR)

    rank_biserial = (
        2 * u_stat / (n_R * n_NR)
        - 1
    )

    print(
        "Rank-biserial effect:",
        rank_biserial
    )

    results.append({
                "n_R": n_R,
                "n_NR": n_NR,
                "R_median": scores_R.median(),
                "NR_median": scores_NR.median(),
                "R_mean": scores_R.mean(),
                "NR_mean": scores_NR.mean(),
                "Mann-Whitney U": u_stat,
                "p_value": p_value,
                "rank_biserial": rank_biserial
            })

results = pd.DataFrame(results)

results.to_csv(os.path.join(OUTPUT_DIR, "test_signature.csv"))
# ============================================================
# PLOT: SIGNATURE SCORE BY PRODUCT AND RESPONSE
# ============================================================
plot_data = sample_results.copy()

plot_data["response"] = pd.Categorical(
    plot_data["response"], categories=["NR", "R"], ordered=True
)

# 1. Создаем фигуру и сохраняем ссылку на нее (fig) для последующего сохранения
fig, ax = plt.subplots(figsize=(8, 6))

# 2. Строим stripplot, передавая созданную ось в параметр ax
g = sns.stripplot(
    data=plot_data,
    x="response",
    y="signature_score",
    hue="product",
    dodge=True,
    jitter=0.08,
    size=8,
    ax=ax,
)

plt.axhline(0, linestyle="--", linewidth=1)

# 3. Настраиваем подписи напрямую через объект g (или ax), без .ax_heatmap
g.set_xlabel("Response")
g.set_ylabel("Signature score")
g.set_title(
    "CAR-T response signature in independent scRNA-seq cohort", pad=20
)

plt.legend(title="Product")
plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "response_signsture.png")

# 4. Сохраняем фигуру с помощью сохраненной переменной fig
fig.savefig(output_file, dpi=300, bbox_inches="tight")

plt.close()


#sensitivity analysis

# SENSITIVITY ANALYSIS: MINIMUM NUMBER OF CELLS

MIN_CELLS_THRESHOLDS = [
    50,
    100,
    200,
    500
]

sensitivity_results = []

for min_cells in MIN_CELLS_THRESHOLDS:

    data = sample_results[
        sample_results["n_cells"] >= min_cells
    ].copy()

    for product in ["Kymriah", "Yescarta"]:

        subset = data[
            data["product"] == product
        ].copy()

        scores_R = subset.loc[
            subset["response"] == "R",
            "signature_score"
        ].dropna()

        scores_NR = subset.loc[
            subset["response"] == "NR",
            "signature_score"
        ].dropna()

        n_R = len(scores_R)
        n_NR = len(scores_NR)

        # Need at least 2 samples in each group
        if n_R < 2 or n_NR < 2:

            sensitivity_results.append({
                "min_cells": min_cells,
                "product": product,
                "n_total": len(subset),
                "n_R": n_R,
                "n_NR": n_NR,
                "median_R": np.nan,
                "median_NR": np.nan,
                "mean_R": np.nan,
                "mean_NR": np.nan,
                "median_difference": np.nan,
                "U": np.nan,
                "p_value": np.nan,
                "rank_biserial": np.nan
            })

            continue

        u_stat, p_value = mannwhitneyu(
            scores_R,
            scores_NR,
            alternative="two-sided"
        )

        rank_biserial = (
            2 * u_stat / (n_R * n_NR)
            - 1
        )

        sensitivity_results.append({
            "min_cells": min_cells,
            "product": product,
            "n_total": len(subset),
            "n_R": n_R,
            "n_NR": n_NR,
            "median_R": scores_R.median(),
            "median_NR": scores_NR.median(),
            "mean_R": scores_R.mean(),
            "mean_NR": scores_NR.mean(),
            "median_difference": (
                scores_R.median()
                - scores_NR.median()
            ),
            "U": u_stat,
            "p_value": p_value,
            "rank_biserial": rank_biserial
        })


sensitivity_results = pd.DataFrame(
    sensitivity_results
)

sensitivity_results.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_results_gse197628.csv"))

roc_results = []

for product in ["Kymriah", "Yescarta"]:

    subset = sample_results[
        sample_results["product"] == product
    ].copy()

    subset = subset[
        subset["response"].isin(["R", "NR"])
    ].copy()

    y_true = (
        subset["response"] == "R"
    ).astype(int)

    y_score = subset["signature_score"]

    auc = roc_auc_score(
        y_true,
        y_score
    )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score
    )

    roc_results.append({
        "product": product,
        "n_total": len(subset),
        "n_R": (y_true == 1).sum(),
        "n_NR": (y_true == 0).sum(),
        "AUC": auc
    })

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(
        f"ROC — {product}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"ROC_{product}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


roc_results = pd.DataFrame(
    roc_results
)

roc_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "roc_auc_results.tsv"
    ),
    sep="\t",
    index=False
)