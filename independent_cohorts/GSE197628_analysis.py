
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

import gseapy as gp

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference


from scipy.stats import mannwhitneyu

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)



# PATHS


H5AD = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/scRNAseq/CART_fulldataset_clustered.h5ad"
)



OUTPUT_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/GSE197628"
)


SIGNATURE_FILE = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/CAR_T_response_signature.tsv"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# PARAMETERS

RANDOM_SEED = 42


# LOAD H5AD

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


# Check data
print("Checking adata.X...")
print("Shape:", adata.X.shape)
print("Dtype:", adata.X.dtype)

print("\nChecking adata.layers['umi']...")
print("Shape:", adata.layers["umi"].shape)
print("Dtype:", adata.layers["umi"].dtype)

print("\nChecking adata.raw.X...")
print("Shape:", adata.raw.X.shape)
print("Dtype:", adata.raw.X.dtype)


def inspect_matrix(matrix, name):
    print(f"\n{name}")

    # Read only a small subset of cells and genes
    subset = matrix[:100, :100]

    if sparse.issparse(subset):
        values = subset.data
    else:
        values = np.asarray(subset).ravel()

    if len(values) == 0:
        print("No stored non-zero values")
        return

    print("Min:", values.min())
    print("Max:", values.max())
    print("Non-integer values:",
          np.sum(~np.isclose(values, np.rint(values), atol=1e-6)))
    print("Example values:", values[:20])


inspect_matrix(adata.X, "adata.X")
inspect_matrix(adata.layers["umi"], "adata.layers['umi']")
inspect_matrix(adata.raw.X, "adata.raw.X")


# LOAD SIGNATURE

signature = pd.read_csv(
    SIGNATURE_FILE, sep="\t"
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


# AGGREGATE CELLS TO SAMPLE LEVEL


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


# CALCULATE SAMPLE-LEVEL SIGNATURE SCORE


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

# PLOT: SIGNATURE SCORE BY PRODUCT AND RESPONSE

plot_data = sample_results.copy()

plot_data["response"] = pd.Categorical(
    plot_data["response"], categories=["NR", "R"], ordered=True
)

fig, ax = plt.subplots(figsize=(8, 6))
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


g.set_xlabel("Response")
g.set_ylabel("Signature score")
g.set_title(
    "CAR-T response signature in independent scRNA-seq cohort", pad=20
)

plt.legend(title="Product")
plt.tight_layout()

output_file = os.path.join(OUTPUT_DIR, "response_signsture.png")

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


# HEATMAP: SIGNATURE GENES ACROSS SAMPLES

# Use the sample-level expression matrix already calculated above
heatmap_data = sample_expr.copy()

# Keep signature genes that are present
heatmap_genes = [
    gene for gene in available_genes_sc
    if gene in heatmap_data.columns
]

heatmap_data = heatmap_data[heatmap_genes]

# Z-score each gene across samples
heatmap_z = (
    heatmap_data - heatmap_data.mean(axis=0)
).div(
    heatmap_data.std(axis=0, ddof=1).replace(0, np.nan),
    axis=1
).fillna(0)

# Sample annotations
heatmap_meta = sample_metadata.reindex(heatmap_z.index).copy()

heatmap_meta["response"] = heatmap_meta["response"].astype(str)
heatmap_meta["product"] = heatmap_meta["product"].astype(str)


# Sort samples by product and response


heatmap_meta["response"] = pd.Categorical(
    heatmap_meta["response"],
    categories=["NR", "R"],
    ordered=True
)

sort_order = heatmap_meta.sort_values(
    "response",
    kind="stable"
).index

heatmap_z = heatmap_z.reindex(sort_order)
heatmap_meta = heatmap_meta.reindex(sort_order)

response_palette = {
    "NR": "#1F77B4",
    "R": "#D62728"
}

product_palette = {
    "Kymriah": "#2CA02C",
    "Yescarta": "#9467BD"
}

col_colors = pd.DataFrame({
    "Response": heatmap_meta["response"].map(response_palette),
    "Product": heatmap_meta["product"].map(product_palette)
}, index=heatmap_z.index)

# Draw heatmap: genes as rows, samples as columns
fig_width = max(10, heatmap_z.shape[0] * 0.45)
fig_height = max(6, len(heatmap_genes) * 0.25)

g = sns.clustermap(
    heatmap_z.T,
    row_cluster=True,
    col_cluster=False,
    cmap="vlag",
    center=0,
    col_colors=col_colors,
    xticklabels=True,
    yticklabels=True,
    figsize=(fig_width, fig_height),
    dendrogram_ratio=(0.12, 0.05),
    cbar_kws={"label": "Gene-wise Z-score"}
)
from matplotlib.patches import Patch
legend_elements = [
    Patch(
        facecolor=color,
        label=label
    )
    for label, color
    in response_palette.items()
]


legend_elements2 = [
    Patch(
        facecolor=color,
        label=label
    )
    for label, color
    in product_palette.items()
]

legend1 = g.ax_heatmap.legend(
    handles=legend_elements,
    title="Response",
    bbox_to_anchor=(1.15, 1),
    loc="upper left",
    frameon=False
)

# Preserve the first legend
g.ax_heatmap.add_artist(legend1)

# T-cell subtype legend
g.ax_heatmap.legend(
    handles=legend_elements2,
    title="CAR-T product type",
    bbox_to_anchor=(1.15, 0.78),
    loc="upper left",
    frameon=False
)

g.ax_heatmap.set_xlabel("Samples")
g.ax_heatmap.set_ylabel("Signature genes")

g.ax_heatmap.set_title(
    "CAR-T response signature in GSE197628",
    pad=20,
    y=1.03
)

heatmap_file = os.path.join(
    OUTPUT_DIR,
    "GSE197628_signature_gene_heatmap.png"
)

g.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

print("Saved signature heatmap:", heatmap_file)


# Save the exact expression matrix used in the heatmap
heatmap_z.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "GSE197628_signature_gene_heatmap_zscores.tsv"
    ),
    sep="\t"
)



# RANKED GSEA: PSEUDOBULK R VS NR

PANEL_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data"
)

panel_files = [
    ("P1", "gene_panel_1.csv"),
    ("P2", "gene_panel_5.csv"),
    ("P3", "gene_panel_3.csv"),
    ("P4", "gene_panel_4.csv")
]

panels = {}

for prefix, filename in panel_files:

    panel_path = os.path.join(PANEL_DIR, filename)

    if not os.path.exists(panel_path):
        print("WARNING: panel not found:", panel_path)
        continue

    panel_df = pd.read_csv(panel_path)

    for col in panel_df.columns:

        genes = (
            panel_df[col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        genes = [
            gene for gene in genes
            if gene and gene.lower() != "nan"
        ]

        panel_name = f"{prefix}_{col}"

        panels[panel_name] = list(
            dict.fromkeys(genes)
        )

print("\nLoaded gene panels:")
for panel_name, genes in panels.items():
    print(panel_name, len(genes), "genes")


# PREPARE PSEUDOBULK COUNTS — MEMORY-EFFICIENT VERSION


from scipy import sparse

selected_cell_indices = np.where(mask.values)[0]

cell_meta = obs.copy()
cell_meta["barcode"] = cell_meta["barcode"].astype(str)

# All genes in adata.raw
#raw_gene_names = adata.raw.var_names.astype(str)
raw_gene_names = adata.var_names.astype(str)

if len(selected_cell_indices) != len(cell_meta):
    raise ValueError(
        "Mismatch between selected cells and metadata."
    )

# Map every selected cell to its pseudobulk sample
sample_codes, sample_names = pd.factorize(
    cell_meta["barcode"],
    sort=True
)

n_samples = len(sample_names)
n_genes = len(raw_gene_names)

print("Selected cells:", len(selected_cell_indices))
print("Pseudobulk samples:", n_samples)
print("Genes:", n_genes)


# Aggregate counts in chunks using sparse matrices

# This matrix contains one row per sample and one column per gene.
# It avoids converting all single-cell counts to a dense array.
pseudobulk_sparse = sparse.csr_matrix(
    (n_samples, n_genes),
    dtype=np.float64
)

CHUNK_SIZE = 2000

for start in range(0, len(selected_cell_indices), CHUNK_SIZE):

    end = min(
        start + CHUNK_SIZE,
        len(selected_cell_indices)
    )

    cell_idx_chunk = selected_cell_indices[start:end]
    sample_idx_chunk = sample_codes[start:end]

    #X_chunk = adata.raw.X[cell_idx_chunk, :]
    X_chunk = adata.layers["umi"][cell_idx_chunk, :]

    if not sparse.issparse(X_chunk):
        X_chunk = sparse.csr_matrix(X_chunk)
    else:
        X_chunk = X_chunk.tocsr()

    # Convert the cell-to-sample mapping into a sparse matrix.
    aggregation_matrix = sparse.csr_matrix(
        (
            np.ones(len(sample_idx_chunk), dtype=np.float64),
            (
                sample_idx_chunk,
                np.arange(len(sample_idx_chunk))
            )
        ),
        shape=(n_samples, len(sample_idx_chunk))
    )

    # Sum cells in this chunk by sample
    pseudobulk_sparse += aggregation_matrix @ X_chunk

    if end % 10000 == 0 or end == len(selected_cell_indices):
        print(f"Aggregated {end}/{len(selected_cell_indices)} cells")

# ------------------------------------------------------------
# Build pseudobulk metadata
# ------------------------------------------------------------

pseudobulk_meta_rows = []

for barcode in sample_names:

    barcode_meta = cell_meta[
        cell_meta["barcode"] == barcode
    ]

    responses = barcode_meta["response"].dropna().unique()
    products = barcode_meta["product"].dropna().unique()

    if len(responses) != 1:
        print("Skipping inconsistent response:", barcode)
        pseudobulk_meta_rows.append({
            "barcode": barcode,
            "response": np.nan,
            "product": np.nan,
            "n_cells": len(barcode_meta)
        })
        continue

    if len(products) != 1:
        print("Skipping inconsistent product:", barcode)
        pseudobulk_meta_rows.append({
            "barcode": barcode,
            "response": responses[0],
            "product": np.nan,
            "n_cells": len(barcode_meta)
        })
        continue

    pseudobulk_meta_rows.append({
        "barcode": barcode,
        "response": responses[0],
        "product": products[0],
        "n_cells": len(barcode_meta)
    })

pseudobulk_meta = (
    pd.DataFrame(pseudobulk_meta_rows)
    .set_index("barcode")
)


# Convert to DataFrame and sum duplicate gene names


# The pseudobulk matrix is much smaller than the cell-level matrix.
pseudobulk_counts = pd.DataFrame.sparse.from_spmatrix(
    pseudobulk_sparse,
    index=sample_names,
    columns=raw_gene_names
)

# Aggregate duplicate gene symbols if present
pseudobulk_counts = pseudobulk_counts.T.groupby(
    level=0,
    sort=False
).sum().T

# Remove genes with zero total counts
pseudobulk_counts = pseudobulk_counts.loc[
    :, pseudobulk_counts.sum(axis=0) > 0
]

# Save results
pseudobulk_counts.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "GSE197628_pseudobulk_counts.tsv"
    ),
    sep="\t"
)

pseudobulk_meta.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "GSE197628_pseudobulk_metadata.tsv"
    ),
    sep="\t"
)

print("Pseudobulk matrix:", pseudobulk_counts.shape)
print("\nResponse counts:")
print(pseudobulk_meta["response"].value_counts())

print("\nProduct counts:")
print(pseudobulk_meta["product"].value_counts())
print("Raw matrix type:", type(adata.raw.X))
print("Example values:", adata.raw.X[:5, :10])



# DESEQ2 AND PRERANK GSEA


gsea_results = []
de_results = []

# Analyze each CAR-T product separately
for product in ["Kymriah", "Yescarta"]:

    print("\n" + "=" * 70)
    print("Product:", product)
    print("=" * 70)

    meta_product = pseudobulk_meta[
        pseudobulk_meta["product"] == product
    ].copy()

    meta_product = meta_product[
        meta_product["response"].isin(["R", "NR"])
    ]

    sample_ids = meta_product.index.tolist()

    counts_product = pseudobulk_counts.loc[
        sample_ids
    ].copy()

    counts_product = counts_product.loc[
        :, counts_product.sum(axis=0) > 0
    ]

    print("Samples:", len(meta_product))
    print(meta_product["response"].value_counts())

    if (
        (meta_product["response"] == "R").sum() < 2
        or (meta_product["response"] == "NR").sum() < 2
    ):
        print(
            "Skipping DESeq2: need at least 2 samples in both "
            "response groups for this product."
        )
        continue

    # DESeq2 input: rows = samples, columns = genes
    counts_deseq = counts_product.copy()
    counts_deseq.index = counts_deseq.index.astype(str)
    counts_deseq.columns = counts_deseq.columns.astype(str)

    meta_deseq = meta_product[["response"]].copy()
    meta_deseq.index = meta_deseq.index.astype(str)

    # NR is the reference; positive statistic means higher in R
    meta_deseq["response"] = pd.Categorical(
        meta_deseq["response"],
        categories=["NR", "R"]
    )

    inference = DefaultInference(n_cpus=4)

    dds = DeseqDataSet(
        counts=counts_deseq,
        metadata=meta_deseq,
        design="~response",
        refit_cooks=True,
        inference=inference
    )

    dds.deseq2()

    stats = DeseqStats(
        dds,
        contrast=["response", "R", "NR"],
        inference=inference
    )

    stats.summary()

    de = stats.results_df.copy()
    de.index.name = "gene"
    de = de.reset_index()

    de["product"] = product

    de_file = os.path.join(
        OUTPUT_DIR,
        f"GSE197628_pseudobulk_DE_{product}.tsv"
    )

    de.to_csv(
        de_file,
        sep="\t",
        index=False
    )

    print("Saved DE results:", de_file)
    de_results.append(de)

    # Rank genes by Wald statistic
    ranking = de[["gene", "stat"]].dropna().copy()

    ranking["gene"] = ranking["gene"].astype(str)

    ranking = ranking[
        np.isfinite(ranking["stat"])
    ]

    ranking = (
        ranking
        .drop_duplicates(subset="gene")
        .sort_values("stat", ascending=False)
    )

    ranking_file = os.path.join(
        OUTPUT_DIR,
        f"GSE197628_GSEA_ranking_{product}.tsv"
    )

    ranking.to_csv(
        ranking_file,
        sep="\t",
        index=False
    )

    print("Saved ranking:", ranking_file)

    # Run preranked GSEA separately for every panel
    for panel_name, panel_genes in panels.items():

        overlap = set(panel_genes).intersection(
            set(ranking["gene"])
        )

        print(
            panel_name,
            "- genes in ranking:",
            len(overlap)
        )

        if len(overlap) < 3:
            print("Skipping: fewer than 3 overlapping genes.")
            continue

        try:
            pre_res = gp.prerank(
                rnk=ranking[["gene", "stat"]],
                gene_sets={panel_name: panel_genes},
                min_size=3,
                max_size=500,
                permutation_num=1000,
                seed=RANDOM_SEED,
                outdir=None,
                verbose=False
            )

            res = pre_res.res2d.copy()

            if res.empty:
                continue

            res["panel"] = panel_name
            res["product"] = product

            gsea_results.append(res)

        except Exception as e:
            print(
                "GSEA failed:",
                product,
                panel_name,
                str(e)
            )



# SAVE GSEA RESULTS AND PLOT NES HEATMAP


if gsea_results:

    gsea_all = pd.concat(
        gsea_results,
        ignore_index=True
    )

    gsea_file = os.path.join(
        OUTPUT_DIR,
        "GSE197628_ranked_GSEA_all_panels.tsv"
    )

    gsea_all.to_csv(
        gsea_file,
        sep="\t",
        index=False
    )

    print("\nSaved GSEA results:", gsea_file)

    # Locate output columns irrespective of GSEApy formatting
    col_lookup = {
        str(col).strip().lower(): col
        for col in gsea_all.columns
    }

    nes_col = col_lookup.get("nes")
    fdr_col = col_lookup.get("fdr q-val")

    if nes_col is None or fdr_col is None:
        raise KeyError(
            "Could not identify NES/FDR columns. Available columns: "
            + str(gsea_all.columns.tolist())
        )

    gsea_all["NES"] = pd.to_numeric(
        gsea_all[nes_col],
        errors="coerce"
    )

    gsea_all["FDR q-val"] = pd.to_numeric(
        gsea_all[fdr_col],
        errors="coerce"
    )

    # Rows = panels, columns = products
    nes_matrix = gsea_all.pivot_table(
        index="panel",
        columns="product",
        values="NES",
        aggfunc="first"
    )

    fdr_matrix = gsea_all.pivot_table(
        index="panel",
        columns="product",
        values="FDR q-val",
        aggfunc="first"
    )

    #panel_order = list(panels.keys())

    selected_panels = [
    # Naive / memory
            "P1_Naive_Like",
            "P3_T_EarlyMem",
            "P1_Central_Memory",
            "P3_T_NonExhausted",
        
            # Effector-memory
            "P1_Effector_Memory",
            "P1_Tissue_resident_memory",
        
            # Effector
            "P1_Effector",
            "P1_Cytotoxic",
        
            # Activation / functional states
            "P4_BTM_T cell_activation",
            "P1_Th1",
            "P1_TfH",
            "P1_Treg",
        
            # Exhaustion trajectory
            "P1_Pre_Exhausted",
            "P2_TPEX_up",
            "P2_TEX_int_up",
            "P2_TEX_term_up",
            "P4_Exhaustion_Wherry",
            "P4_Exhaustion_Long",
            "P1_Exhausted_T",
            "P3_Exhausted_T"
]

    # Keep only available signatures, preserving the specified order
    panel_order = [
    p for p in selected_panels
    if p in nes_matrix.index
]

    nes_matrix = nes_matrix.reindex(index=panel_order)

    desired_cell_order = [
    "Kymriah", "Yescarta"
]

    nes_matrix = nes_matrix.reindex(
    columns=[
        c for c in desired_cell_order
        if c in nes_matrix.columns
    ]
)

    fdr_matrix = fdr_matrix.reindex(
        index=nes_matrix.index,
        columns=nes_matrix.columns
    )

    # Significance annotations:
    # *  FDR < 0.05 and |NES| > 1
    # ** FDR < 0.01 and |NES| > 1
    annot = pd.DataFrame(
        "",
        index=nes_matrix.index,
        columns=nes_matrix.columns
    )

    for panel_name in nes_matrix.index:
        for ct in nes_matrix.columns:

            nes = nes_matrix.loc[panel_name, product]
            fdr = fdr_matrix.loc[panel_name, product]

            if pd.isna(nes) or pd.isna(fdr):
                continue

            if abs(nes) > 1 and fdr < 0.05:
                annot.loc[panel_name, ct] = "**"
            
            elif abs(nes) > 1 and fdr < 0.1:
                annot.loc[panel_name, ct] = f"*"

    if nes_matrix.empty:
        print(
            "NES matrix is empty. Check cell_type names "
            "and GSEA results."
        )
    else:
        max_abs_nes = np.nanmax(
        np.abs(nes_matrix.to_numpy())
        )
      
    
        if not np.isfinite(max_abs_nes) or max_abs_nes == 0:
                max_abs_nes = 1

        plt.figure(
        figsize=(8, max(5, len(nes_matrix) * 0.4))
    )

        sns.heatmap(
        nes_matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs_nes,
        vmax=max_abs_nes,
        annot=annot,
        fmt="",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Normalized Enrichment Score (NES)"}
    )

        plt.xlabel("CAR-T product")
        plt.ylabel("Gene panel")
        plt.title("GSE197628: ranked GSEA by CAR-T product")
        plt.tight_layout()

        gsea_heatmap_file = os.path.join(
        OUTPUT_DIR,
        "GSE197628_ranked_GSEA_NES_heatmap.png"
    )

        plt.savefig(
        gsea_heatmap_file,
        dpi=300,
        bbox_inches="tight"
    )

        plt.close()

        print("Saved GSEA heatmap:", gsea_heatmap_file)


if de_results:
    de_all = pd.concat(de_results, ignore_index=True)

    de_all.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "GSE197628_pseudobulk_DE_all_products.tsv"
        ),
        sep="\t",
        index=False
    )


