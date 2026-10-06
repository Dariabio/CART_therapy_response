import os
import tarfile
import scanpy as sc
import numpy as np
import pandas as pd
import GEOparse
from scipy.io import mmread
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve


input_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/scRNAseq/GSE151511"
)

raw_dir = os.path.join(
    input_dir,
    "GSE151511_RAW"
)

output_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/GSE151511"
)

os.makedirs(
    output_dir,
    exist_ok=True
)

#LOAD metadata
def info_parser(soft_file):
   
    gse = GEOparse.get_GEO(filepath=soft_file)

    all_clin = {}
    for gsm_name, gsm in gse.gsms.items():
        gsm_clin = {}
        for key, values in gsm.metadata.items():
            if key.startswith("characteristics_ch"):
                for item in values:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        gsm_clin[k.strip()] = v.strip()
                    else:
                        gsm_clin[key] = item

      
            elif key in ["title"]:
                gsm_clin[key] = values[0] if isinstance(values, list) else values

        all_clin[gsm_name] = gsm_clin


    clin_df = pd.DataFrame.from_dict(all_clin, orient="index").reset_index().rename(columns={"index": "sample_id"})

    return clin_df

metadata = info_parser(
    os.path.join(input_dir, "GSE151511_family.soft.gz"))

metadata = metadata.rename(
    columns={
        "3mo pet/ct": "response"
    }
)

metadata["sample_id"] = (
    metadata["sample_id"]
    .astype(str)
    .str.strip()
)

metadata["response"] = (
    metadata["response"]
    .astype(str)
    .str.strip()
)

#READ FILES

extract_dir = os.path.join(
    raw_dir,
    "extracted"
)

os.makedirs(
    extract_dir,
    exist_ok=True
)


tar_path = (
    "/Users/darakuzmina/Desktop/CAR_T/data/scRNAseq/"
    "GSE151511/GSE151511_RAW/"
    "GSM4579891_ac01_bc_matrix.tar.gz"
)

with tarfile.open(tar_path, "r:gz") as tar:
    print("Files inside archive:")
    for member in tar.getmembers():
        print(member.name)

# ============================================================
# READ ONE 10X ARCHIVE
# ============================================================

def read_patient_tar(
    tar_path,
    extract_dir,
    metadata_row
):

    filename = os.path.basename(
        tar_path
    )

    # --------------------------------------------------------
    # GSM ID
    # --------------------------------------------------------

    gsm_id = filename.split("_")[0]

    # --------------------------------------------------------
    # Patient / sample ID
    # --------------------------------------------------------

    title = metadata_row["title"]

    # e.g. scRNA-seq_ac01
    patient_id = title.replace(
        "scRNA-seq_",
        ""
    )

    # --------------------------------------------------------
    # Extraction directory
    # --------------------------------------------------------

    patient_dir = os.path.join(
        extract_dir,
        gsm_id
    )

    os.makedirs(
        patient_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Extract archive
    # --------------------------------------------------------

    with tarfile.open(
        tar_path,
        "r:gz"
    ) as tar:

        tar.extractall(
            patient_dir
        )

    # --------------------------------------------------------
    # Find matrix directory
    # --------------------------------------------------------

    matrix_dir = None

    for root, dirs, files in os.walk(
        patient_dir
    ):

        files_set = set(files)

        has_matrix = any(
            f.endswith("_matrix.mtx")
            for f in files_set
        )

        has_barcodes = any(
            f.endswith("_barcodes.tsv")
            for f in files_set
        )

        has_genes = any(
            f.endswith("_genes.tsv")
            for f in files_set
        )

        if (
            has_matrix
            and has_barcodes
            and has_genes
        ):

            matrix_dir = root
            break

    if matrix_dir is None:

        raise FileNotFoundError(
            f"10x files not found in {tar_path}"
        )

    print(
        "Matrix directory:",
        matrix_dir
    )

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    matrix_file = next(
        f for f in os.listdir(matrix_dir)
        if f.endswith("_matrix.mtx")
    )

    barcode_file = next(
        f for f in os.listdir(matrix_dir)
        if f.endswith("_barcodes.tsv")
    )

    gene_file = next(
        f for f in os.listdir(matrix_dir)
        if f.endswith("_genes.tsv")
    )

    matrix_path = os.path.join(
        matrix_dir,
        matrix_file
    )

    barcode_path = os.path.join(
        matrix_dir,
        barcode_file
    )

    gene_path = os.path.join(
        matrix_dir,
        gene_file
    )

    # --------------------------------------------------------
    # Read matrix
    # --------------------------------------------------------

    matrix = mmread(
        matrix_path
    ).tocsr()

    # 10x matrix is genes × cells
    # AnnData requires cells × genes

    matrix = matrix.T.tocsr()

    # --------------------------------------------------------
    # Read barcodes
    # --------------------------------------------------------

    barcodes = pd.read_csv(
        barcode_path,
        sep="\t",
        header=None
    )[0].astype(str)

    # --------------------------------------------------------
    # Read genes
    # --------------------------------------------------------

    genes = pd.read_csv(
        gene_path,
        sep="\t",
        header=None
    )

    gene_ids = (
        genes.iloc[:, 0]
        .astype(str)
    )

    gene_symbols = (
        genes.iloc[:, 1]
        .astype(str)
    )

    # --------------------------------------------------------
    # Create AnnData
    # --------------------------------------------------------

    adata = sc.AnnData(
        X=matrix
    )

    adata.obs_names = barcodes

    adata.var_names = gene_ids

    adata.var["gene_symbol"] = (
        gene_symbols.values
    )

    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    adata.obs["sample_id"] = gsm_id

    adata.obs["patient_id"] = patient_id

    adata.obs["response"] = (
        metadata_row["response"]
    )

    adata.obs["histology"] = (
        metadata_row["histology"]
    )

    adata.obs["cell_type"] = (
        metadata_row["cell type"]
    )

    adata.obs["title"] = (
        metadata_row["title"]
    )

    return adata


# ============================================================
# FIND ARCHIVES
# ============================================================

tar_files = sorted(
    f
    for f in os.listdir(raw_dir)
    if f.endswith(".tar.gz")
)

print()
print(
    "Number of archives:",
    len(tar_files)
)


# ============================================================
# READ ALL PATIENTS
# ============================================================

datasets = []

summary = []


for i, filename in enumerate(
    tar_files,
    start=1
):

    print()
    print("=" * 70)
    print(
        f"{i}/{len(tar_files)}"
    )
    print(filename)
    print("=" * 70)

    tar_path = os.path.join(
        raw_dir,
        filename
    )

    # --------------------------------------------------------
    # GSM ID from filename
    # --------------------------------------------------------

    gsm_id = filename.split("_")[0]

    # --------------------------------------------------------
    # Find metadata
    # --------------------------------------------------------

    metadata_match = metadata[
        metadata["sample_id"] == gsm_id
    ]

    if metadata_match.empty:

        print(
            "WARNING: metadata not found:",
            gsm_id
        )

        continue

    if len(metadata_match) > 1:

        raise ValueError(
            f"Multiple metadata rows for {gsm_id}"
        )

    metadata_row = (
        metadata_match.iloc[0]
    )

    print(
        "Response:",
        metadata_row["response"]
    )

    print(
        "Histology:",
        metadata_row["histology"]
    )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    adata_patient = read_patient_tar(
        tar_path,
        extract_dir,
        metadata_row
    )

    print(
        "Shape:",
        adata_patient.shape
    )

    # --------------------------------------------------------
    # Save individual patient
    # --------------------------------------------------------

    patient_id = (
        adata_patient
        .obs["patient_id"]
        .iloc[0]
    )

    output_file = os.path.join(
        output_dir,
        f"{patient_id}.h5ad"
    )

    adata_patient.write(
        output_file
    )

    print(
        "Saved:",
        output_file
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary.append({
        "sample_id": gsm_id,
        "patient_id": patient_id,
        "response": metadata_row["response"],
        "histology": metadata_row["histology"],
        "cell_type": metadata_row["cell type"],
        "n_cells": adata_patient.n_obs,
        "n_genes": adata_patient.n_vars
    })

    datasets.append(
        adata_patient
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    summary
)

summary_file = os.path.join(
    output_dir,
    "GSE151511_samples.tsv"
)

summary.to_csv(
    summary_file,
    sep="\t",
    index=False
)

'''
signature = pd.read_csv(
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/13_signature/CAR_T_response_signature.tsv",
    sep="\t"
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
'''

signature = pd.read_csv(
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/15_signature_validation/"
    "signature_elastic_net.tsv",
    sep="\t"
)

signature_genes = (
    signature["gene"]
    .astype(str)
    .str.strip()
    .tolist()
)


# remove duplicates while preserving order
signature_genes = list(
    dict.fromkeys(
        signature_genes
    )
)
# 4. Process patients
# ============================================================

patient_expression = []

analysis_summary = summary[
    summary["response"].isin(["CR", "PD"])
].copy()

print(
    analysis_summary["response"].value_counts()
)

for _, row in analysis_summary.iterrows():

    patient_id = row["patient_id"]

    h5ad_file = os.path.join(
        output_dir,
        f"{patient_id}.h5ad"
    )

    print(
        f"\nProcessing {patient_id}"
    )

    adata = sc.read_h5ad(
        h5ad_file
    )

    # --------------------------------------------------------
    # Gene symbols
    # --------------------------------------------------------

    gene_symbols = (
        adata.var["gene_symbol"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Find signature genes
    # --------------------------------------------------------

    available_genes = [
        gene
        for gene in signature_genes
        if gene in set(gene_symbols)
    ]

    missing_genes = [
        gene
        for gene in signature_genes
        if gene not in set(gene_symbols)
    ]

    print(
        "Signature genes found:",
        len(available_genes),
        "/",
        len(signature_genes)
    )

    if missing_genes:
        print(
            "Missing:",
            missing_genes
        )

    # --------------------------------------------------------
    # Extract expression
    # --------------------------------------------------------

    # --------------------------------------------------------
# Extract counts for signature genes
# --------------------------------------------------------

    gene_symbols_array = (
    adata.var["gene_symbol"]
    .astype(str)
    .str.strip()
    .values
)

    signature_mask = np.isin(
    gene_symbols_array,
    available_genes
)

    X = adata.X[:, signature_mask]

    if not isinstance(X, np.ndarray):
        X = X.toarray()

    X = np.asarray(
    X,
    dtype=float
)

    selected_symbols = gene_symbols_array[
    signature_mask
]

# --------------------------------------------------------
# Sum duplicate gene symbols
# --------------------------------------------------------

    X_df = pd.DataFrame(
    X,
    columns=selected_symbols
)

    X_df = X_df.T.groupby(
    level=0
).sum().T

# Keep signature order
    X_df = X_df[
    available_genes
]

    X = X_df.to_numpy()

    # --------------------------------------------------------
    # Library-size normalization
    # --------------------------------------------------------

    library_size = np.asarray(
        adata.X.sum(axis=1)
    ).flatten()

    X = (
        X
        / library_size[:, None]
        * 1e4
    )

    X = np.log1p(X)

    # --------------------------------------------------------
    # Mean expression across cells
    # --------------------------------------------------------

    mean_expression = X.mean(axis=0)

    patient_row = pd.Series(
        mean_expression,
        index=available_genes,
        name=patient_id
    )

    patient_expression.append(
        patient_row
    )


# ============================================================
# 5. Combine patients
# ============================================================

patient_expression = pd.DataFrame(
    patient_expression
)

patient_expression_file = os.path.join(
    output_dir,
    "GSE151511_patient_expression.tsv"
)

patient_expression.to_csv(
    patient_expression_file,
    sep="\t"
)

print(
    "\nPatient expression matrix:",
    patient_expression.shape
)

print(
    patient_expression.head()
)

print(
    adata.var["gene_symbol"]
    .duplicated()
    .sum()
)

print(
    patient_expression.shape
)

print(
    set(signature_genes)
    - set(patient_expression.columns)
)

print(
    analysis_summary[
        ["patient_id", "response"]
    ].sort_values("patient_id")
)

print(
    "\nCounts:"
)

print(
    analysis_summary["response"].value_counts()
)

print(patient_expression.shape)
print(patient_expression.columns.tolist())


gene_z = (
    patient_expression
    - patient_expression.mean(axis=0)
) / patient_expression.std(axis=0, ddof=1)

signature_score = gene_z.mean(axis=1)

scores = pd.DataFrame({
    "patient_id": patient_expression.index,
    "signature_score": signature_score.values
})

scores = scores.merge(
    analysis_summary[
        ["patient_id", "response"]
    ],
    on="patient_id",
    how="left"
)

scores_file = os.path.join(
    output_dir,
    "GSE151511_signature_scores.tsv"
)

scores.to_csv(
    scores_file,
    sep="\t",
    index=False
)

print(
    scores.groupby("response")["signature_score"]
    .agg(["count", "median", "mean", "std"])
)

from scipy.stats import mannwhitneyu

cr = scores.loc[
    scores["response"] == "CR",
    "signature_score"
]

pd_ = scores.loc[
    scores["response"] == "PD",
    "signature_score"
]

u_stat, p_value = mannwhitneyu(
    cr,
    pd_,
    alternative="two-sided"
)

print("U =", u_stat)
print("p =", p_value)

from sklearn.metrics import roc_auc_score

y_true = (
    scores["response"] == "CR"
).astype(int)

auc = roc_auc_score(
    y_true,
    scores["signature_score"]
)

print("AUC =", auc)

stats = pd.DataFrame([{
    "cohort": "GSE151511",
    "n_total": len(scores),
    "n_CR": len(cr),
    "n_PD": len(pd_),
    "CR_median": cr.median(),
    "PD_median": pd_.median(),
    "CR_mean": cr.mean(),
    "PD_mean": pd_.mean(),
    "MWU_U": u_stat,
    "MWU_p": p_value,
    "AUC_CR": auc
}])

stats_file = os.path.join(
    output_dir,
    "GSE151511_signature_statistics.tsv"
)

stats.to_csv(
    stats_file,
    sep="\t",
    index=False
)

print(
    "Saved:",
    stats_file
)

print("\nStatistics:")
print(stats)

fpr, tpr, thresholds = roc_curve(
    y_true,
    scores["signature_score"]
)

plt.figure(
    figsize=(6, 6)
)

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "False positive rate"
)

plt.ylabel(
    "True positive rate"
)

plt.title(
    "GSE151511: DE-derived response signature"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

roc_file = os.path.join(
    output_dir,
    "GSE151511_ROC.png"
)

plt.savefig(
    roc_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    roc_file
)

plt.figure(
    figsize=(6, 5)
)

groups = ["CR", "PD"]

data = [
    scores.loc[
        scores["response"] == group,
        "signature_score"
    ]
    for group in groups
]

plt.boxplot(
    data,
    labels=groups,
    showfliers=False
)

# individual patients
for i, group in enumerate(groups, start=1):

    values = scores.loc[
        scores["response"] == group,
        "signature_score"
    ]

    x = np.random.normal(
        i,
        0.04,
        size=len(values)
    )

    plt.scatter(
        x,
        values,
        alpha=0.8
    )

plt.ylabel(
    "Signature score"
)

plt.title(
    "GSE151511: DE-derived response signature"
)

plt.tight_layout()

score_plot_file = os.path.join(
    output_dir,
    "GSE151511_signature_scores.png"
)

plt.savefig(
    score_plot_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    score_plot_file
)

