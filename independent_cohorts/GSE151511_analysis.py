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
import gseapy as gp
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
import seaborn as sns

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


# READ ONE 10X ARCHIVE


def read_patient_tar(
    tar_path,
    extract_dir,
    metadata_row
):

    filename = os.path.basename(
        tar_path
    )

   
    # GSM ID
    

    gsm_id = filename.split("_")[0]

   
    # Patient / sample ID


    title = metadata_row["title"]

    # e.g. scRNA-seq_ac01
    patient_id = title.replace(
        "scRNA-seq_",
        ""
    )

    
    # Extraction directory


    patient_dir = os.path.join(
        extract_dir,
        gsm_id
    )

    os.makedirs(
        patient_dir,
        exist_ok=True
    )

  
    # Extract archive
    

    with tarfile.open(
        tar_path,
        "r:gz"
    ) as tar:

        tar.extractall(
            patient_dir
        )

    
    # Find matrix directory
    

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

    
    # Find files
    

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

    
    # Read matrix
    

    matrix = mmread(
        matrix_path
    ).tocsr()

    # 10x matrix is genes × cells
    # AnnData requires cells × genes

    matrix = matrix.T.tocsr()

    
    # Read barcodes
    

    barcodes = pd.read_csv(
        barcode_path,
        sep="\t",
        header=None
    )[0].astype(str)

    
    # Read genes
    

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

    
    # Create AnnData
    

    adata = sc.AnnData(
        X=matrix
    )

    adata.obs_names = barcodes

    adata.var_names = gene_ids

    adata.var["gene_symbol"] = (
        gene_symbols.values
    )

    
    # Add metadata
    

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



# FIND ARCHIVES


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



# READ ALL PATIENTS


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

    
    # GSM ID from filename
    

    gsm_id = filename.split("_")[0]

    
    # Find metadata
    

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

    
    # Read
    

    adata_patient = read_patient_tar(
        tar_path,
        extract_dir,
        metadata_row
    )

    print(
        "Shape:",
        adata_patient.shape
    )

    
    # Save individual patient
    

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

    
    # Summary
    

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



# SUMMARY TABLE


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


# remove duplicates while preserving order
signature_genes = list(
    dict.fromkeys(
        signature_genes
    )
)
# 4. Process patients


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

    
    # Gene symbols
    

    gene_symbols = (
        adata.var["gene_symbol"]
        .astype(str)
        .str.strip()
    )

    
    # Find signature genes
    

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

    
    # Extract expression
    

    
# Extract counts for signature genes


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


# Sum duplicate gene symbols


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

    
    # Library-size normalization
    

    library_size = np.asarray(
        adata.X.sum(axis=1)
    ).flatten()

    X = (
        X
        / library_size[:, None]
        * 1e4
    )

    X = np.log1p(X)

    
    # Mean expression across cells
    

    mean_expression = X.mean(axis=0)

    patient_row = pd.Series(
        mean_expression,
        index=available_genes,
        name=patient_id
    )

    patient_expression.append(
        patient_row
    )



# 5. Combine patients


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


# 1. HEATMAP: SIGNATURE GENES ACROSS PATIENTS

# Use the expression matrix created above
heatmap_data = patient_expression.copy()

# Keep only genes that are present
heatmap_genes = [
    gene for gene in signature_genes
    if gene in heatmap_data.columns
]

heatmap_data = heatmap_data[heatmap_genes]

# Z-score each gene across patients
heatmap_z = (
    heatmap_data - heatmap_data.mean(axis=0)
) / heatmap_data.std(axis=0, ddof=1)

heatmap_z = heatmap_z.replace(
    [np.inf, -np.inf],
    np.nan
).fillna(0)

# Response annotation
response_annotation = (
    analysis_summary
    .drop_duplicates("patient_id")
    .set_index("patient_id")
    .reindex(heatmap_z.index)["response"]
)

sample_order = (
    response_annotation
    .sort_values(key=lambda x: x.map({"PD": 0, "CR": 1}))
    .index
)

# Reorder expression matrix and response annotation
heatmap_data = heatmap_data.loc[sample_order]
heatmap_z = heatmap_z.loc[sample_order]
response_annotation = response_annotation.loc[sample_order]

response_palette = {
    "PD": "#1F77B4",
    "CR": "#D62728"
}

col_colors = pd.DataFrame({
    "Response": response_annotation.map(response_palette)
}, index=heatmap_z.index)


from matplotlib.patches import Patch
# Cluster genes, preserve patient order
g = sns.clustermap(
    heatmap_z.T,
    cmap="vlag",
    center=0,
    col_colors=col_colors,
    col_cluster=False,
    row_cluster=True,
    xticklabels=True,
    yticklabels=True,
    figsize=(14, 10),
    cbar_kws={"label": "Z-score"}
)

legend_elements = [
    Patch(
        facecolor=color,
        label=label
    )
    for label, color
    in response_palette.items()
]

g.ax_heatmap.set_title(
    "GSE151511: response signature expression",
    pad=20,
    y=1.03
)

# Legend: move it to the right of the heatmap
g.ax_heatmap.legend(
    handles=legend_elements,
    title="Response",
    loc="upper left",
    bbox_to_anchor=(1.15, 1.0),
    frameon=False,
    borderaxespad=0
)

g.ax_heatmap.set_xlabel("Patients")
g.ax_heatmap.set_ylabel("Signature genes")


heatmap_file = os.path.join(
    output_dir,
    "GSE151511_signature_gene_heatmap.png"
)

g.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")

print("Saved:", heatmap_file)

# 2. RANKED GSEA: GENE PANELS



panel_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/raw"
)

# Files are read in the requested order
panel_files = [
    ("P1", "gene_panel_1.csv"),
    ("P2", "gene_panel_5.csv"),
    ("P3", "gene_panel_3.csv"),
    ("P4", "gene_panel_4.csv")
]

panels = {}

for prefix, filename in panel_files:

    panel_path = os.path.join(
        panel_dir,
        filename
    )

    if not os.path.exists(panel_path):
        print("WARNING: panel file not found:", panel_path)
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

        # Remove duplicate genes while preserving order
        panels[panel_name] = list(
            dict.fromkeys(genes)
        )

print("\nGene panels loaded:")
for panel_name, genes in panels.items():
    print(panel_name, len(genes), "genes")



# 3. LOAD SAMPLE METADATA FOR PSEUDOBULK


gsea_metadata = summary[
    summary["response"].isin(["CR", "PD"])
].copy()

gsea_metadata["cell_type"] = (
    gsea_metadata["cell_type"]
    .astype(str)
    .str.strip()
)

gsea_metadata["response"] = (
    gsea_metadata["response"]
    .astype(str)
    .str.strip()
)

print("\nCell types found in metadata:")
print(
    summary["cell_type"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nUnique cell types (repr):")
for ct in summary["cell_type"].dropna().unique():
    print(repr(ct))


cell_types = [
    "CART cells"
]

gsea_results = []
de_results = []



# 4. PSEUDOBULK DE AND PRERANK GSEA


for cell_type in cell_types:

    print("\n" + "=" * 70)
    print("Cell type:", cell_type)
    print("=" * 70)

    subset = gsea_metadata[
        gsea_metadata["cell_type"] == cell_type
    ].copy()

    if subset.empty:
        print("No samples for", cell_type)
        continue

    
    # Aggregate raw counts per patient
    

    pseudobulk_counts = []
    pseudobulk_metadata = []

    for patient_id, patient_rows in subset.groupby("patient_id"):

        responses = patient_rows["response"].unique()

        if len(responses) != 1:
            print(
                "WARNING: inconsistent response for patient",
                patient_id
            )
            continue

        patient_counts = []

        for _, row in patient_rows.iterrows():

            sample_id = row["sample_id"]

            h5ad_file = os.path.join(
                output_dir,
                f"{patient_id}.h5ad"
            )

            if not os.path.exists(h5ad_file):
                print("Missing h5ad:", h5ad_file)
                continue

            adata_sample = sc.read_h5ad(h5ad_file)

            # Keep raw counts from the matching cell type
            # and sample, to avoid mixing different cell types.
            cell_mask = (
                adata_sample.obs["cell_type"].astype(str).values
                == cell_type
            )

            if "sample_id" in adata_sample.obs.columns:
                cell_mask &= (
                    adata_sample.obs["sample_id"].astype(str).values
                    == sample_id
                )

            if not np.any(cell_mask):
                continue

            X = adata_sample.X[cell_mask, :]

            if not isinstance(X, np.ndarray):
                X = X.toarray()

            X = np.asarray(X, dtype=np.int64)

            # Sum counts across cells
            counts = X.sum(axis=0)

            gene_symbols = (
                adata_sample.var["gene_symbol"]
                .astype(str)
                .str.strip()
                .values
            )

            counts_df = pd.DataFrame({
                "gene": gene_symbols,
                "count": counts
            })

            counts_df = counts_df[
                counts_df["gene"].notna()
                & (counts_df["gene"] != "")
                & (counts_df["gene"] != "nan")
            ]

            # Aggregate duplicated gene symbols
            counts_df = counts_df.groupby(
                "gene",
                as_index=True
            )["count"].sum()

            patient_counts.append(counts_df)

        if not patient_counts:
            continue

        # If several GSM samples exist for one patient,
        # sum their counts by gene.
        combined_counts = pd.concat(
            patient_counts,
            axis=1
        ).fillna(0).sum(axis=1)

        pseudobulk_counts.append(
            combined_counts.rename(patient_id)
        )

        pseudobulk_metadata.append({
            "patient_id": patient_id,
            "response": responses[0]
        })

    if len(pseudobulk_counts) < 2:
        print("Not enough pseudobulk samples.")
        continue

    counts_df = pd.concat(
        pseudobulk_counts,
        axis=1
    ).fillna(0)

    counts_df = counts_df.round().astype(int)

    meta_df = pd.DataFrame(
        pseudobulk_metadata
    ).drop_duplicates("patient_id")

    meta_df = meta_df.set_index("patient_id")
    meta_df = meta_df.reindex(counts_df.columns)

    # Drop genes with zero total counts
    counts_df = counts_df.loc[
        counts_df.sum(axis=1) > 0
    ]

    print("Pseudobulk matrix:", counts_df.shape)
    print("Responses:")
    print(meta_df["response"].value_counts())

    
    # Check sample numbers
    

    if (
        (meta_df["response"] == "CR").sum() < 2
        or (meta_df["response"] == "PD").sum() < 2
    ):
        print(
            "WARNING: fewer than 2 patients in one response group; "
            "DESeq2/GSEA may be unreliable."
        )

    
    # DESeq2
    

    counts_for_deseq = counts_df.T.copy()
    counts_for_deseq.index = counts_for_deseq.index.astype(str)

    meta_for_deseq = meta_df.copy()
    meta_for_deseq.index = meta_for_deseq.index.astype(str)

    meta_for_deseq["response"] = pd.Categorical(
        meta_for_deseq["response"],
        categories=["PD", "CR"]
    )

    inference = DefaultInference(
        n_cpus=4
    )

    dds = DeseqDataSet(
        counts=counts_for_deseq,
        metadata=meta_for_deseq,
        design="~response",
        refit_cooks=True,
        inference=inference
    )

    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=["response", "CR", "PD"],
        inference=inference
    )

    stat_res.summary()

    de = stat_res.results_df.copy()
    de.index.name = "gene"
    de = de.reset_index()

    de["cell_type"] = cell_type

    de_file = os.path.join(
        output_dir,
        f"GSE151511_pseudobulk_DE_{cell_type.replace('+', 'plus').replace(' ', '_')}.tsv"
    )

    de.to_csv(
        de_file,
        sep="\t",
        index=False
    )

    print("Saved DE results:", de_file)

    de_results.append(de)

    
    # Create ranking for preranked GSEA
    # Wald statistic ranks genes by evidence and direction.
    # Positive = higher in CR; negative = higher in PD.
    

    ranking = de[
        ["gene", "stat"]
    ].dropna()

    ranking = ranking[
        np.isfinite(ranking["stat"])
    ]

    ranking = ranking.drop_duplicates(
        subset="gene"
    )

    ranking = ranking.sort_values(
        "stat",
        ascending=False
    )

    ranking_file = os.path.join(
        output_dir,
        f"GSE151511_GSEA_ranking_{cell_type.replace('+', 'plus').replace(' ', '_')}.tsv"
    )

    ranking.to_csv(
        ranking_file,
        sep="\t",
        index=False
    )

    print("Saved ranking:", ranking_file)

    
    # Run preranked GSEA for every panel
    

    for panel_name, panel_genes in panels.items():

        overlap = set(panel_genes) & set(ranking["gene"])

        print(
            panel_name,
            "- genes in ranking:",
            len(overlap)
        )

        if len(overlap) < 3:
            print("Skipping panel with fewer than 3 overlapping genes.")
            continue

        try:
            pre_res = gp.prerank(
                rnk=ranking[["gene", "stat"]],
                gene_sets={panel_name: panel_genes},
                min_size=3,
                max_size=500,
                permutation_num=1000,
                seed=42,
                outdir=None,
                verbose=False
            )

            res = pre_res.res2d.copy()

            if res.empty:
                continue

            res["panel"] = panel_name
            res["cell_type"] = cell_type

            gsea_results.append(res)

        except Exception as e:
            print(
                "GSEA failed for",
                panel_name,
                cell_type,
                ":",
                e
            )



# 5. SAVE GSEA RESULTS

if gsea_results:

    gsea_all = pd.concat(
        gsea_results,
        ignore_index=True
    )

    gsea_file = os.path.join(
        output_dir,
        "GSE151511_ranked_GSEA_all_panels.tsv"
    )

    gsea_all.to_csv(
        gsea_file,
        sep="\t",
        index=False
    )

    print("\nSaved GSEA results:", gsea_file)

    # Normalize GSEApy output column names
    col_lookup = {
    str(col).strip().lower(): col
    for col in gsea_all.columns
}

    nes_col = col_lookup.get("nes")
    fdr_col = col_lookup.get("fdr q-val")


    if nes_col is None:
        raise KeyError(
            "NES column not found in GSEA results: "
            + str(gsea_all.columns.tolist())
        )

    if fdr_col is None:
        raise KeyError(
            "FDR column not found in GSEA results: "
            + str(gsea_all.columns.tolist())
        )

    
    # Combined NES matrix
    # Rows = panels, columns = cell types
    

    gsea_all["NES"] = pd.to_numeric(
        gsea_all[nes_col],
        errors="coerce"
    )

    gsea_all["FDR q-val"] = pd.to_numeric(
        gsea_all[fdr_col],
        errors="coerce"
    )

    gsea_all["cell_type"] = (
        gsea_all["cell_type"].astype(str)
    )

    nes_matrix = gsea_all.pivot_table(
        index="panel",
        columns="cell_type",
        values="NES",
        aggfunc="first"
    )

    fdr_matrix = gsea_all.pivot_table(
        index="panel",
        columns="cell_type",
        values="FDR q-val",
        aggfunc="first"
    )

    
    

    # Keep only selected panels in the exact requested order

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

# Warn if requested signatures are absent from GSEA results
    missing_panels = [
    p for p in selected_panels
    if p not in nes_matrix.index
]

    if missing_panels:
        print("WARNING: these panels are absent from GSEA results:")
        print(missing_panels)

# Keep only available signatures, preserving the specified order
    panel_order = [
    p for p in selected_panels
    if p in nes_matrix.index
]

    nes_matrix = nes_matrix.reindex(index=panel_order)

    desired_cell_order = [
    "CART cells"
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


    
    # Significance stars
    # *  FDR < 0.05 and |NES| > 1
    # ** FDR < 0.01 and |NES| > 1
    

    annot = pd.DataFrame(
        "",
        index=nes_matrix.index,
        columns=nes_matrix.columns
    )

    for panel_name in nes_matrix.index:
        for ct in nes_matrix.columns:

            nes = nes_matrix.loc[panel_name, ct]
            fdr = fdr_matrix.loc[panel_name, ct]

            if pd.isna(nes) or pd.isna(fdr):
                continue

            if abs(nes) > 1 and fdr < 0.05:
                annot.loc[panel_name, ct] = "**"

            elif abs(nes) > 1 and fdr < 0.1:
                annot.loc[panel_name, ct] = f"*"


    # Symmetric color scale centered at zero

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
        figsize=(
            7,
            max(4, len(nes_matrix) * 0.45)
        )
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

        plt.xlabel("")
        plt.ylabel("Gene panels")
        plt.title(
        "GSE151511: ranked GSEA by CAR T-cell subtype"
    )

        plt.tight_layout()

        gsea_heatmap_file = os.path.join(
        output_dir,
        "GSE151511_ranked_GSEA_NES_heatmap.png"
    )

        plt.savefig(
        gsea_heatmap_file,
        dpi=300,
        bbox_inches="tight"
    )

        plt.close()
