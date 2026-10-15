#%%
import os
import numpy as np
import pandas as pd
import GEOparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)
from matplotlib.patches import Patch
import gseapy as gp
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

from scipy.stats import mannwhitneyu
#%%

input_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/GSE223655"
)

annotation_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/annotation/Homo_sapiens.gene_info.gz"
)

output_dir = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/GSE223655"
)

panel_dir = ("/Users/darakuzmina/Desktop/CAR_T/"
             "data/raw")

os.makedirs(
    output_dir,
    exist_ok=True
)
#%%


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

info_223655  = info_parser(
    os.path.join(input_dir, "GSE223655_family.soft.gz"))

expression_GSE223655 = pd.read_csv(os.path.join(input_dir, "GSE223655_raw_counts_GRCh38.p13_NCBI.tsv.gz"), sep="\t")


info_223655 = info_223655[
    info_223655["cell subtype"].isin([
        "CD8+CAR T cells",
        "CD4+CAR T cells"
    ])
].copy()




def annotate_gene_ids_ncbi(expression, info_file_path="Homo_sapiens.gene_info.gz", id_col = None):

    df = expression.copy()
    

    ncbi_df = pd.read_csv(
        info_file_path, 
        compression='gzip', 
        sep='\t', 
        usecols=['GeneID', 'Symbol']
    )
    
    # dict for mapping
    entrez_to_symbol = dict(zip(ncbi_df['GeneID'].astype(str), ncbi_df['Symbol']))
    
    # column of Entrez ID
    if id_col in df.columns:
        raw_ids = df[id_col]
    else:
        raw_ids = df.index.to_series()
        
    print("🧬 Mapping of Entrez ID to Gene Symbol")
    # Remove the fractional part of the identifier (.0), which often breaks the match
    query_ids = pd.to_numeric(raw_ids, errors='coerce').dropna().astype('Int64').astype(str)
    
    df['symbol'] = query_ids.map(entrez_to_symbol)
    df = df.drop(id_col, axis=1)
    df = df.set_index("symbol")
    df = df.reset_index()
    
        
    return df

expression_GSE223655 = annotate_gene_ids_ncbi(expression_GSE223655, 
info_file_path = annotation_dir, id_col='GeneID')


def filter_expression(expression_df, clinical_df, col_id, output_path, file_id='default'):
    # 1. Index
    clinical_df = clinical_df.set_index(col_id)
    
    #first column of genes is index
    first_col_name = expression_df.columns[0]

    expression_df = expression_df.set_index(first_col_name)


    # group by gene (index) to remove dublicates
    # genes in column
    expression_df = expression_df.groupby(level=0).mean(numeric_only=True)
    expression_df = expression_df.astype("int64")

    # new name
    expression_df.index.name = None

    # 2. Common samples
    common_samples = clinical_df.index.intersection(expression_df.columns)
    
    # 3. Filter by  common_samples
    # order
    clinical_df = clinical_df.reindex(common_samples)
    expression_df = expression_df.reindex(columns=common_samples)

    # 4. Check
    if not (clinical_df.index == expression_df.columns).all():
        print("Error")
    else:
        print(f"{len(common_samples)} samples")
    
    # index
    clinical_df = clinical_df.reset_index().rename(columns={'index': 'sample_id'})
    #expression_df = expression_df.reset_index()

    
    clinical_df.to_csv(os.path.join(output_path, f'info_{file_id}_pr.csv'))
    expression_df.to_csv(os.path.join(output_path, f'expression_{file_id}_pr.csv'))

    return expression_df, clinical_df


expression_GSE223655, info_223655 = filter_expression( 
    expression_GSE223655, info_223655, col_id = "sample_id", output_path = output_dir, file_id='GSE223655')

def norm_func_vst(expression_df, clinical_df, output_path = None, file_id="default"):

    clinical_df = clinical_df.copy()
    expression_df = expression_df.copy()

    clinical_df = clinical_df.set_index("sample_id")
    clinical_df = clinical_df.rename(
    columns={"disease state": "disease_state"}
)

    
    counts = expression_df.copy()[(expression_df > 0).any(axis=1)].T # samples & genes
    #dds = DeseqDataSet(counts=counts, metadata=clinical_df, design_factors='disease_state', refit_cooks=True)
    dds = DeseqDataSet(
    counts=counts,
    metadata=clinical_df,
    design="~1",
    refit_cooks=True,
    inference=DefaultInference()
)
    dds.deseq2()

    # --- ПОЛУЧЕНИЕ VST ДАННЫХ ---
    dds.vst() # Трансформация
    vst_counts = pd.DataFrame(dds.layers['vst_counts'], index=dds.obs_names, columns=dds.var_names)
    # Сохранение (теперь сохраняем VST)
    vst_counts.to_csv(os.path.join(output_path, f'{file_id}_vst_counts.csv'))
   
    return vst_counts

expression_GSE223655_vst = norm_func_vst(expression_GSE223655, info_223655, output_path = output_dir, file_id='GSE223655')

print(expression_GSE223655_vst.head())

pca = PCA(
    n_components=10
)

pcs = pca.fit_transform(
    expression_GSE223655_vst
)


pc_df = pd.DataFrame(
    pcs[:, :2],
    columns=["PC1", "PC2"],
    index=expression_GSE223655_vst.index
)

meta = info_223655.copy()
meta = meta.set_index("sample_id")

pc_df = pc_df.join(meta)


def make_pca(
    data,
    pca_object,
    hue,
    filename,
    title=None):

    plt.figure(
        figsize=(7, 6)
    )

    sns.scatterplot(
        data= data,
        x="PC1",
        y="PC2",
        hue=hue,
        palette="tab10",
        s=80
    )

    plt.xlabel(
        f"PC1 ({pca_object.explained_variance_ratio_[0]:.2%})"
    )

    plt.ylabel(
        f"PC2 ({pca_object.explained_variance_ratio_[1]:.2%})"
    )

    if title:
        plt.title(title)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            filename
        ),
        dpi=300
    )

    plt.close()


make_pca(
    pc_df,
    pca,
    "disease state",
    "PCA_response.png",
    "PCA: response"
)

make_pca(
    pc_df,
    pca,
    "cell subtype",
    "PCA_cells.png",
    "PCA: cells"
)

make_pca(
    pc_df,
    pca,
    "patient id",
    "PCA_patients.png",
    "PCA: patients"
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
'''

# remove duplicates while preserving order
signature_genes = list(
    dict.fromkeys(
        signature_genes
    )
)

available_genes = [
    gene
    for gene in signature_genes
    if gene in expression_GSE223655_vst.columns
]

missing_genes = [
    gene
    for gene in signature_genes
    if gene not in expression_GSE223655_vst.columns
]

print(
    f"Signature genes: {len(signature_genes)}"
)

print(
    f"Available: {len(available_genes)}"
)

print(
    f"Missing: {len(missing_genes)}"
)

print(
    "Missing genes:",
    missing_genes
)

#gene-wise Z-score
expr = expression_GSE223655_vst[
    available_genes
].copy()

gene_z = (
    expr
    - expr.mean(axis=0)
) / expr.std(axis=0, ddof=1)

signature_score = gene_z[
    available_genes
].mean(axis=1)

#table sample-level

score_df = info_223655[
    [
        "sample_id",
        "patient id",
        "disease state",
        "cell subtype"
    ]
].copy()

score_df["signature_score"] = (
    score_df["sample_id"]
    .map(signature_score)
)

#насколько signature score отличается между CD8 и CD4 внутри одного пациента?

wide = score_df.pivot(
    index="patient id",
    columns="cell subtype",
    values="signature_score"
)

wide["CD8_minus_CD4"] = (
    wide["CD8+CAR T cells"]
    -
    wide["CD4+CAR T cells"]
)

#add patient data
patient_response = (
    score_df[
        ["patient id", "disease state"]
    ]
    .drop_duplicates()
    .set_index("patient id")
)

wide = wide.join(patient_response)

wide.to_csv(os.path.join(output_dir, "scores_signature.csv"))

print(wide)
print(wide["disease state"].value_counts())


#separate cd8 and cd4
cd8 = wide[
    ["CD8+CAR T cells", "disease state"]
].dropna().copy()

cd8.columns = ["score", "response"]

print(cd8.groupby("response")["score"].describe())

cd4 = wide[
    ["CD4+CAR T cells", "disease state"]
].dropna().copy()

cd4.columns = ["score", "response"]

print(cd4.groupby("response")["score"].describe())

#mannwhitneyu


cd8_cr = cd8.loc[
    cd8["response"] == "CR patient",
    "score"
]

cd8_pd = cd8.loc[
    cd8["response"] == "PD patient",
    "score"
]

u_cd8, p_cd8 = mannwhitneyu(
    cd8_cr,
    cd8_pd,
    alternative="two-sided"
)

print("CD8")
print("CR:", len(cd8_cr))
print("PD:", len(cd8_pd))
print("U:", u_cd8)
print("p:", p_cd8)


cd4_cr = cd4.loc[
    cd4["response"] == "CR patient",
    "score"
]

cd4_pd = cd4.loc[
    cd4["response"] == "PD patient",
    "score"
]

u_cd4, p_cd4 = mannwhitneyu(
    cd4_cr,
    cd4_pd,
    alternative="two-sided"
)

print("CD4")
print("CR:", len(cd4_cr))
print("PD:", len(cd4_pd))
print("U:", u_cd4)
print("p:", p_cd4)


#Effect size — rank-biserial correlation
rank_biserial_cd8 = (
    2 * u_cd8
    / (len(cd8_cr) * len(cd8_pd))
    - 1
)

print(
    "CD8 rank-biserial:",
    rank_biserial_cd8
)

rank_biserial_cd4 = (
    2 * u_cd4
    / (len(cd4_cr) * len(cd4_pd))
    - 1
)

print(
    "CD4 rank-biserial:",
    rank_biserial_cd4
)

#save

# ============================================================
# ROC / AUC
# ============================================================

from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# CD8
# ------------------------------------------------------------

y_cd8 = (
    cd8["response"] == "CR patient"
).astype(int)

auc_cd8 = roc_auc_score(
    y_cd8,
    cd8["score"]
)

print(
    "\nCD8 AUC:",
    auc_cd8
)


# ------------------------------------------------------------
# CD4
# ------------------------------------------------------------

y_cd4 = (
    cd4["response"] == "CR patient"
).astype(int)

auc_cd4 = roc_auc_score(
    y_cd4,
    cd4["score"]
)

print(
    "CD4 AUC:",
    auc_cd4
)

# ============================================================
# FINAL STATISTICS
# MWU + rank-biserial + AUC
# ============================================================

results = []

results.append({
    "cell_subtype": "CD8+CAR T cells",
    "n_CR": len(cd8_cr),
    "n_PD": len(cd8_pd),
    "median_CR": cd8_cr.median(),
    "median_PD": cd8_pd.median(),
    "MWU_U": u_cd8,
    "MWU_p": p_cd8,
    "rank_biserial": rank_biserial_cd8,
    "AUC_CR": auc_cd8
})

results.append({
    "cell_subtype": "CD4+CAR T cells",
    "n_CR": len(cd4_cr),
    "n_PD": len(cd4_pd),
    "median_CR": cd4_cr.median(),
    "median_PD": cd4_pd.median(),
    "MWU_U": u_cd4,
    "MWU_p": p_cd4,
    "rank_biserial": rank_biserial_cd4,
    "AUC_CR": auc_cd4
})

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(
        output_dir,
        "GSE223655_signature_statistics.tsv"
    ),
    sep="\t",
    index=False
)

print("\nFinal statistics:")
print(results_df)

# ============================================================
# ROC CURVES
# ============================================================

fpr_cd8, tpr_cd8, thresholds_cd8 = roc_curve(
    y_cd8,
    cd8["score"]
)

fpr_cd4, tpr_cd4, thresholds_cd4 = roc_curve(
    y_cd4,
    cd4["score"]
)


plt.figure(
    figsize=(7, 7)
)

plt.plot(
    fpr_cd8,
    tpr_cd8,
    linewidth=2,
    label=f"CD8+ CAR T cells (AUC = {auc_cd8:.3f})"
)

plt.plot(
    fpr_cd4,
    tpr_cd4,
    linewidth=2,
    label=f"CD4+ CAR T cells (AUC = {auc_cd4:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC: CAR-T response signature"
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "ROC_signature_CD8_CD4.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# PATIENT-LEVEL PLOTS
# ============================================================

def patient_level_plot(
    data,
    title,
    filename
):

    plot_df = data.copy()

    plt.figure(
        figsize=(6, 6)
    )

    sns.boxplot(
        data=plot_df,
        x="response",
        y="score",
        order=["CR patient", "PD patient"],
        width=0.5,
        showcaps=True,
        showfliers=False
    )

    sns.stripplot(
        data=plot_df,
        x="response",
        y="score",
        order=["CR patient", "PD patient"],
        jitter=0.12,
        size=8,
        alpha=0.8
    )

    plt.xlabel(
        "Clinical response"
    )

    plt.ylabel(
        "Signature score"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            filename
        ),
        dpi=300
    )

    plt.close()


patient_level_plot(
    cd8,
    "CD8+ CAR T cells",
    "CD8_signature_patient_level.png"
)

patient_level_plot(
    cd4,
    "CD4+ CAR T cells",
    "CD4_signature_patient_level.png"
)

#Heatamap
vst_genes = set(
    expression_GSE223655_vst.columns
)

'''
available_genes = [
    gene
    for gene in signature_genes
    if gene in expression_GSE223655_vst.columns
]
'''


#%%
# ============================================================
# 6. DEG HEATMAP
# ============================================================

heatmap_data = (
    expression_GSE223655_vst[
        available_genes
    ]
    .T
)

# ------------------------------------------------------------
# Sort samples:
# nonresponder -> responder
# ------------------------------------------------------------

response_order = {
    "PD patient": 0,
    "CR patient": 1
}


meta["response_order"] = (
    meta["disease state"]
    .map(response_order)
)


sorted_samples = (
    meta
    .sort_values(
        "response_order"
    )
    .index
)


heatmap_data = heatmap_data[
    sorted_samples
]


# ------------------------------------------------------------
# Annotation colours
# ------------------------------------------------------------

response_palette = {
    "PD patient": "#1F77B4",
    "CR patient": "#D62728"
}

tcells_palette = {
    "CD8+CAR T cells": "#D37D0E",
    "CD4+CAR T cells": "#F7E084"
}


col_colors = pd.DataFrame({
    "disease state": meta.loc[sorted_samples, "disease state"].map(response_palette),
    "cell subtype": meta.loc[sorted_samples, "cell subtype"].map(tcells_palette)
})
#%%
# ============================================================
# HEATMAP
# ============================================================

g = sns.clustermap(
    heatmap_data,
    metric="correlation",
    method="average",
    z_score=0,
    cmap="vlag",
    col_colors=col_colors,
    col_cluster=False,
    row_cluster=True,
    figsize=(14, 10),
    xticklabels=True,
    yticklabels=True
)


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
    in tcells_palette.items()
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
    title="T-cells",
    bbox_to_anchor=(1.15, 0.78),
    loc="upper left",
    frameon=False
)



g.fig.suptitle(
    "Product DEGs: responder vs nonresponder "
    f"({len(available_genes)} genes)",
    y=1.02,
    fontsize=14
)

heatmap_file = os.path.join(
    output_dir,
    "product_DEG_GSE223655_heatmap.png"
)


g.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    g.fig
)

# %%
# ssGSEA
# model considering same patients
# ranked gsea

# 1. Build gene panels

panels = {}
# Panel 1
gp1 = pd.read_csv(
    os.path.join(
    panel_dir,
    "gene_panel_1.csv"
)
    )

for col in gp1.columns:

    genes = (
        gp1[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    genes = [
        g for g in genes
        if g != "nan"
    ]

    panels[
        f"P1_{col}"
    ] = genes

# Panel 2
gp2 = pd.read_csv(
    os.path.join(
    panel_dir,
    "gene_panel_5.csv"
)
    )

for col in gp2.columns:

    genes = (
        gp2[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    genes = [
        g for g in genes
        if g != "nan"
    ]

    panels[
        f"P2_{col}"
    ] = genes

# Panel_2 preprossesing




# Panel 3

gp3 = pd.read_csv(
    os.path.join(
    panel_dir,
    "gene_panel_3.csv"
)
    )

for col in gp3.columns:

    genes = (
        gp3[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    panels[
        f"P3_{col}"
    ] = genes


# Panel 4

gp4 = pd.read_csv(
    os.path.join(
    panel_dir,
    "gene_panel_4.csv"
)
    )

for col in gp4.columns:

    genes = (
        gp4[col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    panels[
        f"P4_{col}"
    ] = genes



expression = expression_GSE223655_vst.T.copy()

# 3. Run ssGSEA
ss = gp.ssgsea(
    data= expression,
    gene_sets=panels,
    outdir= output_dir ,
    permutation_num=0, # for ssGSEA
    min_size=3,
    max_size=500,
    seed=42
)

# saved results to read

scores_raw = ss.res2d

scores_all = scores_raw.pivot_table(
    index="Name",
    columns="Term",
    values="NES"
)

scores_all.to_csv(os.path.join(output_dir, "scores_all.csv"))

print("ssGSEA completed")

# %%
# Присоединяем метаданные
df = (
    meta
    .join(scores_all, how="inner", validate="one_to_one")
    .reset_index()
)
df

# Преобразуем значения scores в числа
scores_all = scores_all.apply(pd.to_numeric, errors="coerce")

# Проверяем типы
print(scores_all.dtypes.value_counts())

signatures = scores_all.columns.tolist()
results = []

for sig in signatures:
    for ct in ["CD8+CAR T cells", "CD4+CAR T cells"]:

        sub = df[df["cell subtype"] == ct].copy()

        # Гарантируем числовой формат
        sub[sig] = pd.to_numeric(sub[sig], errors="coerce")

        cr = sub.loc[
            sub["disease state"] == "CR patient", sig
        ].dropna()

        pd_group = sub.loc[
            sub["disease state"] == "PD patient", sig
        ].dropna()

        if len(cr) < 2 or len(pd_group) < 2:
            continue

        stat, pvalue = mannwhitneyu(
            cr.to_numpy(dtype=float),
            pd_group.to_numpy(dtype=float),
            alternative="two-sided"
        )

        results.append({
            "signature": sig,
            "cell subtype": ct,
            "n_CR": len(cr),
            "n_PD": len(pd_group),
            "median_CR": cr.median(),
            "median_PD": pd_group.median(),
            "median_difference": cr.median() - pd_group.median(),
            "pvalue": pvalue
        })

res = pd.DataFrame(results)

if not res.empty:
    res["FDR"] = multipletests(
        res["pvalue"],
        method="fdr_bh"
    )[1]

    res = res.sort_values(["cell subtype", "FDR"])

print(res.to_string(index=False))

res.to_csv(
    os.path.join(output_dir, "ssgsea_response_comparison.csv"),
    index=False
)
# %%

def run_ranked_gsea_by_celltype(
    counts_df,
    metadata_df,
    gene_sets,
    output_dir,
    min_size=3,
    max_size=500,
    permutations=1000
):
    """
    counts_df:
        genes x samples, raw integer counts;
        unique gene symbols as index.

    metadata_df:
        sample_id, patient id, disease state, cell subtype.

    gene_sets:
        dict of gene-set names -> gene symbols.
    """

    os.makedirs(output_dir, exist_ok=True)

    cell_types = [
        "CD8+CAR T cells",
        "CD4+CAR T cells"
    ]

    all_de_results = {}
    all_gsea_results = {}

    meta = metadata_df.copy()

    # Consistent sample IDs
    meta["sample_id"] = meta["sample_id"].astype(str)
    counts_df = counts_df.copy()
    counts_df.columns = counts_df.columns.astype(str)

    for cell_type in cell_types:

        print(f"\n{'=' * 60}")
        print(f"Cell type: {cell_type}")
        print(f"{'=' * 60}")

        # 1. Select samples of this cell type
        submeta = meta.loc[
            (meta["cell subtype"] == cell_type)
            & meta["disease state"].isin(
                ["CR patient", "PD patient"]
            )
        ].copy()

        # One observation per patient within each cell type
        if submeta["patient id"].duplicated().any():
            raise ValueError(
                f"Multiple samples per patient in {cell_type}. "
                "The design must account for repeated measurements."
            )

        sample_ids = submeta["sample_id"].tolist()

        if len(sample_ids) == 0:
            print("No samples found; skipping.")
            continue

        # Metadata indexed by sample ID
        metadata = (
            submeta
            .set_index("sample_id")
            .loc[sample_ids]
            .copy()
        )

        metadata = metadata.rename(
            columns={"disease state": "disease_state"}
        )

        # 2. Raw counts: samples x genes
        counts = counts_df.loc[:, sample_ids].T.copy()

        counts = counts.apply(
            pd.to_numeric,
            errors="raise"
        )

        if (counts < 0).any().any():
            raise ValueError("Raw counts cannot be negative.")

        if not np.allclose(
            counts.to_numpy(),
            np.round(counts.to_numpy())
        ):
            raise ValueError(
                "Counts are not integers. Use the original raw "
                "counts and correctly aggregate duplicate genes."
            )

        counts = counts.round().astype("int64")

        # Remove genes with zero counts in every sample
        counts = counts.loc[:, counts.sum(axis=0) > 0]

        # Check group sizes
        print(
            metadata["disease_state"].value_counts().to_dict()
        )

        if metadata["disease_state"].nunique() != 2:
            print("Only one response group; skipping.")
            continue

        if metadata["disease_state"].value_counts().min() < 2:
            print("Too few samples in one group; skipping.")
            continue

        # 3. Differential expression with PyDESeq2
        dds = DeseqDataSet(
            counts=counts,
            metadata=metadata,
            design="~ disease_state",
            refit_cooks=True,
            inference=DefaultInference()
        )

        dds.deseq2()

        # Positive log2FC/statistic means CR > PD
        ds = DeseqStats(
            dds,
            contrast=[
                "disease_state",
                "CR patient",
                "PD patient"
            ],
            inference=DefaultInference()
        )

        ds.summary()

        de = ds.results_df.copy()
        de.index.name = "gene"
        de = de.reset_index()

        de.to_csv(
            os.path.join(
                output_dir,
                f"DEG_CR_vs_PD_{cell_type.split('+')[0]}.tsv"
            ),
            sep="\t",
            index=False
        )

        all_de_results[cell_type] = de

        # 4. Prepare ranked gene list
        # Wald statistic includes direction and strength of evidence
        ranking = de[["gene", "stat"]].copy()

        ranking["gene"] = ranking["gene"].astype(str).str.strip()
        ranking["stat"] = pd.to_numeric(
            ranking["stat"],
            errors="coerce"
        )

        ranking = ranking.replace(
            [np.inf, -np.inf],
            np.nan
        ).dropna()

        ranking = ranking[
            (ranking["gene"] != "")
            & (ranking["gene"] != "nan")
        ]

        ranking = (
            ranking
            .sort_values("stat", ascending=False)
            .drop_duplicates("gene")
        )

        ranking.to_csv(
            os.path.join(
                output_dir,
                f"ranked_genes_{cell_type.split('+')[0]}.tsv"
            ),
            sep="\t",
            index=False
        )

        # 5. Ranked GSEA for the supplied gene panels
        gsea_outdir = os.path.join(
            output_dir,
            f"GSEA_{cell_type.split('+')[0]}"
        )

        os.makedirs(gsea_outdir, exist_ok=True)

        pre = gp.prerank(
            rnk=ranking[["gene", "stat"]],
            gene_sets=gene_sets,
            outdir=gsea_outdir,
            min_size=min_size,
            max_size=max_size,
            permutation_num=permutations,
            seed=42,
            threads=4,
            verbose=True
        )

        gsea_res = pre.res2d.copy()

        gsea_res.to_csv(
            os.path.join(
                gsea_outdir,
                "ranked_GSEA_results.tsv"
            ),
            sep="\t",
            index=False
        )

        all_gsea_results[cell_type] = gsea_res

        print(f"\nCompleted: {cell_type}")
        print(
            gsea_res[
                [
                    col for col in
                    ["Term", "NES", "NOM p-val", "FDR q-val",
                     "Lead_genes"]
                    if col in gsea_res.columns
                ]
            ].to_string(index=False)
        )

    return all_de_results, all_gsea_results

de_results, gsea_results = run_ranked_gsea_by_celltype(
    counts_df=expression_GSE223655,
    metadata_df=info_223655,
    gene_sets=panels,
    output_dir=os.path.join(
        output_dir,
        "ranked_GSEA"
    ),
    min_size=3,
    max_size=500,
    permutations=1000
)
# %%
gsea_tables = []

for cell_type, res in gsea_results.items():
    tmp = res.copy()
    tmp["cell_type"] = cell_type
    tmp["NES"] = pd.to_numeric(tmp["NES"], errors="coerce")
    tmp["FDR q-val"] = pd.to_numeric(
        tmp["FDR q-val"], errors="coerce"
    )
    gsea_tables.append(tmp[["Term", "cell_type", "NES", "FDR q-val"]])

gsea_combined = pd.concat(gsea_tables, ignore_index=True)

# Save combined results
gsea_combined.to_csv(
    os.path.join(output_dir, "GSEA_combined_CD4_CD8.tsv"),
    sep="\t",
    index=False
)

# Build NES matrix: rows = signatures, columns = cell types
nes_matrix = gsea_combined.pivot(
    index="Term",
    columns="cell_type",
    values="NES"
)

# Build significance annotation
fdr_matrix = gsea_combined.pivot(
    index="Term",
    columns="cell_type",
    values="FDR q-val"
)

stars = pd.DataFrame(
    "",
    index=nes_matrix.index,
    columns=nes_matrix.columns
)

# * FDR < 0.05; ** FDR < 0.01
for term in nes_matrix.index:
    for cell_type in nes_matrix.columns:
        nes = nes_matrix.loc[term, cell_type]
        fdr = fdr_matrix.loc[term, cell_type]

        if pd.isna(nes) or pd.isna(fdr):
            continue

        if abs(nes) > 1 and fdr < 0.05:
            stars.loc[term, cell_type] = "**"
        elif abs(nes) > 1 and fdr < 0.1:
            stars.loc[term, cell_type] = "*"


# Укажи сигнатуры в нужном порядке — названия должны совпадать с Term в GSEA
signature_order = [
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


# Оставляем только сигнатуры, которые присутствуют в результатах
signature_order = [
    s for s in signature_order
    if s in nes_matrix.index
]

# Применяем одинаковый порядок к NES и звёздочкам
nes_matrix = nes_matrix.reindex(signature_order)
stars = stars.reindex(signature_order)

# Symmetric color scale centered at zero
max_abs_nes = np.nanmax(np.abs(nes_matrix.to_numpy()))
if not np.isfinite(max_abs_nes) or max_abs_nes == 0:
    max_abs_nes = 1

# Plot
height = max(5, 0.35 * len(nes_matrix))

fig, ax = plt.subplots(figsize=(6, height))

sns.heatmap(
    nes_matrix,
    cmap="RdBu_r",
    center=0,
    vmin=-max_abs_nes,
    vmax=max_abs_nes,
    annot=stars,
    fmt="",
    linewidths=0.5,
    linecolor="white",
    mask=nes_matrix.isna(),
    cbar_kws={"label": "Normalized Enrichment Score (NES)"},
    ax=ax
)

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Ranked GSEA: CR vs PD")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig.tight_layout()

fig.savefig(
    os.path.join(output_dir, "GSEA_heatmap_CD4_CD8.png"),
    dpi=300,
    bbox_inches="tight"
)


plt.show()


