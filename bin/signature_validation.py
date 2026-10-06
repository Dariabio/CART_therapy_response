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

os.makedirs(
    output_dir,
    exist_ok=True
)



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

    # 1. Get DEGs of ECM and normalized data of them
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

