#%%

import os
import numpy as np
import pandas as pd
import re
import glob
import urllib.request
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
import gseapy as gp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

#%%

project_dir = "/Users/darakuzmina/Desktop/CAR_T"
ann_dir = "/Users/darakuzmina/Desktop/crc_metastasis_analysis/data"
results_dir = os.path.join(project_dir, "results") 
figures_dir = os.path.join(project_dir, "figures") 
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

dir_raw = os.path.join(project_dir, "data", "raw")
dir_proc = os.path.join(project_dir, "data", "processed")
#%%
# 1. Merging all samples in one matrix expression
# tpm for GSEA ?
# raw counts for PCA and deseq2
folder = (os.path.join(dir_raw, "data_expression"))
pattern = os.path.join(folder, "*.tsv")
files = sorted(glob.glob(pattern))

# list of dfs
dfs1 = []
dfs2 = []

for fpath in files:
    fname = os.path.basename(fpath)
    sample_name = fname.replace(".tsv", "")

    df = pd.read_csv(fpath, sep="\t")

    df1 = df[["target_id", "tpm"]]
    df2 = df[["target_id", "est_counts"]]

    df1 = df1.rename(columns={"tpm": sample_name})
    df2 = df2.rename(columns={"est_counts": sample_name})

    dfs1.append(df1)
    dfs2.append(df2)

merged = dfs1[0]
for df in dfs1[1:]:
    merged = pd.merge(merged, df, on="target_id", how="outer")

merged.to_csv(os.path.join(dir_raw, "expression_tpm.tsv"), sep="\t")

merged = dfs2[0]
for df in dfs2[1:]:
    merged = pd.merge(merged, df, on="target_id", how="outer")

merged.to_csv(os.path.join(dir_raw, "expression_raw.tsv"), sep="\t")

# 2. Preprocess data
# Annotation and filtering

#%%
expression = pd.read_csv(os.path.join(dir_raw, "expression_raw.tsv"), sep="\t")
expression_tpm = pd.read_csv(os.path.join(dir_raw, "expression_tpm.tsv"), sep="\t")

def annotate_enst_ids_ncbi(expression, ann_dir, expr_type =None, id_col=None):
    df = expression.copy()

    
    mapping_file = os.path.join(ann_dir, "gene2ensembl.gz")
    gene_info_file = os.path.join(ann_dir, "Homo_sapiens.gene_info.gz")

    ensembl_df = pd.read_csv(
        mapping_file,
        compression="gzip",
        sep="\t",
        usecols=["#tax_id", "GeneID", "Ensembl_rna_identifier"],
    )

    # filter human (9606)
    ensembl_df = ensembl_df[ensembl_df["#tax_id"] == 9606]

    # filter zero rows
    ensembl_df = ensembl_df[
        (ensembl_df["Ensembl_rna_identifier"].notna())
        & (ensembl_df["Ensembl_rna_identifier"] != "-")
    ]

    # NCBI GeneID -> Official Symbol

    ncbi_df = pd.read_csv(
        gene_info_file, compression="gzip", sep="\t", usecols=["GeneID", "Symbol"]
    )

    #Merge tables
    mapping_df = pd.merge(ensembl_df, ncbi_df, on="GeneID", how="inner")

    # Re
    mapping_df["Ensembl_rna_clean"] = (
        mapping_df["Ensembl_rna_identifier"]
        .astype(str)
        .apply(lambda x: x.split(".")[0].strip())
    )

    # Mapping dict
    enst_to_symbol = dict(
        zip(mapping_df["Ensembl_rna_clean"], mapping_df["Symbol"])
    )

    # Orig IDs
    if id_col in df.columns:
        raw_ids = df[id_col].astype(str)
    else:
        raw_ids = df.index.to_series().astype(str)

    # remove ID versions
    query_ids = raw_ids.apply(lambda x: x.split(".")[0].strip())

    # Mapping
    df["symbol"] = query_ids.map(enst_to_symbol)

    if id_col in df.columns:
        df = df.drop(id_col, axis=1)

    # Remove duplicates

    if expr_type == "raw":
    
            df = df.groupby("symbol").sum().reset_index()
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].round().astype(int)

    else:
            df = df.groupby("symbol").mean(numeric_only=True).reset_index()
    

    # Remove unmapped
    df = df.dropna(subset=["symbol"])
    df = df.set_index("symbol")
    df = df.drop("Unnamed: 0", axis=1)

    print("✅ Аннотация успешно завершена!")
    return df

# Saving annotated raw and tpm files
expression_tpm = annotate_enst_ids_ncbi(
    expression_tpm, ann_dir=ann_dir, id_col='target_id', expr_type="tpm")

expression = annotate_enst_ids_ncbi(
    expression, ann_dir=ann_dir, id_col='target_id', expr_type="raw")

expression_tpm.to_csv(os.path.join(dir_raw, "expression_tpm_ann.tsv"), sep="\t")

expression.to_csv(os.path.join(dir_raw, "expression_raw_ann.tsv"), sep="\t")

#%%
expression = pd.read_csv(os.path.join(dir_raw, "expression_raw_ann.tsv"), index_col=0, sep="\t")
print("i love ryazhenka <3")

# 3. Check matching between expression and clinical data
info_cart = pd.read_csv(os.path.join(dir_raw,"clinical_data.csv"))
# _ in patient_id
info_cart["sample_id"] = info_cart["sample_id"].str.replace(" ", "_")

expr_samples = set(expression.columns)
clin_samples = set(info_cart["sample_id"])
common_samples = sorted(expr_samples & clin_samples)
print(f"Common samples: {len(common_samples)}")

expression_cart = expression.loc[:, common_samples]
meta = info_cart[info_cart["sample_id"].isin(common_samples)]
meta = meta.set_index("sample_id").reindex(common_samples)
assert list(expression_cart.columns) == list(meta.index), "Образцы не совпадают"

#remove zero counts
counts = expression_cart.copy()[(expression_cart > 0).any(axis=1)].T  # samples × genes
# match indices
common = sorted(set(counts.index) & set(meta.index))
counts = counts.reindex(common)
meta = meta.reindex(common)
assert list(counts.index) == list(meta.index), "Индексы counts и meta не совпадают"


# Make category covariates

meta["batch_nexp"] = meta["batch_nexp"].astype(str)
meta["batch_seq"] = meta["batch_seq"].astype(str)
meta["group_analysis_1"] = meta["group_analysis_1"].astype(str)
meta["group_analysis_2"] = meta["group_analysis_2"].astype(str)
meta["timepoint"] = meta["timepoint"].astype(str)
meta["patient_id"] = meta["patient_id"].astype(str)

# correlation between batch_nexp and batch_seq to choose batch
tab = pd.crosstab(meta["batch_nexp"], meta["batch_seq"])
chi2, p, dof, expected = chi2_contingency(tab)
print("Chi2:", chi2)
print("p-value:", p)
print("df:", dof)

# Cramér's V
n = tab.values.sum()
phi2 = chi2 / n
r, k = tab.shape
phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
V = np.sqrt(phi2corr / min(k-1, r-1))
print("Cramér's V:", V)
plt.figure(figsize=(6, 4))
sns.heatmap(
    tab,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.xlabel("batch_seq")
plt.ylabel("batch_nexp")
plt.tight_layout()
plt.show()

#%%

#4. DESeq2 without 3 and 4 group
# to see full picture, clusterization
# PCA, check outlier and remove batch effects
# resonder/unresponder and old/new experiment

# нужно ли учитывать timepoint???

design_formula = "~ batch_nexp + group_analysis_1"

dds = DeseqDataSet(
    counts=counts,
    metadata=meta,
    design=design_formula
)
dds.deseq2()
vsd = dds.vst()

vst = dds.layers['vst_counts']

#vst counts
mat_vst = pd.DataFrame(
    dds.layers['vst_counts'],
    index=dds.obs_names,   # samples
    columns=dds.var_names  # genes
)
# for GSEA
expr_all = mat_vst.T  # genes X samples
expr_all.index.name = "gene"


expr_all.to_csv(os.path.join(dir_proc, "expr_all_vst.tsv"), sep="\t")

# mat_vst: samples × genes
pca = PCA(n_components=10)
pcs = pca.fit_transform(mat_vst)  # (n_samples, 10)

pc_df = pd.DataFrame(
    pcs[:, :2],
    columns=["PC1", "PC2"],
    index=mat_vst.index
).join(meta)  # meta.index = sample_id = dds.obs_names


#5. PCA plots
#  group_analysis_1
plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="group_analysis_1",
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_group1.png"), bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="group_analysis_2",
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_group2.png"), bbox_inches='tight', dpi=300)
plt.show()


plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="timepoint",      # product / d28 / 2months
    style="group_analysis_1",  # responsive / unresponsive / unknown
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_timepoint_group1.png"), bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="timepoint",      # product / d28 / 2months
    style="group_analysis_2",  # responsive / unresponsive / unknown
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_timepoint_group2.png"), bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="patient_id",
    palette="tab20",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_patient_id.png"), bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="batch_nexp",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_nbatch.png"), bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_df,
    x="PC1", y="PC2",
    hue="batch_seq",
    s=80
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_seqbatch.png"), bbox_inches='tight', dpi=300)
plt.show()

# by timepoint
# subset meta per timepoint
meta_prod = meta[meta["timepoint"] == "product"]
mat_vst_prod = mat_vst.loc[meta_prod.index]

pca_prod = PCA(n_components=10)
pcs_prod = pca_prod.fit_transform(mat_vst_prod)

pc_prod_df = pd.DataFrame(
    pcs_prod[:, :2],
    columns=["PC1", "PC2"],
    index=mat_vst_prod.index
).join(meta_prod)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_prod_df,
    x="PC1", y="PC2",
    hue="group_analysis_1",
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca_prod.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca_prod.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_product_group1.png"), bbox_inches='tight', dpi=300)
plt.show()

#d7

meta_d7 = meta[meta["timepoint"] == "d7 after"]
mat_vst_d7 = mat_vst.loc[meta_d7.index]

pca_d7 = PCA(n_components=10)
pcs_d7 = pca_d7.fit_transform(mat_vst_d7)

pc_d7_df = pd.DataFrame(
    pcs_d7[:, :2],
    columns=["PC1", "PC2"],
    index=mat_vst_d7.index
).join(meta_d7)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_d7_df,
    x="PC1", y="PC2",
    hue="group_analysis_1",
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca_d7.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca_d7.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_d7_after_group1.png"), bbox_inches='tight', dpi=300)
plt.show()

#d28
meta_d28 = meta[meta["timepoint"] == "d28 after"]
mat_vst_d28 = mat_vst.loc[meta_d28.index]

pca_d28 = PCA(n_components=10)
pcs_d28 = pca_d28.fit_transform(mat_vst_d28)

pc_d28_df = pd.DataFrame(
    pcs_d28[:, :2],
    columns=["PC1", "PC2"],
    index=mat_vst_d28.index
).join(meta_d28)

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=pc_d28_df,
    x="PC1", y="PC2",
    hue="group_analysis_1",
    palette="tab10",
    s=80
)
plt.xlabel(f"PC1 ({pca_d28.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca_d28.explained_variance_ratio_[1]:.2%})")
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "pca_d28_after_group1.png"), bbox_inches='tight', dpi=300)
plt.show()


#6. DE analysis per each timepoint (product / d7 / d28)


meta_all = meta.copy()
# remove unknown (3 and 4 group)
meta = meta[meta['group_analysis_1'] != "unknown"]

# for GSEA
meta_prod = meta[meta["timepoint"] == "product"].copy()
counts_prod = counts.loc[meta_prod.index]

meta_d7 = meta[meta["timepoint"] == "d7 after"].copy()
counts_d7 = counts.loc[meta_d7.index]

meta_d28 = meta[meta["timepoint"] == "d28 after"].copy()
counts_d28 = counts.loc[meta_d28.index]

#product

dds_prod = DeseqDataSet(
    counts=counts_prod,
    metadata=meta_prod,
    design="~ batch_nexp + group_analysis_1"
)
dds_prod.deseq2()

ds = DeseqStats(
    dds_prod,
    contrast=["group_analysis_1", "responder", "nonresponder"]
)
ds.summary()
results_prod = ds.results_df


# after 7
dds_d7 = DeseqDataSet(
    counts=counts_d7,
    metadata=meta_d7,
    design="~ batch_nexp + group_analysis_1"
)
dds_d7.deseq2()

ds = DeseqStats(
    dds_d7,
    contrast=["group_analysis_1", "responder", "nonresponder"]
)
ds.summary()
results_d7 = ds.results_df

# after 28
dds_d28 = DeseqDataSet(
    counts=counts_d28,
    metadata=meta_d28,
    design="~ batch_nexp + group_analysis_1"
)
dds_d28.deseq2()
ds = DeseqStats(
    dds_d28,
    contrast=["group_analysis_1", "responder", "nonresponder"]
)
ds.summary()
results_d28 = ds.results_df

# Saving ranked gene list for GSEA
# Use log2FoldChange as metric
def save_files(de_results, timepoint=None):
    de_df = de_results.copy()

    gsea_input = de_df[["log2FoldChange"]].dropna()
    gsea_input = gsea_input.sort_index()  # по gene name

    gsea_input.to_csv(os.path.join(results_dir, f"DE_{timepoint}_ranked.csv"),
                  sep="\t")

    return gsea_input

gsea_prod = save_files(results_prod, timepoint="prod")
gsea_d7 = save_files(results_d7, timepoint="d7")
gsea_d28 = save_files(results_d28, timepoint="d28")

# %%
# 7. ranked GSEA global

DE_prod_ranked = pd.read_csv(os.path.join(results_dir,"DE_prod_ranked.csv"), sep="\t", index_col = False)
DE_d7_ranked = pd.read_csv(os.path.join(results_dir,"DE_d7_ranked.csv"), sep="\t", index_col = False)
DE_d28_ranked = pd.read_csv(os.path.join(results_dir,"DE_d28_ranked.csv"), sep="\t", index_col = False)

def run_gsea_public_databases(DE_results, out_base_dir, timepoint):
    df = DE_results.copy()
    df = df.drop_duplicates(subset='gene', keep='first').copy()
        
    # We add micro-noise and sort (from the highest LFC among responders to the lowest)
    np.random.seed(42)
    noise = np.random.normal(0, 1e-9, size=len(df))
    df['rank_metric'] = df['log2FoldChange'] + noise
    df = df.sort_values('rank_metric', ascending=False)
        
    rnk_input = df[['gene', 'rank_metric']]
    
    # Список баз данных, по которым запускаем анализ
    public_sets = ['KEGG_2021_Human', 'GO_Biological_Process_2021']
    
    for db in public_sets:
        out_dir = os.path.join(out_base_dir, f"GSEA_{db}_{timepoint}")
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"Running GSEA prerank for {timepoint} using {db}...")
        pr = gp.prerank(
            rnk=rnk_input,
            gene_sets=db,  # database
            outdir=out_dir,
            permutation_num=1000,
            min_size=15,   # for big db higher
            max_size=500,
            seed=42, verbose=True
        )
        
        # Filter significant (FDR < 0.05)
        significant_terms = pr.res2d[pr.res2d['FDR q-val'] < 0.05]
        significant_terms.to_csv(
            os.path.join(out_dir, f"significant_results_{db}_{timepoint}.tsv"), 
            sep="\t", index=False
        )

base_output = os.path.join(results_dir, "public_gsea_results")

# Positive NES have responders, negative — nonresponders
run_gsea_public_databases(DE_prod_ranked, base_output, "product")
run_gsea_public_databases(DE_d7_ranked, base_output, "d7")
run_gsea_public_databases(DE_d28_ranked, base_output, "d28")

#%%
# 8. Creating and processing gene panels
gp_1 = pd.read_csv(os.path.join(dir_raw, "gene_panel_1.csv"))
gp_2 = pd.read_csv(os.path.join(dir_raw, "gene_panel_2.csv"))
gp_3 = pd.read_csv(os.path.join(dir_raw, "gene_panel_3.csv"))
gp_4 = pd.read_csv(os.path.join(dir_raw, "gene_panel_4.csv"))

#making dict
panels_dict_1 = {}
for col in gp_1.columns:
    genes = gp_1[col].dropna().astype(str).tolist()
    genes = [g.strip() for g in genes if g and g != "nan"]
    panels_dict_1[f"P1_{col}"] = genes

panels_dict_3 = {}
for col in gp_3.columns:
    genes = gp_3[col].dropna().astype(str).tolist()
    genes = [g.strip() for g in genes if g and g != "nan"]
    panels_dict_3[f"P3_{col}"] = genes

panels_dict_4 = {}
for col in gp_4.columns:
    genes = gp_4[col].dropna().astype(str).tolist()
    genes = [g.strip() for g in genes if g and g != "nan"]
    panels_dict_4[f"P4_{col}"] = genes

panels_dict_2 = {}
gp_2["subpop"] = gp_2["Annotation"] + "_" + gp_2["Gene_Category"]
for name, grp in gp_2.groupby("subpop"):
    genes = grp["Gene"].dropna().astype(str).tolist()
    genes = [g.strip() for g in genes if g and g != "nan"]
    panels_dict_2[f"P2_{name}"] = genes

# merging panels
panels = {}
panels.update(panels_dict_1)
panels.update(panels_dict_2)
panels.update(panels_dict_3)
panels.update(panels_dict_4)

# %%
# 9. ranked GSEA (use DE results of each timepoint) for 4 gene panels

# function for prerank per one group comparisan
def run_group_prerank(DE_results, panels, out_dir):
    df = DE_results.copy()
    df = df.drop_duplicates(subset='gene', keep='first').copy()
    
    # We add micro-noise and sort (from the highest LFC among responders to the lowest)
    np.random.seed(42)
    noise = np.random.normal(0, 1e-9, size=len(df))
    df['rank_metric'] = df['log2FoldChange'] + noise
    df = df.sort_values('rank_metric', ascending=False)
    
    rnk_input = df[['gene', 'rank_metric']]
    
    pr = gp.prerank(
        rnk=rnk_input,
        gene_sets=panels,  
        outdir=out_dir,
        permutation_num=1000,
        min_size=3,
        max_size=500,
        seed=42
    )
    # Results (NES, p-value)
    return pr.res2d


res_prod = run_group_prerank(
    DE_prod_ranked, 
    panels=panels, 
    out_dir=os.path.join(results_dir, 'GSEA_rank_groups_product')
)

res_d7 = run_group_prerank(
    DE_d7_ranked, 
    panels=panels, 
    out_dir=os.path.join(results_dir, 'GSEA_rank_groups_d7')
)

res_d28 = run_group_prerank(
    DE_d28_ranked, 
    panels=panels, 
    out_dir=os.path.join(results_dir, 'GSEA_rank_groups_d28')
)

# Positive NES have responders, negative — nonresponders
print(res_prod[['Term', 'NES', 'NOM p-val', 'FDR q-val']].head())

#%%
# 10. ssGSEA - 4 gene panels per each sample (without DE, need to compare responders with nonresponders)
# all expression matrix icluding 3 and 4 group (as works per one sample)
expr_all = pd.read_csv(os.path.join(dir_proc, "expr_all_vst.tsv"), index_col=0, sep="\t")
gene_sets = panels

out_dir = os.path.join(results_dir, 'results_gsea')
os.makedirs(out_dir, exist_ok=True)

ss = gp.ssgsea(
    data=expr_all,
    gene_sets=gene_sets,
    outdir= out_dir ,
    permutation_num=0,# for ssGSEA
    min_size=3,
    max_size=500,
    seed=42
)
#%%

scores_raw = pd.read_csv(os.path.join(results_dir, "results_gsea", "gseapy.gene_set.ssgsea.report.csv"), sep=",")

scores_wide = scores_raw.pivot_table(
    index="Name",
    columns="Term",
    values="NES"  # или "ES"
)
# scores_wide: samples × subpopulations
scores_all = scores_wide
#%%
# Compare responders vs nonresponders
# remove 3 and 4 groups per test
scores_known = scores_all.loc[
    meta_all["group_analysis_1"].isin(["responder", "nonresponder"])
]

scores_unknown = scores_all.loc[
    meta_all["group_analysis_2"].isin(["3", "4"])
]

# saving for calculating dynamics


scores_all.to_csv(os.path.join(results_dir, "results_gsea", "scores_all.csv"))
scores_known.to_csv(os.path.join(results_dir, "results_gsea", "scores_known.csv"))
scores_unknown.to_csv(os.path.join(results_dir, "results_gsea", "scores_unknown.csv"))
meta_all.to_csv(os.path.join(dir_proc, "meta_all.csv"))
# subset per each timepoint

meta_prod = meta_all[meta_all["timepoint"] == "product"].copy()
common_prod = sorted(set(meta_prod.index) & set(scores_known.index))
meta_prod = meta_prod.loc[common_prod]
scores_prod = scores_known.loc[common_prod]

meta_d7 = meta_all[meta_all["timepoint"] == "d7 after"].copy()
common_d7 = sorted(set(meta_d7.index) & set(scores_known.index))
meta_d7 = meta_d7.loc[common_d7]
scores_d7 = scores_known.loc[common_d7]

meta_d28 = meta_all[meta_all["timepoint"] == "d28 after"].copy()
common_d28 = sorted(set(meta_d28.index) & set(scores_known.index))
meta_d28 = meta_d28.loc[common_d28]
scores_d28 = scores_known.loc[common_d28]

#%%
# Test - compare responders with nonresponders ssGSEA results

def test_respvsnonresp(meta_df, scores_df, timepoint=None):
    meta = meta_df.copy()
    scores = scores_df.copy()
    
    results = []
    for col in scores.columns:
        resp = scores.loc[meta["group_analysis_1"] == "responder", col]
        nonresp = scores.loc[meta["group_analysis_1"] == "nonresponder", col]
    
        if len(resp) < 2 or len(nonresp) < 2:
            continue

        if resp.std() == 0 and nonresp.std() == 0:
            continue
    

        stat, p = mannwhitneyu(
            resp, nonresp, alternative="two-sided"
        )

        results.append({
            "subpopulation": col,
            "p_value": p,
            "mean_resp": resp.mean(),
            "mean_nonresp": nonresp.mean(),
            "median_resp": resp.median(),  # Для Манна-Уитни полезно выводить медианы
            "median_nonresp": nonresp.median(),
        })

    df_results = pd.DataFrame(results)
    
    reject, fdr_corrected, _, _ = multipletests(
        df_results["p_value"], alpha=0.1, method="fdr_bh"
    )
    df_results["FDR"] = fdr_corrected

    # Сортируем по возрастанию FDR, а затем по p_value
    df_results = df_results.sort_values(by=["FDR", "p_value"])

    df_results.to_csv(os.path.join(results_dir, f"GSEA_4panel_res_{timepoint}.csv"),
                      sep="\t")

    return df_results

gsea_respan_prod = test_respvsnonresp(meta_prod, scores_prod, timepoint="product")
gsea_respan_d7 = test_respvsnonresp(meta_d7, scores_d7, timepoint="d7")
gsea_respan_d28 = test_respvsnonresp(meta_d28, scores_d28, timepoint="d28")

# %%
# Vizualization of GSEA results
# ssGSEA results - PCA by each point
# ssGSEA results - heatmap (all 4 gene panels) by each point

def pca_ssgsea(scores_df, meta_df, timepoint=None):
    scores_plot = scores_df.copy()

    # 1. PCA plot

    pca = PCA(n_components=2)

    pcs = pca.fit_transform(scores_plot)

    pca_df = pd.DataFrame(
    pcs,
    columns=["PC1","PC2"],
    index=scores_plot.index)

    meta_pca = meta_df.copy()
    pca_df = pca_df.join(meta_pca)

    plt.figure(figsize=(8,6))

    sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="group_analysis_1",
    style="timepoint",
    s=120
)

    # Add patient ids
    for patient_id, row in pca_df.iterrows():
        plt.text(
                row["PC1"],  
                row["PC2"], 
                str(row["patient_id"]), 
                fontsize=8,        
                alpha=0.8,         
                weight="semibold" 
            )

    plt.xlabel(
    f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)"
)

    plt.ylabel(
    f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)"
)

    plt.tight_layout()

    plt.savefig(
    os.path.join(figures_dir,f"PCA_{timepoint}_ssGSEA.png"),
    dpi=300
)
    plt.show()

    # 2. Heatmap

    scores_z = scores_df.copy()

    scores_z = scores_z.apply(
    zscore,
    axis=0)

    meta_hm = meta_df.copy()

    # sort by annotation
    sorted_index = meta_hm.sort_values(by="group_analysis_1").index
    scores_z = scores_z.loc[sorted_index]
    meta_hm = meta_hm.loc[sorted_index]

    new_labels = meta_hm["patient_id"]
    scores_z.index = new_labels
    meta_hm.index = new_labels

    row_colors = pd.DataFrame({
        "Response": meta_hm["group_analysis_1"].map({
            "responder":"red",
            "nonresponder":"blue"
        })
    }, index=scores_z.index)

    g = sns.clustermap(
        scores_z,
        cmap="RdBu_r",
        center=0,
        col_cluster=False,
        row_cluster=False, # Set to False to keep strict sorting without tree reordering
        row_colors=row_colors,
        figsize=(12,10),
        xticklabels=True,
        yticklabels=True
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=8, rotation=90)
    
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=15, rotation=0)

    plt.savefig(
    os.path.join(
        figures_dir,
        f"Heatmap_ssGSEA_{timepoint}.png"
    ),
    dpi=300
)

    plt.show()

# pca and heatmap ssGSEA
pca_ssgsea(scores_prod, meta_prod, timepoint="product")
pca_ssgsea(scores_d7, meta_d7, timepoint="d7")
pca_ssgsea(scores_d28, meta_d28, timepoint="d28")

# %%
# ssGSEA - comparison between resp vs unresp
# product as the first point

# Сначала собираем ваши результаты по всем трем точкам в один словарь для удобного поиска в цикле
gsea_results_dict = {
    "product": gsea_respan_prod,
    "d7 after": gsea_respan_d7,   
    "d28 after": gsea_respan_d28   
}

# Из "product" (как в вашем исходном коде) берем топ-10 сигнатур для визуализации
top_signatures = (
    gsea_respan_prod
    .sort_values("p_value")
    .head(10)
    ["subpopulation"]
)

# Список, в который мы запишем финальный отчет по нарисованным графикам
plots_log = []

for sig in top_signatures:
    for tp in [
        "product",
        "d7 after",
        "d28 after"
    ]:
        # 1. Достаем готовую таблицу результатов для текущей временной точки
        current_gsea_res = gsea_results_dict.get(tp)
        if current_gsea_res is None or current_gsea_res.empty:
            continue
            
        # 2. Ищем строку для нашей текущей сигнатуры (sig)
        sig_stats = current_gsea_res[current_gsea_res["subpopulation"] == sig]
        if sig_stats.empty:
            continue
            
        # Вытаскиваем посчитанные вашей функцией p-value и FDR
        p_val = sig_stats["p_value"].values[0]
        fdr_val = sig_stats["FDR"].values[0]

        # КРИТИЧЕСКИЙ ШАГ: Если p-value в этой временной точке >= 0.05, график НЕ строим
        if p_val >= 0.05:
            continue

        # 3. Находим образцы и строим DataFrame для графика
        samples_tp = meta_all[meta_all["timepoint"] == tp].index
        common = scores_all.index[scores_all.index.isin(samples_tp)]

        if len(common) == 0:
            continue

        plot_df = pd.DataFrame({
            "score": scores_all.loc[common, sig],
            "group": meta_all.loc[common, "group_analysis_1"]
        })

        plot_df = plot_df[
            plot_df["group"].isin(["responder", "nonresponder"])
        ]

        if plot_df.empty:
            continue

        # 4. Если проверка пройдена, добавляем информацию в итоговую табличку графиков
        plots_log.append({
            "Signature": sig,
            "Timepoint": tp,
            "p_value": p_val,
            "FDR": fdr_val,
            "File_Name": f"Boxplot_{sig}_{tp}.png"
        })

        # 5. Отрисовка boxplot
        plt.figure(figsize=(5, 4))
        target_order = ["responder", "nonresponder"]
        
        sns.boxplot(
            data=plot_df,
            x="group",
            y="score",
            order=target_order
        )

        sns.stripplot(
            data=plot_df,
            x="group",
            y="score",
            color="black",
            size=6,
            order=target_order,
            jitter=0.2
        )

        # Выводим в заголовок оригинальный p-value из вашей таблицы
        plt.title(f"{sig}\n{tp} (p-val: {p_val:.4f})")
        plt.tight_layout()

        os.makedirs(os.path.join(figures_dir, "ssGSEA"), exist_ok=True)

        plt.savefig(
            os.path.join(
                figures_dir, "ssGSEA",
                f"Boxplot_{sig}_{tp}.png"
            ),
            dpi=300
        )

        plt.close()

# --- СОХРАНЕНИЕ СВОДНОЙ ТАБЛИЦЫ ВЫПЛУНУТЫХ ГРАФИКОВ ---
if plots_log:
    df_plots_log = pd.DataFrame(plots_log)
    df_plots_log = df_plots_log.sort_values(by="p_value")
    
    # Сохраняем отчет (только по тем парам, для которых создались графики)
    summary_path = os.path.join(figures_dir, "ssGSEA", "generated_plots_summary.csv")
    df_plots_log.to_csv(summary_path, index=False, sep="\t")

# %%
# ranked GSEA 4 panels
nes_prod = pd.to_numeric(res_prod.set_index("Term")["NES"], errors='coerce')
nes_d7 = pd.to_numeric(res_d7.set_index("Term")["NES"], errors='coerce')
nes_d28 = pd.to_numeric(res_d28.set_index("Term")["NES"], errors='coerce')

# Объединяем в матрицу
nes_mat = pd.concat(
    [
        nes_prod.rename("product"),
        nes_d7.rename("d7"),
        nes_d28.rename("d28")
    ],
    axis=1
)

# Заполняем пропуски нулями (если термины не совпадают между таймпоинтами)
# Для NES (Normalized Enrichment Score) ноль означает отсутствие обогащения
nes_mat = nes_mat.fillna(0)

# Строим кластерную карту
sns.clustermap(
    nes_mat,
    cmap="RdBu_r",
    center=0,
    figsize=(12, 35),
    col_cluster=False
)

plt.savefig(
    os.path.join(figures_dir, "NES_timepoints_heatmap.png"),
    dpi=300
)

plt.show()
# %%
# significant NES in responders and nonresponders ranked GSEA (4 panels)

def get_best(res_df, fdr=0.05):
    res = res_df.copy()

    # Приводим к float, чтобы сохранить дробную часть p-value и NES
    num_col = ["FDR q-val", "NES"]
    res[num_col] = res[num_col].astype(float)

    # Фильтруем: оставляем ВСЕ строки, где FDR меньше заданного порога
    sig = res[res["FDR q-val"] < fdr].copy()

    # Разделяем на положительные и отрицательные NES
    # И сортируем их для удобства (лучшие по NES и FDR будут сверху)
    pos = sig[sig["NES"] > 0].sort_values(
        ["NES", "FDR q-val"], 
        ascending=[False, True]
    )
    
    neg = sig[sig["NES"] < 0].sort_values(
        ["NES", "FDR q-val"], 
        ascending=[True, True]
    )

    # Функция теперь возвращает два полноценных DataFrame
    return pos, neg


prod_pos, prod_neg = get_best(res_prod)

d7_pos, d7_neg = get_best(res_d7)

d28_pos, d28_neg = get_best(res_d28)

# make panel of these pictures
# %%


