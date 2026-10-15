import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from sklearn.decomposition import PCA

# Arguments

parser = argparse.ArgumentParser(
    description="Normalize expression, vst transformation and create PCA plots"
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
    "--output_dir",
    required=True
)

parser.add_argument(
    "--figures",
    required=True
)

args = parser.parse_args()

# Output directory

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.figures, exist_ok=True)

# Load data
counts = pd.read_csv(args.expression, sep="\t", index_col=0)
meta = pd.read_csv(args.clinical, sep="\t", index_col=0)


#1. DESeq2: design,normalization, vst and PCA
# PCA, check outlier and remove batch effects

design_formula = "~ batch_nexp"

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

expr_all.to_csv(os.path.join(args.output_dir, "expr_all_vst.tsv"), sep="\t")


#2. PCA

pca = PCA(
    n_components=10
)

pcs = pca.fit_transform(
    mat_vst
)


pc_df = pd.DataFrame(
    pcs[:, :2],
    columns=["PC1", "PC2"],
    index=mat_vst.index
)

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
            args.figures,
            filename
        ),
        dpi=300
    )

    plt.close()


make_pca(
    pc_df,
    pca,
    "group_analysis_1",
    "PCA_response.png",
    "PCA: response"
)

make_pca(
    pc_df,
    pca,
    "group_analysis_2",
    "PCA_original_groups.png",
    "PCA: original groups"
)


make_pca(
    pc_df,
    pca,
    "response",
    "PCA_new_groups.png",
    "PCA: new groups"
)

make_pca(
    pc_df,
    pca,
    "aggressiveness",
    "PCA_aggressive.png",
    "PCA: aggressive"
)

make_pca(
    pc_df,
    pca,
    "bone_marrow",
    "PCA_bone_marrow.png",
    "PCA: bone_marrow"
)

make_pca(
    pc_df,
    pca,
    "lesion",
    "PCA_widespread.png",
    "PCA: widespread"
)

make_pca(
    pc_df,
    pca,
    "timepoint",
    "PCA_timepoint.png",
    "PCA: timepoint"
)

make_pca(
    pc_df,
    pca,
    "batch_nexp",
    "PCA_batch.png",
    "PCA: batch"
)

print("VST and PCA completed.")

# PCA per each timepoint
# subset per meta, expression
meta_prod = meta[meta["timepoint"] == "product"]
mat_vst_prod = mat_vst.loc[meta_prod.index]

meta_d7 = meta[meta["timepoint"] == "d7 after"]
mat_vst_d7 = mat_vst.loc[meta_d7.index]

meta_d28 = meta[meta["timepoint"] == "d28 after"]
mat_vst_d28 = mat_vst.loc[meta_d28.index]

def make_pca_timepoint(info, mat_exp, hue, title, filename):
    info_df = info.copy()
    mat_exp_df  = mat_exp.copy()

    pca = PCA(n_components=10)
    pca_time = pca.fit_transform(mat_exp_df)

    pc_time_df = pd.DataFrame(
    pca_time[:, :2],
    columns=["PC1", "PC2"],
    index=mat_exp_df.index
).join(info_df)

    plt.figure(
            figsize=(7, 6)
        )
    
    sns.scatterplot(
    data= pc_time_df,
    x="PC1",
    y="PC2",
    hue=hue,
    palette="tab10",
    s=80)
    
    plt.xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.2%})"
        )
    
    plt.ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.2%})"
        )
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    
    plt.savefig(
    os.path.join(
    args.figures, filename),dpi=300)
    
    plt.close()

make_pca_timepoint(meta_prod, mat_vst_prod, "group_analysis_1", "pca_product.png", "PCA: product")
make_pca_timepoint(meta_d7, mat_vst_d7,"group_analysis_1", "pca_d7.png", "PCA: after d7")
make_pca_timepoint(meta_d28, mat_vst_d28, "group_analysis_1", "pca_d28.png", "PCA: after d28")

make_pca_timepoint(meta_prod, mat_vst_prod, "response", "pca_product_new.png", "PCA: product")
make_pca_timepoint(meta_d7, mat_vst_d7,"response", "pca_d7_new.png", "PCA: after d7")
make_pca_timepoint(meta_d28, mat_vst_d28, "response", "pca_d28_new.png", "PCA: after d28")







