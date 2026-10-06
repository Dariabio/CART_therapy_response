import os
import argparse
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns



parser = argparse.ArgumentParser()


parser.add_argument(
    "--metadata",
    required=True
)

parser.add_argument(
    "--signature",
    required=True
) # cart_response_signature

parser.add_argument(
    "--gene_frequency_file",
    required=True,
    help=(
        "elastic_net_LOOCV_gene_stability"
    )
)

parser.add_argument(
    "--bootstrap_stability_file",
    required=True,
    help=(
        "elastic_net_bootstrap_gene_stability"
    )
)

parser.add_argument(
    "--expr_all_vst",
    required=True,
    help="VST expression matrix: genes x samples"
)


parser.add_argument(
    "--output_dir",
    required=True
)

parser.add_argument(
    "--figures_dir",
    required=True
)

args = parser.parse_args()

os.makedirs(
    args.output_dir,
    exist_ok=True
)

os.makedirs(
    args.figures_dir,
    exist_ok=True
)

signature = pd.read_csv(
    args.signature,
    sep="\t"
)

metadata = pd.read_csv(
    args.metadata
)

responder_genes_df  = signature[signature["direction"] == "responder_UP"]

responder_genes = signature.loc[
    signature["direction"] == "responder_UP",
    "gene"
].tolist()

# ELASTIC NET RESPONSE SIGNATURE

gene_frequency_file = pd.read_csv(args.gene_frequency_file,  sep="\t")
bootstrap_stability_file = pd.read_csv(args.bootstrap_stability_file,  sep="\t")

#LOOCV stability
loocv = gene_frequency_file[
    gene_frequency_file["selection_frequency"] >= 0.70
].copy()

bootstrap = bootstrap_stability_file[
    (bootstrap_stability_file["selection_frequency"] >= 0.70) &
    (bootstrap_stability_file["mean_coefficient"] > 0) &
    (bootstrap_stability_file["median_coefficient"] > 0)
].copy()

common_genes = loocv.merge(
    bootstrap,
    on="gene",
    suffixes=("_loocv", "_bootstrap")
)

common_with_degs = responder_genes_df[responder_genes_df['gene'].isin(common_genes['gene'])]

common_genes.to_csv(os.path.join(args.output_dir, "signature_elastic_net.tsv"), sep="\t")
common_with_degs.to_csv(os.path.join(args.output_dir, "signature_degs_elastic_net.tsv"), sep="\t")

vst = pd.read_csv(
    args.expr_all_vst,
    sep="\t",
    index_col=0
).T

meta = pd.read_csv(
    args.metadata,
    sep="\t",
    index_col=0
)

common_all_samples = meta.index.intersection(
    vst.index
)


if len(common_all_samples) == 0:
    raise ValueError(
        "No common sample IDs between metadata "
        "and VST matrix."
    )


print(
    f"Common metadata/VST samples: "
    f"{len(common_all_samples)}"
)

#  FILTER PRODUCT SAMPLES
# ============================================================

meta_product = meta[
    (meta["timepoint"] == "product") &
    (
        meta["response"].isin(
            [
                "responder",
                "nonresponder"
            ]
        )
    )
].copy()


if meta_product.empty:
    raise ValueError(
        "No product samples with "
        "responder/nonresponder labels found."
    )


common_product_samples = (
    meta_product.index.intersection(
        vst.index
    )
)


if len(common_product_samples) == 0:
    raise ValueError(
        "No common PRODUCT samples between "
        "metadata and VST."
    )


meta_product = meta_product.loc[
    common_product_samples
].copy()


vst_product = vst.loc[
    common_product_samples
].copy()


print(
    f"Product samples used: "
    f"{len(vst_product)}"
)


# match genes and expression
vst_genes = set(
    vst_product.columns
)


genes = [
    gene
    for gene in common_genes['gene']
    if gene in vst_genes
]

heatmap_data = (
    vst_product[
        genes
    ]
    .T
)


# ------------------------------------------------------------
# Sort samples:
# nonresponder -> responder
# ------------------------------------------------------------

response_order = {
    "nonresponder": 0,
    "responder": 1
}


meta_product["response_order"] = (
    meta_product["response"]
    .map(response_order)
)


sorted_samples = (
    meta_product
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
    "nonresponder": "#1F77B4",
    "responder": "#D62728"
}


col_colors = (
    meta_product
    .loc[
        sorted_samples,
        "response"
    ]
    .map(response_palette)
)


if col_colors.isna().any():
    raise ValueError(
        "Unknown response values detected "
        "when creating heatmap annotation."
    )


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


g.ax_heatmap.legend(
    handles=legend_elements,
    title="Response",
    bbox_to_anchor=(1.35, 1),
    loc="upper left"
)


g.fig.suptitle(
    "Product Elastic net gens: responder vs nonresponder "
    f"({len(genes)} genes)",
    y=1.02,
    fontsize=14
)


heatmap_file = os.path.join(
    args.figures_dir,
    "product_signature_elastic_net_heatmap.png"
)


g.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    g.fig
)
