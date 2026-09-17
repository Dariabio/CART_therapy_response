import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from matplotlib.patches import Patch


# ARGUMENTS


parser = argparse.ArgumentParser(
    description="CAR-T response signature analysis"
)

parser.add_argument(
    "--expr_all_vst",
    required=True,
    help="VST expression matrix: genes x samples"
)

parser.add_argument(
    "--metadata",
    required=True,
    help="Metadata table"
)

parser.add_argument(
    "--de_product",
    required=True,
    help="DESeq2 product results for responder vs nonresponder"
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

top_n = 50

# Minimum number of genes required
# in EACH signature direction
MIN_SIGNATURE_GENES = 5


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    args.output_dir,
    exist_ok=True
)

os.makedirs(
    args.figures,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

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


de = pd.read_csv(
    args.de_product,
    sep="\t",
    index_col=0
)


# ============================================================
# 2. CHECK VST / METADATA SAMPLE IDs
# ============================================================

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


# ============================================================
# 3. FILTER PRODUCT SAMPLES
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


# ============================================================
# 4. FILTER DE RESULTS
# ============================================================

required_de_columns = [
    "padj",
    "log2FoldChange"
]


missing_columns = [
    col
    for col in required_de_columns
    if col not in de.columns
]


if missing_columns:
    raise ValueError(
        "DESeq2 product file is missing "
        f"columns: {missing_columns}"
    )


de_product = de[
    de["padj"].notna()
].copy()


n_padj = len(de_product)


print(
    f"DEGs with non-NA padj: "
    f"{n_padj}"
)


de_product = de_product[
    (de_product["padj"] < 0.05) &
    (de_product["log2FoldChange"].abs() > 1)
].copy()


if de_product.empty:
    raise ValueError(
        "No significant DEGs found in "
        "DESeq2_product.tsv using "
        "padj < 0.05 and |log2FC| > 1."
    )


print(
    f"Significant DEGs: "
    f"{len(de_product)}"
)


# ============================================================
# 5. MATCH DEG GENES TO VST
# ============================================================

vst_genes = set(
    vst_product.columns
)


de_genes = [
    gene
    for gene in de_product.index
    if gene in vst_genes
]


if len(de_genes) == 0:

    debug_file = os.path.join(
        args.output_dir,
        "product_significant_DEGs.tsv"
    )

    de_product.to_csv(
        debug_file,
        sep="\t"
    )

    raise ValueError(
        "Significant DEGs were found in "
        "DESeq2_product.tsv, but NONE of them "
        "were found in expr_all_vst.tsv. "
        "Check gene identifiers."
    )


# Save matched DEGs

matched_de = de_product.loc[
    de_genes
].copy()


matched_de_file = os.path.join(
    args.output_dir,
    "product_significant_DEGs.tsv"
)


matched_de.to_csv(
    matched_de_file,
    sep="\t"
)


print(
    f"Significant DEGs matched to VST: "
    f"{len(matched_de)}"
)


# ============================================================
# 6. DEG HEATMAP
# ============================================================

heatmap_data = (
    vst_product[
        de_genes
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
    "Product DEGs: responder vs nonresponder "
    f"({len(de_genes)} genes)",
    y=1.02,
    fontsize=14
)


heatmap_file = os.path.join(
    args.figures,
    "product_DEG_heatmap.png"
)


g.savefig(
    heatmap_file,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    g.fig
)


if not os.path.exists(
    heatmap_file
):
    raise RuntimeError(
        "Heatmap was not created: "
        f"{heatmap_file}"
    )


print(
    "DEG heatmap saved:",
    heatmap_file
)


# ============================================================
# 7. DEFINE RESPONSE SIGNATURE
# ============================================================

# log2FC > 1:
# higher in responders
#
# log2FC < -1:
# higher in nonresponders


up_genes = (
    de_product[
        de_product["log2FoldChange"] > 1
    ]
    .sort_values(
        "log2FoldChange",
        ascending=False
    )
)


down_genes = (
    de_product[
        de_product["log2FoldChange"] < -1
    ]
    .sort_values(
        "log2FoldChange",
        ascending=True
    )
)


top_up = [
    gene
    for gene in up_genes.index
    if gene in vst_genes
][:top_n]


top_down = [
    gene
    for gene in down_genes.index
    if gene in vst_genes
][:top_n]


# ============================================================
# 8. REQUIRE MINIMUM 5 GENES PER DIRECTION
# ============================================================

if len(top_up) < MIN_SIGNATURE_GENES:

    raise ValueError(
        f"Responder signature contains only "
        f"{len(top_up)} genes. At least "
        f"{MIN_SIGNATURE_GENES} genes are required."
    )

'''
if len(top_down) < MIN_SIGNATURE_GENES:

    raise ValueError(
        f"Nonresponder signature contains only "
        f"{len(top_down)} genes. At least "
        f"{MIN_SIGNATURE_GENES} genes are required."
    )
'''


print(
    f"Responder signature: "
    f"{len(top_up)} genes"
)

print(
    f"Nonresponder signature: "
    f"{len(top_down)} genes"
)


# ============================================================
# 9. SAVE SIGNATURE
# ============================================================

signature = pd.DataFrame({

    "gene": (
        top_up +
        top_down
    ),

    "direction": (
        ["responder_UP"] * len(top_up)
        +
        ["nonresponder_UP"] * len(top_down)
    )

})


signature_file = os.path.join(
    args.output_dir,
    "CAR_T_response_signature.tsv"
)


signature.to_csv(
    signature_file,
    sep="\t",
    index=False
)


print(
    "Signature saved:",
    signature_file
)


# ============================================================
# 10. PREPARE SIGNATURE GENES
# ============================================================

responder_genes = (
    signature.loc[
        signature["direction"] == "responder_UP",
        "gene"
    ]
    .dropna()
    .astype(str)
    .tolist()
)


nonresponder_genes = (
    signature.loc[
        signature["direction"] == "nonresponder_UP",
        "gene"
    ]
    .dropna()
    .astype(str)
    .tolist()
)


# ============================================================
# 11. CHECK SIGNATURE GENES IN VST
# ============================================================

responder_genes_present = [
    gene
    for gene in responder_genes
    if gene in vst_product.columns
]


nonresponder_genes_present = [
    gene
    for gene in nonresponder_genes
    if gene in vst_product.columns
]


print(
    f"Responder genes present in VST: "
    f"{len(responder_genes_present)}/"
    f"{len(responder_genes)}"
)


print(
    f"Nonresponder genes present in VST: "
    f"{len(nonresponder_genes_present)}/"
    f"{len(nonresponder_genes)}"
)


if len(responder_genes_present) < MIN_SIGNATURE_GENES:

    raise ValueError(
        "Fewer than 5 responder signature genes "
        "are available in VST."
    )

'''
if len(nonresponder_genes_present) < MIN_SIGNATURE_GENES:

    raise ValueError(
        "Fewer than 5 nonresponder signature genes "
        "are available in VST."
    )
'''


# ============================================================
# 12. GENE-WISE Z-SCORE
# ============================================================

def zscore_columns(df):
    """
    Gene-wise z-score across product samples.
    """

    means = df.mean(axis=0)

    stds = df.std(
        axis=0,
        ddof=1
    )

    stds = stds.replace(
        0,
        np.nan
    )

    return (
        df - means
    ) / stds

'''
score_genes = list(
    dict.fromkeys(
        responder_genes_present +
        nonresponder_genes_present
    )
)
'''
score_genes = list(
    dict.fromkeys(
        responder_genes_present
    )
)


expr_signature = (
    vst_product[
        score_genes
    ]
    .copy()
)


expr_signature_z = (
    zscore_columns(
        expr_signature
    )
)


# ============================================================
# 13. CALCULATE SIGNATURE SCORES
# ============================================================

# Responder component:
# high value = high expression of responder genes

meta_product["responder_score"] = (
    expr_signature_z[
        responder_genes_present
    ]
    .mean(axis=1)
)


# ============================================================
# 14. AGGREGATE TO PATIENT LEVEL
# ============================================================

# Logistic regression should not treat multiple samples
# from the same patient as independent observations.

patient_scores = (
    meta_product[
        [
            "patient_id",
            "response",
            "aggressiveness",
            "bone_marrow",
            "lesion",
            "responder_score"
        ]
    ]
    .groupby("patient_id")
    .agg({
        "responder_score": "mean",
        "response": "first",
        "aggressiveness": "first",
        "bone_marrow": "first",
        "lesion": "first"
    })
    .copy()
)


# ============================================================
# 15. CHECK PATIENT-LEVEL CLINICAL CONSISTENCY
# ============================================================

clinical_columns = [
    "response",
    "aggressiveness",
    "bone_marrow",
    "lesion"
]


for column in clinical_columns:

    consistency = (
        meta_product
        .groupby("patient_id")[column]
        .nunique(
            dropna=True
        )
    )

    inconsistent = consistency[
        consistency > 1
    ]

    if len(inconsistent) > 0:

        print(
            f"WARNING: inconsistent {column} "
            f"for patients: "
            f"{list(inconsistent.index)}"
        )


# ============================================================
# 16. BINARY ENCODING
# ============================================================

patient_scores["response_binary"] = (
    patient_scores["response"]
    .map({
        "nonresponder": 0,
        "responder": 1
    })
)


patient_scores["aggressiveness_binary"] = (
    patient_scores["aggressiveness"]
    .map({
        "indolent": 0,
        "aggressive": 1
    })
)


patient_scores["bone_marrow_binary"] = (
    patient_scores["bone_marrow"]
    .map({
        "bone_marrow_negative": 0,
        "bone_marrow_positive": 1
    })
)


patient_scores["lesion_binary"] = (
    patient_scores["lesion"]
    .map({
        "localized": 0,
        "widespread": 1
    })
)


# ============================================================
# 17. SAVE DATA USED FOR LOGISTIC REGRESSION
# ============================================================

regression_data_file = os.path.join(
    args.output_dir,
    "signature_logistic_regression_data.tsv"
)


patient_scores.to_csv(
    regression_data_file,
    sep="\t"
)


print(
    "Regression dataset saved:",
    regression_data_file
)


# ============================================================
# 18. LOGISTIC REGRESSION FUNCTION
# ============================================================

def fit_logistic(
    df,
    response_column,
    predictors
):

    columns = [
        response_column
    ] + predictors

    analysis_df = (
        df[columns]
        .dropna()
        .copy()
    )

    if len(analysis_df) < 5:

        raise ValueError(
            f"Not enough complete observations "
            f"for model with predictors: "
            f"{predictors}"
        )

    y = (
        analysis_df[
            response_column
        ]
        .astype(float)
    )

    X = (
        analysis_df[
            predictors
        ]
        .astype(float)
    )

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    if y.nunique() < 2:

        raise ValueError(
            "Response contains fewer than two classes."
        )

    model = sm.Logit(
        y,
        X
    ).fit(
        disp=False
    )

    result_rows = []

    confidence_intervals = (
        model.conf_int()
    )

    for predictor in predictors:

        beta = (
            model.params[
                predictor
            ]
        )

        p_value = (
            model.pvalues[
                predictor
            ]
        )

        ci_low_beta = (
            confidence_intervals
            .loc[
                predictor,
                0
            ]
        )

        ci_high_beta = (
            confidence_intervals
            .loc[
                predictor,
                1
            ]
        )

        result_rows.append({

            "predictor": predictor,

            "n": len(
                analysis_df
            ),

            "coefficient": beta,

            "odds_ratio": np.exp(
                beta
            ),

            "ci95_low": np.exp(
                ci_low_beta
            ),

            "ci95_high": np.exp(
                ci_high_beta
            ),

            "p_value": p_value

        })

    return pd.DataFrame(
        result_rows
    )


# ============================================================
# 19. UNIVARIATE LOGISTIC REGRESSION
# ============================================================

print(
    "\nRunning univariate logistic regression..."
)


univariate_predictors = [
    "responder_score"
]


univariate_results = []


for predictor in univariate_predictors:

    try:

        model_result = fit_logistic(
            patient_scores,
            "response_binary",
            [predictor]
        )

        model_result[
            "model"
        ] = "univariate"

        univariate_results.append(
            model_result
        )

    except Exception as error:

        print(
            f"WARNING: univariate model failed "
            f"for {predictor}: {error}"
        )


if len(univariate_results) == 0:

    raise RuntimeError(
        "No univariate logistic regression "
        "could be fitted."
    )


logistic_univariate = pd.concat(
    univariate_results,
    ignore_index=True
)


# Multiple-testing correction
logistic_univariate[
    "padj"
] = np.nan


valid_p = (
    logistic_univariate[
        "p_value"
    ].notna()
)


from statsmodels.stats.multitest import multipletests


if valid_p.sum() > 0:

    logistic_univariate.loc[
        valid_p,
        "padj"
    ] = multipletests(
        logistic_univariate.loc[
            valid_p,
            "p_value"
        ],
        method="fdr_bh"
    )[1]


# Human-readable names

logistic_univariate[
    "predictor_label"
] = (
    logistic_univariate[
        "predictor"
    ]
    .replace({
        "responder_score":
            "Responder component"
    })
)


univariate_file = os.path.join(
    args.output_dir,
    "logistic_univariate.tsv"
)


logistic_univariate.to_csv(
    univariate_file,
    sep="\t",
    index=False
)


print(
    "Univariate logistic regression saved:",
    univariate_file
)

# ============================================================
# ОТЛАДКА: Проверка пропусков перед регрессией
# ============================================================


print(
    "\n" + "="*60
)

print(
    "DIAGNOSTICS: Checking missing values in patient_scores"
)

print(
    "="*60
)


check_columns = [
    "response",
    "response_binary",
    "responder_score",
    "aggressiveness",
    "aggressiveness_binary",
    "bone_marrow",
    "bone_marrow_binary",
    "lesion",
    "lesion_binary"
]


print(
    f"\nTotal patients: {len(patient_scores)}\n"
)


for col in check_columns:
    
    if col in patient_scores.columns:
        
        missing = patient_scores[col].isna().sum()
        unique_vals = patient_scores[col].nunique()
        
        print(
            f"{col:30} | "
            f"missing: {missing:3} | "
            f"unique: {unique_vals}"
        )
    
    else:
        
        print(
            f"{col:30} | NOT IN DATAFRAME"
        )


print(
    "\n" + "="*60 + "\n"
)


# Проверяем сколько полных случаев для каждой комбинации
print(
    "Complete cases for different model combinations:\n"
)


# Univariate
complete_univariate = patient_scores[
    ["response_binary", "responder_score"]
].dropna()

print(
    f"Univariate (response + responder_score): "
    f"{len(complete_univariate)} patients"
)


# Bivariate models
for covariate in ["aggressiveness_binary", "bone_marrow_binary", "lesion_binary"]:
    
    cols = ["response_binary", "responder_score", covariate]
    
    complete = patient_scores[cols].dropna()
    
    print(
        f"Bivariate (+ {covariate:25}): {len(complete)} patients"
    )


# Full model
full_cols = [
    "response_binary",
    "responder_score",
    "aggressiveness_binary",
    "bone_marrow_binary",
    "lesion_binary"
]

complete_full = patient_scores[full_cols].dropna()

print(
    f"Full model (all 5 variables):         {len(complete_full)} patients"
)


print(
    "\n" + "="*60 + "\n"
)

# ============================================================
# 20. MULTIVARIABLE LOGISTIC REGRESSION
# ============================================================

print(
    "\nRunning multivariable logistic regression..."
)


clinical_covariates = [
    "aggressiveness_binary",
    "bone_marrow_binary",
    "lesion_binary"
]


full_predictors = [
    "responder_score"
] + clinical_covariates


# ------------------------------------------------------------
# Check complete cases
# ------------------------------------------------------------

full_model_columns = [
    "response_binary"
] + full_predictors


full_model_data = (
    patient_scores[
        full_model_columns
    ]
    .dropna()
    .copy()
)


print(
    "\nVariables used in full multivariable model:"
)

print(
    full_model_columns
)


print(
    f"Complete cases for full model: "
    f"{len(full_model_data)}"
)


if len(full_model_data) > 0:

    print(
        "\nResponse distribution:"
    )

    print(
        full_model_data[
            "response_binary"
        ]
        .value_counts()
        .sort_index()
    )


# ------------------------------------------------------------
# Fit full model
# ------------------------------------------------------------

if len(full_model_data) < 5:

    raise ValueError(
        "\nFull multivariable logistic regression "
        "cannot be fitted.\n"
        f"Only {len(full_model_data)} complete cases "
        "remain after removing missing values.\n"
        "Check missing values in:\n"
        + "\n".join(
            full_model_columns
        )
    )


if (
    full_model_data["response_binary"]
    .nunique()
    < 2
):

    raise ValueError(
        "Full multivariable model contains "
        "only one response class."
    )


model_result = fit_logistic(
    patient_scores,
    "response_binary",
    full_predictors
)


model_result[
    "model"
] = "responder_score_adjusted"


# ============================================================
# MULTIPLE-TESTING CORRECTION
# ============================================================

model_result[
    "padj"
] = multipletests(
    model_result[
        "p_value"
    ],
    method="fdr_bh"
)[1]


# ============================================================
# HUMAN-READABLE LABELS
# ============================================================

def get_predictor_label(
    predictor
):

    label_map = {

        "responder_score":
            "Responder component",

        "aggressiveness_binary":
            "Aggressive disease",

        "bone_marrow_binary":
            "Bone marrow involvement",

        "lesion_binary":
            "Widespread lesion"

    }

    return label_map.get(
        predictor,
        predictor
    )


model_result[
    "predictor_label"
] = (
    model_result[
        "predictor"
    ]
    .apply(
        get_predictor_label
    )
)


# ============================================================
# SAVE
# ============================================================

multivariable_file = os.path.join(
    args.output_dir,
    "logistic_multivariable.tsv"
)


model_result.to_csv(
    multivariable_file,
    sep="\t",
    index=False
)


