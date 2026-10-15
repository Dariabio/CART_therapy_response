import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

from matplotlib.patches import Patch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    LeaveOneOut
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score
)


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


# ============================================================
# 21. ELASTIC NET RESPONSE SIGNATURE

# Elastic Net receives gene expression + response
# and selects genes itself.
# then evaluate the model using LOOCV.
# bootstrap is used to evaluate how stable
# gene selection is

# PARAMETERS

# Maximum number of genes entering Elastic Net.
# It is a computational pre-filter based only on variance.
# It is calculated separately inside each training fold.


MAX_CANDIDATE_GENES = 1000


# Elastic Net hyperparameters.
#
# C:
# inverse regularization strength.
#
# Smaller C = stronger regularization.
#
# l1_ratio:
#
# 0.0 -> pure L2
# 1.0 -> pure L1 (LASSO)
# between -> Elastic Net

C_GRID = [0.01, 0.1, 1.0, 10.0]

L1_RATIO_GRID = [0.1, 0.5, 0.9, 1.0]


# Number of bootstrap iterations.
#
# Start with 200 while developing/debugging.
#
# For the final analysis you can increase this to
# 500 or 1000.1

N_BOOTSTRAPS = 200


RANDOM_STATE = 42


# ============================================================
# 21.2 PREPARE PATIENT-LEVEL PRODUCT DATA
# ============================================================

elastic_meta = meta_product.copy()
elastic_meta["response_binary"] = (
    elastic_meta["response"]
    .map({
        "nonresponder": 0,
        "responder": 1
    })
)

if elastic_meta["response_binary"].isna().any():

    bad_values = (
        elastic_meta.loc[
            elastic_meta["response_binary"].isna(),
            "response"
        ]
        .unique()
        .tolist()
    )

    raise ValueError(
        "Unknown response values detected: "
        f"{bad_values}"
    )

# Expression matrix

elastic_expression = (
    vst_product
    .copy()
)

# Aggregate expression to patient level
# one patient = one observation.

patient_expression = (
    elastic_expression
    .join(
        elastic_meta[
            ["patient_id"]
        ],
        how="inner"
    )
    .groupby("patient_id")
    .mean()
)

# Patient-level response

patient_response = (
    elastic_meta[
        [
            "patient_id",
            "response_binary"
        ]
    ]
    .drop_duplicates(
        subset="patient_id"
    )
    .set_index("patient_id")
)

# Match patients

common_patients = (
    patient_expression.index
    .intersection(
        patient_response.index
    )
)

patient_expression = (
    patient_expression
    .loc[
        common_patients
    ]
)

patient_response = (
    patient_response
    .loc[
        common_patients
    ]
)

# Final checks

if len(common_patients) < 10:

    raise ValueError(
        "Too few patients available for "
        "Elastic Net analysis: "
        f"{len(common_patients)}"
    )


y = (
    patient_response[
        "response_binary"
    ]
    .astype(int)
    .values
)


X_df = (
    patient_expression
    .copy()
)


if len(np.unique(y)) != 2:

    raise ValueError(
        "Elastic Net requires both response classes."
    )

# FUNCTION: SELECT HIGH-VARIANCE GENES
# before Elastic Net we keep the most variable
# genes.

# This function is called separately on TRAINING data.
#
# Therefore the held-out patient never influences
# gene filtering

def select_variable_genes(
    X_train,
    max_genes=MAX_CANDIDATE_GENES
):

    variances = (
        X_train
        .var(axis=0)
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    variances = (
        variances[
            variances > 0
        ]
    )

    if len(variances) == 0:

        raise ValueError(
            "No genes with non-zero variance "
            "were found in training data."
        )

    n_genes = min(
        max_genes,
        len(variances)
    )

    selected_genes = (
        variances
        .nlargest(n_genes)
        .index
        .tolist()
    )

    return selected_genes

# FUNCTION: CREATE ELASTIC NET SEARCH


def make_elastic_net_search(
    random_state=RANDOM_STATE
):

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------
    #
    # StandardScaler:
    #
    # gene expression is standardized within training data.
    #
    # LogisticRegression:
    #
    # Elastic Net regularization.
    #
    # The scaler is inside the pipeline so that it is fitted
    # ONLY on the training part of each CV split.
    #

    pipeline = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=10000,
                random_state=random_state
            )
        )
    ])

    # Hyperparameter search

    parameter_grid = {

        "model__C":
            C_GRID,

        "model__l1_ratio":
            L1_RATIO_GRID

    }

    # Inner CV
    #
    # This CV is used ONLY to select:
    #
    #     C
    #     l1_ratio
    #
    # It is NOT the final performance estimate.
    #
    # The outer LOOCV below provides the honest
    # out-of-sample estimate.
    #

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )


    search = GridSearchCV(

        estimator=pipeline,

        param_grid=parameter_grid,

        scoring="roc_auc",

        cv=inner_cv,

        refit=True,

        n_jobs=-1,

        error_score="raise"

    )


    return search

# LOOCV

#
# For every patient:
#
#     25 patients -> training
#     1 patient    -> test
#
# Inside the 25 training patients:
#
#     select variable genes
#             |
#             v
#     inner CV
#             |
#             v
#     choose C + l1_ratio
#             |
#             v
#     fit Elastic Net
#
# Then the held-out patient is predicted.
#
# Therefore the held-out patient is NEVER used to:
#
#     - select genes
#     - scale genes
#     - select hyperparameters
#     - fit the model
#
# This is the important part that makes the evaluation
# genuinely out-of-sample.
# ============================================================

loo = LeaveOneOut()


loo_predictions = []

loo_coefficients = []


for fold_number, (
    train_indices,
    test_indices
) in enumerate(
    loo.split(X_df),
    start=1
):

    print(
        f"LOOCV fold "
        f"{fold_number}/{len(X_df)}"
    )


    # --------------------------------------------------------
    # Split patients
    # --------------------------------------------------------

    X_train_all = (
        X_df.iloc[
            train_indices
        ]
        .copy()
    )

    X_test_all = (
        X_df.iloc[
            test_indices
        ]
        .copy()
    )


    y_train = (
        y[
            train_indices
        ]
    )

    y_test = (
        y[
            test_indices
        ]
    )


    # --------------------------------------------------------
    # Select variable genes ONLY from training data
    # --------------------------------------------------------

    candidate_genes = (
        select_variable_genes(
            X_train_all
        )
    )


    X_train = (
        X_train_all[
            candidate_genes
        ]
    )

    X_test = (
        X_test_all[
            candidate_genes
        ]
    )


    # --------------------------------------------------------
    # Check that training data contains both classes
    # --------------------------------------------------------

    if len(np.unique(y_train)) < 2:

        raise ValueError(
            "LOOCV training fold contains "
            "only one response class."
        )


    # --------------------------------------------------------
    # Inner Elastic Net CV
    # --------------------------------------------------------

    search = make_elastic_net_search(
        random_state=(
            RANDOM_STATE +
            fold_number
        )
    )


    search.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Prediction for held-out patient
    # --------------------------------------------------------

    test_patient = X_test.index[0]


    probability = (
        search.predict_proba(
            X_test
        )[:, 1][0]
    )


    prediction = (
        search.predict(
            X_test
        )[0]
    )


    decision_score = (
        search.decision_function(
            X_test
        )[0]
    )


    loo_predictions.append({

        "patient_id":
            test_patient,

        "true_response":
            int(y_test[0]),

        "predicted_response":
            int(prediction),

        "predicted_probability":
            probability,

        "decision_score":
            decision_score,

        "best_C":
            search.best_params_[
                "model__C"
            ],

        "best_l1_ratio":
            search.best_params_[
                "model__l1_ratio"
            ],

        "n_candidate_genes":
            len(candidate_genes),

        "n_selected_genes":
            int(
                np.sum(
                    search.best_estimator_
                    .named_steps[
                        "model"
                    ]
                    .coef_[0] != 0
                )
            )

    })


    # --------------------------------------------------------
    # Save coefficients selected in this fold
    # --------------------------------------------------------

    fold_model = (
        search.best_estimator_
        .named_steps[
            "model"
        ]
    )


    fold_coefficients = (
        pd.Series(
            fold_model.coef_[0],
            index=candidate_genes,
            name=f"fold_{fold_number}"
        )
    )


    fold_coefficients = (
        fold_coefficients[
            fold_coefficients != 0
        ]
    )


    for gene, coefficient in (
        fold_coefficients.items()
    ):

        loo_coefficients.append({

            "fold":
                fold_number,

            "patient_left_out":
                test_patient,

            "gene":
                gene,

            "coefficient":
                coefficient

        })


# ============================================================
# 21.6 SAVE LOOCV RESULTS
# ============================================================


loo_predictions_df = pd.DataFrame(
    loo_predictions
)


loo_predictions_file = os.path.join(
    args.output_dir,
    "elastic_net_LOOCV_predictions.tsv"
)


loo_predictions_df.to_csv(
    loo_predictions_file,
    sep="\t",
    index=False
)


loo_coefficients_df = pd.DataFrame(
    loo_coefficients
)


loo_coefficients_file = os.path.join(
    args.output_dir,
    "elastic_net_LOOCV_coefficients.tsv"
)


loo_coefficients_df.to_csv(
    loo_coefficients_file,
    sep="\t",
    index=False
)


print(
    "\nLOOCV predictions saved:",
    loo_predictions_file
)


print(
    "LOOCV coefficients saved:",
    loo_coefficients_file
)


# ============================================================
# 21.7 LOOCV PERFORMANCE
# ============================================================
#
# IMPORTANT:
#
# These predictions are OUT-OF-SAMPLE predictions.
#
# Each patient's probability was generated by a model
# that did not use this patient during training.
# ============================================================


if (
    loo_predictions_df[
        "true_response"
    ].nunique()
    >= 2
):

    loo_auc = roc_auc_score(
        loo_predictions_df[
            "true_response"
        ],
        loo_predictions_df[
            "predicted_probability"
        ]
    )

else:

    loo_auc = np.nan


loo_accuracy = accuracy_score(
    loo_predictions_df[
        "true_response"
    ],
    loo_predictions_df[
        "predicted_response"
    ]
)


loo_balanced_accuracy = (
    balanced_accuracy_score(
        loo_predictions_df[
            "true_response"
        ],
        loo_predictions_df[
            "predicted_response"
        ]
    )
)


print(
    "\n" + "=" * 70
)

print(
    "LOOCV PERFORMANCE"
)

print(
    "=" * 70
)

print(
    f"LOOCV ROC-AUC: "
    f"{loo_auc:.4f}"
)

print(
    f"LOOCV accuracy: "
    f"{loo_accuracy:.4f}"
)

print(
    f"LOOCV balanced accuracy: "
    f"{loo_balanced_accuracy:.4f}"
)

print(
    "=" * 70
)


loo_metrics = pd.DataFrame({

    "metric": [
        "ROC_AUC",
        "accuracy",
        "balanced_accuracy"
    ],

    "value": [
        loo_auc,
        loo_accuracy,
        loo_balanced_accuracy
    ]

})


loo_metrics_file = os.path.join(
    args.output_dir,
    "elastic_net_LOOCV_metrics.tsv"
)


loo_metrics.to_csv(
    loo_metrics_file,
    sep="\t",
    index=False
)


# ============================================================
# 21.8 GENE SELECTION FREQUENCY DURING LOOCV
# ============================================================
#
# This tells us how often each gene was selected among
# the LOOCV training models.
#
# Example:
#
#     Gene A -> 24/26 folds
#     Gene B -> 17/26 folds
#     Gene C ->  2/26 folds
#
# A gene selected repeatedly is more stable than a gene
# appearing in only one fold.
# ============================================================


if not loo_coefficients_df.empty:

    gene_frequency = (
        loo_coefficients_df
        .groupby("gene")
        .agg(
            selection_count=(
                "fold",
                "nunique"
            )
        )
        .reset_index()
    )


    gene_frequency[
        "selection_frequency"
    ] = (
        gene_frequency[
            "selection_count"
        ]
        / len(X_df)
    )


    gene_frequency = (
        gene_frequency
        .sort_values(
            "selection_frequency",
            ascending=False
        )
    )

else:

    gene_frequency = pd.DataFrame(
        columns=[
            "gene",
            "selection_count",
            "selection_frequency"
        ]
    )


gene_frequency_file = os.path.join(
    args.output_dir,
    "elastic_net_LOOCV_gene_stability.tsv"
)


gene_frequency.to_csv(
    gene_frequency_file,
    sep="\t",
    index=False
)


print(
    "LOOCV gene stability saved:",
    gene_frequency_file
)


# ============================================================
# 21.9 BOOTSTRAP GENE STABILITY
# ============================================================
#
# Bootstrap asks a different question:
#
#     "If I repeatedly resample patients,
#      which genes are repeatedly selected?"
#
# This is NOT another estimate of external predictive
# performance.
#
# It is a measure of signature stability.
# ============================================================


print(
    "\n" + "-" * 70
)

print(
    f"Running bootstrap gene stability "
    f"({N_BOOTSTRAPS} iterations)..."
)

print(
    "-" * 70
)


rng = np.random.default_rng(
    RANDOM_STATE
)


bootstrap_selection = []


for bootstrap_number in range(
    1,
    N_BOOTSTRAPS + 1
):

    if (
        bootstrap_number == 1
        or bootstrap_number % 25 == 0
    ):

        print(
            f"Bootstrap "
            f"{bootstrap_number}/"
            f"{N_BOOTSTRAPS}"
        )


    # --------------------------------------------------------
    # Resample patients WITH replacement
    # --------------------------------------------------------

    bootstrap_indices = (
        rng.integers(
            low=0,
            high=len(X_df),
            size=len(X_df)
        )
    )


    # Need at least two response classes.

    bootstrap_y = (
        y[
            bootstrap_indices
        ]
    )


    if len(
        np.unique(
            bootstrap_y
        )
    ) < 2:

        continue


    X_bootstrap_all = (
        X_df.iloc[
            bootstrap_indices
        ]
        .copy()
    )


    # --------------------------------------------------------
    # Variable gene filtering
    # --------------------------------------------------------

    bootstrap_genes = (
        select_variable_genes(
            X_bootstrap_all
        )
    )


    X_bootstrap = (
        X_bootstrap_all[
            bootstrap_genes
        ]
    )


    # --------------------------------------------------------
    # Fit Elastic Net with internal CV
    # --------------------------------------------------------

    try:

        bootstrap_search = (
            make_elastic_net_search(
                random_state=(
                    RANDOM_STATE +
                    bootstrap_number
                )
            )
        )


        bootstrap_search.fit(
            X_bootstrap,
            bootstrap_y
        )


    except Exception as error:

        print(
            f"WARNING: bootstrap "
            f"{bootstrap_number} failed: "
            f"{error}"
        )

        continue


    bootstrap_model = (
        bootstrap_search
        .best_estimator_
        .named_steps[
            "model"
        ]
    )


    bootstrap_coefficients = (
        pd.Series(
            bootstrap_model.coef_[0],
            index=bootstrap_genes
        )
    )


    bootstrap_coefficients = (
        bootstrap_coefficients[
            bootstrap_coefficients != 0
        ]
    )


    for gene, coefficient in (
        bootstrap_coefficients.items()
    ):

        bootstrap_selection.append({

            "bootstrap":
                bootstrap_number,

            "gene":
                gene,

            "coefficient":
                coefficient,

            "C":
                bootstrap_search
                .best_params_[
                    "model__C"
                ],

            "l1_ratio":
                bootstrap_search
                .best_params_[
                    "model__l1_ratio"
                ]

        })


bootstrap_selection_df = pd.DataFrame(
    bootstrap_selection
)


bootstrap_coefficients_file = os.path.join(
    args.output_dir,
    "elastic_net_bootstrap_coefficients.tsv"
)


bootstrap_selection_df.to_csv(
    bootstrap_coefficients_file,
    sep="\t",
    index=False
)


print(
    "Bootstrap coefficients saved:",
    bootstrap_coefficients_file
)


# ============================================================
# 21.10 BOOTSTRAP SELECTION FREQUENCY
# ============================================================


if not bootstrap_selection_df.empty:

    n_successful_bootstraps = (
        bootstrap_selection_df[
            "bootstrap"
        ]
        .nunique()
    )


    bootstrap_stability = (
        bootstrap_selection_df
        .groupby("gene")
        .agg(

            selection_count=(
                "bootstrap",
                "nunique"
            ),

            mean_coefficient=(
                "coefficient",
                "mean"
            ),

            median_coefficient=(
                "coefficient",
                "median"
            ),

            coefficient_sd=(
                "coefficient",
                "std"
            )

        )
        .reset_index()
    )


    bootstrap_stability[
        "selection_frequency"
    ] = (
        bootstrap_stability[
            "selection_count"
        ]
        /
        n_successful_bootstraps
    )


    bootstrap_stability = (
        bootstrap_stability
        .sort_values(
            "selection_frequency",
            ascending=False
        )
    )


else:

    n_successful_bootstraps = 0

    bootstrap_stability = pd.DataFrame(
        columns=[
            "gene",
            "selection_count",
            "mean_coefficient",
            "median_coefficient",
            "coefficient_sd",
            "selection_frequency"
        ]
    )


bootstrap_stability_file = os.path.join(
    args.output_dir,
    "elastic_net_bootstrap_gene_stability.tsv"
)


bootstrap_stability.to_csv(
    bootstrap_stability_file,
    sep="\t",
    index=False
)


print(
    "Bootstrap gene stability saved:",
    bootstrap_stability_file
)


print(
    f"\nSuccessful bootstrap iterations: "
    f"{n_successful_bootstraps}/"
    f"{N_BOOTSTRAPS}"
)


# ============================================================
# 21.11 FINAL ELASTIC NET MODEL
# ============================================================
#
# Now we use ALL discovery patients.
#
# This model is NOT used to estimate performance.
#
# It is used to construct the final signature that will
# subsequently be LOCKED and tested in an independent cohort.
#
# This distinction is important:
#
#     LOOCV -> performance estimate
#
#     final model -> signature construction
# ============================================================


print(
    "\n" + "-" * 70
)

print(
    "Fitting FINAL Elastic Net model..."
)

print(
    "-" * 70
)


final_candidate_genes = (
    select_variable_genes(
        X_df
    )
)


X_final = (
    X_df[
        final_candidate_genes
    ]
)


final_search = (
    make_elastic_net_search(
        random_state=RANDOM_STATE
    )
)


final_search.fit(
    X_final,
    y
)


final_model = (
    final_search
    .best_estimator_
    .named_steps[
        "model"
    ]
)


final_coefficients = (
    pd.Series(
        final_model.coef_[0],
        index=final_candidate_genes
    )
)


final_coefficients = (
    final_coefficients[
        final_coefficients != 0
    ]
    .sort_values(
        key=np.abs,
        ascending=False
    )
)


print(
    "\nFinal Elastic Net parameters:"
)


print(
    f"C = "
    f"{final_search.best_params_['model__C']}"
)


print(
    f"l1_ratio = "
    f"{final_search.best_params_['model__l1_ratio']}"
)


print(
    f"Selected genes = "
    f"{len(final_coefficients)}"
)


# ============================================================
# 21.12 FINAL SIGNATURE TABLE
# ============================================================
#
# Positive coefficient:
#
# higher expression contributes toward responder prediction.
#
# Negative coefficient:
#
# higher expression contributes toward nonresponder prediction.
#
# IMPORTANT:
#
# These are MODEL coefficients, not DESeq2 log2FC values.
# ============================================================


final_signature = pd.DataFrame({

    "gene":
        final_coefficients.index,

    "coefficient":
        final_coefficients.values

})


final_signature[
    "direction"
] = np.where(

    final_signature[
        "coefficient"
    ] > 0,

    "responder",

    "nonresponder"

)


# Add LOOCV stability if available.

if not gene_frequency.empty:

    final_signature = (
        final_signature
        .merge(
            gene_frequency[
                [
                    "gene",
                    "selection_count",
                    "selection_frequency"
                ]
            ],
            on="gene",
            how="left"
        )
    )

else:

    final_signature[
        "selection_count"
    ] = np.nan

    final_signature[
        "selection_frequency"
    ] = np.nan


# Add bootstrap stability.

if not bootstrap_stability.empty:

    final_signature = (
        final_signature
        .merge(
            bootstrap_stability[
                [
                    "gene",
                    "selection_count",
                    "selection_frequency",
                    "mean_coefficient",
                    "median_coefficient",
                    "coefficient_sd"
                ]
            ],
            on="gene",
            how="left",
            suffixes=(
                "_LOOCV",
                "_bootstrap"
            )
        )
    )

else:

    final_signature[
        "selection_count_bootstrap"
    ] = np.nan

    final_signature[
        "selection_frequency_bootstrap"
    ] = np.nan

    final_signature[
        "mean_coefficient"
    ] = np.nan

    final_signature[
        "median_coefficient"
    ] = np.nan

    final_signature[
        "coefficient_sd"
    ] = np.nan


# ------------------------------------------------------------
# Save final signature
# ------------------------------------------------------------


elastic_net_signature_file = os.path.join(
    args.output_dir,
    "CAR_T_response_signature_ElasticNet.tsv"
)


final_signature.to_csv(
    elastic_net_signature_file,
    sep="\t",
    index=False
)


print(
    "\nFinal Elastic Net signature saved:",
    elastic_net_signature_file
)


# ============================================================
# 21.13 SAVE FINAL MODEL PARAMETERS
# ============================================================
#
# These parameters are required to apply the signature
# to an external cohort.
#
# The external cohort must use:
#
#     same genes
#     same scaling parameters
#     same coefficients
# ============================================================


final_scaler = (
    final_search
    .best_estimator_
    .named_steps[
        "scaler"
    ]
)


final_model_parameters = pd.DataFrame({

    "gene":
        final_candidate_genes,

    "mean":
        final_scaler.mean_,

    "scale":
        final_scaler.scale_,

    "coefficient":
        final_model.coef_[0]

})


final_model_parameters_file = os.path.join(
    args.output_dir,
    "CAR_T_response_ElasticNet_model_parameters.tsv"
)


final_model_parameters.to_csv(
    final_model_parameters_file,
    sep="\t",
    index=False
)


print(
    "Final model parameters saved:",
    final_model_parameters_file
)


# ============================================================
# 21.14 FINAL DISCOVERY-COHORT SIGNATURE SCORES
# ============================================================
#
# We calculate the model's linear predictor:
#
#     score =
#         intercept
#         +
#         sum(
#             standardized_expression_gene
#             *
#             coefficient_gene
#         )
#
# Positive score -> more responder-like according to model.
#
# Negative score -> more nonresponder-like according to model.
#
# These scores are NOT used for the LOOCV performance estimate.
# ============================================================


final_X = (
    X_df[
        final_candidate_genes
    ]
)


final_probabilities = (
    final_search
    .predict_proba(
        final_X
    )[:, 1]
)


final_decision_scores = (
    final_search
    .decision_function(
        final_X
    )
)


elastic_net_patient_scores = (
    pd.DataFrame({

        "patient_id":
            X_df.index,

        "response_binary":
            y,

        "ElasticNet_score":
            final_decision_scores,

        "ElasticNet_probability":
            final_probabilities

    })
)


elastic_net_patient_scores_file = os.path.join(
    args.output_dir,
    "CAR_T_response_ElasticNet_patient_scores.tsv"
)


elastic_net_patient_scores.to_csv(
    elastic_net_patient_scores_file,
    sep="\t",
    index=False
)


print(
    "Final patient scores saved:",
    elastic_net_patient_scores_file
)


# ============================================================
# 21.15 SUMMARY
# ============================================================


print(
    "\n" + "=" * 70
)

print(
    "ELASTIC NET SIGNATURE SUMMARY"
)

print(
    "=" * 70
)

print(
    f"Discovery patients: "
    f"{len(X_df)}"
)

print(
    f"Candidate genes: "
    f"{len(final_candidate_genes)}"
)

print(
    f"Final selected genes: "
    f"{len(final_signature)}"
)

print(
    f"LOOCV ROC-AUC: "
    f"{loo_auc:.4f}"
)

print(
    f"LOOCV balanced accuracy: "
    f"{loo_balanced_accuracy:.4f}"
)

print(
    f"Bootstrap iterations: "
    f"{N_BOOTSTRAPS}"
)

print(
    f"Successful bootstrap iterations: "
    f"{n_successful_bootstraps}"
)

print(
    "=" * 70
)


