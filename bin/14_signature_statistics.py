#!/usr/bin/env python3

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Statistical analysis of response-associated "
        "ssGSEA pathways using univariate and "
        "multivariable logistic regression."
    )
)

parser.add_argument(
    "--metadata",
    required=True,
    help="Metadata table"
)

parser.add_argument(
    "--scores_all",
    required=True,
    help="ssGSEA scores matrix: samples x pathways"
)


parser.add_argument(
    "--ssgsea_stat_result",
    required=True,
    help=(
        "Ranked GSEA response results "
        "containing Term and FDR q-val"
    )
)

parser.add_argument(
    "--output_dir",
    required=True,
    help="Output directory"
)

parser.add_argument(
    "--timepoint",
    default="product",
    help="Timepoint used for analysis (default: product)"
)

args = parser.parse_args()


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    args.output_dir,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\nLoading metadata...")

meta = pd.read_csv(
    args.metadata,
    sep="\t",
    index_col=0
)

print(
    f"Metadata shape: {meta.shape}"
)


print("\nLoading ssGSEA scores...")

scores = pd.read_csv(
    args.scores_all,
    index_col=0
)

print(
    f"ssGSEA score matrix: {scores.shape}"
)


print("\nLoading GSEA response results...")

gsea_response = pd.read_csv(
    args.ssgsea_stat_result, sep='\t'
)

print(
    f"GSEA response table: {gsea_response.shape}"
)


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_meta_columns = [
    "patient_id",
    "timepoint",
    "response",
    "bone_marrow",
    "aggressiveness",
    "lesion"
]


missing_meta_columns = [
    col
    for col in required_meta_columns
    if col not in meta.columns
]


if missing_meta_columns:

    raise ValueError(
        "Metadata is missing required columns: "
        f"{missing_meta_columns}"
    )


required_gsea_columns = [
    "subpopulation",
    "FDR"
]


missing_gsea_columns = [
    col
    for col in required_gsea_columns
    if col not in gsea_response.columns
]


if missing_gsea_columns:

    raise ValueError(
        "GSEA response file is missing required columns: "
        f"{missing_gsea_columns}"
    )


# ============================================================
# 3. KEEP PRODUCT SAMPLES
# ============================================================

meta_product = meta.loc[
    (
        meta["timepoint"].astype(str)
        == str(args.timepoint)
    )
    &
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
        f"No {args.timepoint} samples with "
        "responder/nonresponder labels found."
    )


print(
    f"\nMetadata samples at '{args.timepoint}': "
    f"{len(meta_product)}"
)


# ============================================================
# 4. MATCH SAMPLE IDs
# ============================================================

common_samples = (
    meta_product.index
    .intersection(
        scores.index
    )
)


if len(common_samples) == 0:

    raise ValueError(
        "No common sample IDs between metadata "
        "and ssGSEA scores."
    )


meta_product = meta_product.loc[
    common_samples
].copy()


scores_product = scores.loc[
    common_samples
].copy()


print(
    f"Matched product samples: "
    f"{len(common_samples)}"
)


# ============================================================
# 5. RESPONSE ENCODING
# ============================================================

meta_product["response_binary"] = (
    meta_product["response"]
    .map({
        "nonresponder": 0,
        "responder": 1
    })
)


# ============================================================
# 6. PATIENT-LEVEL DATA
# ============================================================

print(
    "\nAggregating ssGSEA scores to patient level..."
)


scores_patient = (
    scores_product
    .groupby(
        meta_product["patient_id"]
    )
    .mean()
)


meta_patient = (
    meta_product
    .groupby(
        "patient_id"
    )
    .first()
    .copy()
)


common_patients = (
    meta_patient.index
    .intersection(
        scores_patient.index
    )
)


meta_patient = meta_patient.loc[
    common_patients
].copy()


scores_patient = scores_patient.loc[
    common_patients
].copy()


print(
    f"Patients included: "
    f"{len(common_patients)}"
)


if len(common_patients) < 10:

    raise ValueError(
        "Very few patients are available for "
        "patient-level logistic regression."
    )


# ============================================================
# 7. PATIENT-LEVEL RESPONSE
# ============================================================

meta_patient["response_binary"] = (
    meta_patient["response"]
    .map({
        "nonresponder": 0,
        "responder": 1
    })
)


# ============================================================
# 8. ENCODE CLINICAL COVARIATES
# ============================================================

meta_patient["bone_marrow_bin"] = (
    meta_patient["bone_marrow"]
    .map({
        "bone_marrow_negative": 0,
        "bone_marrow_positive": 1,
        "negative": 0,
        "positive": 1
    })
)


meta_patient["aggressiveness_bin"] = (
    meta_patient["aggressiveness"]
    .map({
        "indolent": 0,
        "aggressive": 1
    })
)


meta_patient["lesion_bin"] = (
    meta_patient["lesion"]
    .map({
        "localized": 0,
        "widespread": 1
    })
)


# ============================================================
# 9. CLINICAL DIAGNOSTICS
# ============================================================

print(
    "\nClinical variable diagnostics:"
)

clinical_binary = [
    "response_binary",
    "bone_marrow_bin",
    "aggressiveness_bin",
    "lesion_bin"
]

for col in clinical_binary:

    print(
        f"\n{col}:"
    )

    print(
        meta_patient[col]
        .value_counts(
            dropna=False
        )
        .sort_index()
    )


# ============================================================
# 10. FIND SIGNIFICANT RESPONSE GSEA PATHWAYS
# ============================================================

response_gsea = (
    gsea_response[
        gsea_response["FDR"] < 0.15
    ]
    .copy()
)


response_terms = (
    response_gsea["subpopulation"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


print(
    "\nSignificant response GSEA pathways:",
    len(response_terms)
)


if len(response_terms) == 0:

    raise ValueError(
        "No GSEA response pathways with "
        "FDR < 0.1 were found."
    )


# ============================================================
# 11. MATCH GSEA PATHWAYS TO ssGSEA MATRIX
# ============================================================

available_terms = [
    term
    for term in response_terms
    if term in scores_patient.columns
]


missing_terms = [
    term
    for term in response_terms
    if term not in scores_patient.columns
]


print(
    f"Response pathways found in ssGSEA matrix: "
    f"{len(available_terms)}"
)


if missing_terms:

    print(
        "\nGSEA pathways not found in ssGSEA scores:"
    )

    for term in missing_terms:

        print(
            f"  - {term}"
        )


if len(available_terms) == 0:

    raise ValueError(
        "None of the significant GSEA response "
        "pathways were found in scores_all.csv."
    )


# ============================================================
# 12. SAVE PATHWAYS USED
# ============================================================

selected_pathways = pd.DataFrame({
    "pathway": available_terms
})


selected_pathways.to_csv(
    os.path.join(
        args.output_dir,
        "response_gsea_pathways_used.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 13. PATHWAY SCORE DIAGNOSTICS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PATHWAY SCORE DIAGNOSTICS"
)

print(
    "=" * 80
)


score_diagnostics = []


for pathway in available_terms:

    x = pd.to_numeric(
        scores_patient[pathway],
        errors="coerce"
    ).dropna()


    score_diagnostics.append({

        "pathway": pathway,

        "n": len(x),

        "mean": x.mean(),

        "sd": x.std(ddof=1),

        "median": x.median(),

        "min": x.min(),

        "max": x.max(),

        "range": x.max() - x.min(),

        "n_unique": x.nunique(),

        "iqr": (
            x.quantile(0.75)
            -
            x.quantile(0.25)
        )

    })


score_diagnostics = pd.DataFrame(
    score_diagnostics
)


score_diagnostics.to_csv(
    os.path.join(
        args.output_dir,
        "ssgsea_score_diagnostics.tsv"
    ),
    sep="\t",
    index=False
)


print(
    score_diagnostics.to_string(
        index=False
    )
)


# ============================================================
# 14. CHECK IDENTICAL / HIGHLY CORRELATED PATHWAYS
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PATHWAY CORRELATION DIAGNOSTICS"
)

print(
    "=" * 80
)


correlation_matrix = (
    scores_patient[
        available_terms
    ]
    .corr(
        method="pearson"
    )
)


correlation_matrix.to_csv(
    os.path.join(
        args.output_dir,
        "ssgsea_pathway_correlation_matrix.tsv"
    ),
    sep="\t"
)


# ------------------------------------------------------------
# Find highly correlated pairs
# ------------------------------------------------------------

high_corr_pairs = []


for i in range(
    len(available_terms)
):

    for j in range(
        i + 1,
        len(available_terms)
    ):

        pathway_1 = available_terms[i]
        pathway_2 = available_terms[j]

        x = scores_patient[
            pathway_1
        ]

        y = scores_patient[
            pathway_2
        ]

        valid = (
            x.notna()
            &
            y.notna()
        )


        if valid.sum() < 3:

            continue


        corr = x[valid].corr(
            y[valid],
            method="pearson"
        )


        if abs(corr) >= 0.95:

            high_corr_pairs.append({

                "pathway_1": pathway_1,

                "pathway_2": pathway_2,

                "pearson_r": corr

            })


high_corr_pairs = pd.DataFrame(
    high_corr_pairs
)


high_corr_pairs.to_csv(
    os.path.join(
        args.output_dir,
        "highly_correlated_pathways.tsv"
    ),
    sep="\t",
    index=False
)


if not high_corr_pairs.empty:

    print(
        "\nHighly correlated pathway pairs "
        "(|r| >= 0.95):"
    )

    print(
        high_corr_pairs
        .sort_values(
            "pearson_r",
            key=lambda s: s.abs(),
            ascending=False
        )
        .to_string(
            index=False
        )
    )

else:

    print(
        "\nNo pathway pairs with |r| >= 0.95."
    )


# ============================================================
# 15. STANDARDIZE PATHWAY SCORES
# ============================================================
#
# IMPORTANT:
#
# Logistic regression will use:
#
#     pathway_z
#
# rather than the raw ssGSEA score.
#
# Therefore OR means:
#
#     change in odds of response
#     per 1 SD increase in pathway score.
#
# This avoids extremely large OR simply because
# the raw score has a small numerical scale.
# ============================================================

scores_patient_z = pd.DataFrame(
    index=scores_patient.index
)


for pathway in available_terms:

    mean = scores_patient[
        pathway
    ].mean()

    sd = scores_patient[
        pathway
    ].std(
        ddof=1
    )


    if pd.isna(sd) or sd == 0:

        print(
            f"WARNING: zero/undefined SD for "
            f"{pathway}; skipping standardization."
        )

        scores_patient_z[
            pathway
        ] = np.nan

    else:

        scores_patient_z[
            pathway
        ] = (
            scores_patient[pathway]
            - mean
        ) / sd


scores_patient_z.columns = [
    f"{pathway}__z"
    for pathway in scores_patient_z.columns
]


z_pathway_names = (
    scores_patient_z.columns
    .tolist()
)


# Save z-score matrix
scores_patient_z.to_csv(
    os.path.join(
        args.output_dir,
        "patient_ssgsea_z_scores.tsv"
    ),
    sep="\t"
)


# ============================================================
# 16. LOGISTIC REGRESSION FUNCTION
# ============================================================

def run_logistic(
    df,
    predictor,
    covariates=None,
    pathway_name=None
):

    if covariates is None:

        covariates = []


    columns = [
        "response_binary",
        predictor
    ] + covariates


    tmp = (
        df[columns]
        .dropna()
        .copy()
    )


    if len(tmp) < 5:

        return None, (
            "fewer than 5 complete observations"
        )


    if (
        tmp["response_binary"]
        .nunique()
        < 2
    ):

        return None, (
            "only one response class"
        )


    if (
        tmp[predictor]
        .nunique()
        < 2
    ):

        return None, (
            "predictor has no variation"
        )


    X = (
        tmp[
            [predictor] + covariates
        ]
        .astype(float)
    )


    X = sm.add_constant(
        X,
        has_constant="add"
    )


    y = (
        tmp[
            "response_binary"
        ]
        .astype(float)
    )


    # --------------------------------------------------------
    # Check event counts
    # --------------------------------------------------------

    n_responder = int(
        (y == 1).sum()
    )

    n_nonresponder = int(
        (y == 0).sum()
    )


    # --------------------------------------------------------
    # Fit model
    # --------------------------------------------------------

    try:

        with warnings.catch_warnings(
            record=True
        ) as caught_warnings:

            warnings.simplefilter(
                "always"
            )

            model = sm.Logit(
                y,
                X
            ).fit(
                disp=False,
                maxiter=200
            )


    except Exception as error:

        return None, str(error)


    # --------------------------------------------------------
    # Convergence
    # --------------------------------------------------------

    converged = (
        model.mle_retvals
        .get(
            "converged",
            np.nan
        )
    )


    # --------------------------------------------------------
    # Detect suspiciously extreme fitted probabilities
    # --------------------------------------------------------

    fitted_probabilities = (
        model.predict(X)
    )


    extreme_probability_fraction = (
        (
            (fitted_probabilities < 0.01)
            |
            (fitted_probabilities > 0.99)
        )
        .mean()
    )


    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    result_rows = []


    confidence_intervals = (
        model.conf_int()
    )


    for variable in (
        [predictor] + covariates
    ):

        beta = (
            model.params[
                variable
            ]
        )


        ci_low = (
            confidence_intervals
            .loc[
                variable,
                0
            ]
        )


        ci_high = (
            confidence_intervals
            .loc[
                variable,
                1
            ]
        )


        result_rows.append({

            "pathway": (
                pathway_name
                if pathway_name is not None
                else predictor
            ),

            "predictor": variable,

            "n": len(tmp),

            "n_responder":
                n_responder,

            "n_nonresponder":
                n_nonresponder,

            "coefficient": beta,

            "odds_ratio": np.exp(beta),

            "ci95_low": np.exp(
                ci_low
            ),

            "ci95_high": np.exp(
                ci_high
            ),

            "p_value":
                model.pvalues[
                    variable
                ],

            "converged":
                converged,

            "extreme_probability_fraction":
                extreme_probability_fraction

        })


    return (
        pd.DataFrame(
            result_rows
        ),
        None
    )


# ============================================================
# 17. PREPARE REGRESSION DATASET
# ============================================================

regression_df = meta_patient.copy()


for pathway in available_terms:

    z_name = f"{pathway}__z"

    if z_name in scores_patient_z.columns:

        regression_df[
            z_name
        ] = scores_patient_z[
            z_name
        ]


regression_df.to_csv(
    os.path.join(
        args.output_dir,
        "patient_ssgsea_regression_data.tsv"
    ),
    sep="\t"
)


# ============================================================
# 18. UNIVARIATE LOGISTIC REGRESSION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "UNIVARIATE LOGISTIC REGRESSION"
)

print(
    "=" * 80
)


univariate_results = []

univariate_failed = []


for pathway in available_terms:

    predictor = (
        f"{pathway}__z"
    )


    if predictor not in regression_df:

        univariate_failed.append({

            "pathway": pathway,

            "reason":
                "standardized score unavailable"

        })

        continue


    result, error = run_logistic(
        regression_df,
        predictor=predictor,
        pathway_name=pathway
    )


    if result is not None:

        # Keep only pathway coefficient
        result = result[
            result["predictor"]
            == predictor
        ].copy()


        univariate_results.append(
            result
        )


    else:

        univariate_failed.append({

            "pathway": pathway,

            "reason": error

        })


if len(univariate_results) == 0:

    raise RuntimeError(
        "No univariate logistic models "
        "could be fitted."
    )


logistic_univariate = pd.concat(
    univariate_results,
    ignore_index=True
)


# ------------------------------------------------------------
# Rename predictor
# ------------------------------------------------------------

logistic_univariate[
    "predictor"
] = (
    logistic_univariate[
        "pathway"
    ]
)


logistic_univariate[
    "predictor_label"
] = (
    logistic_univariate[
        "pathway"
    ]
)


# ------------------------------------------------------------
# FDR across PATHWAYS only
# ------------------------------------------------------------

logistic_univariate[
    "padj"
] = multipletests(
    logistic_univariate[
        "p_value"
    ],
    method="fdr_bh"
)[1]


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

logistic_univariate = (
    logistic_univariate
    .sort_values(
        "p_value"
    )
)


logistic_univariate.to_csv(
    os.path.join(
        args.output_dir,
        "logistic_univariate.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 19. MULTIVARIABLE LOGISTIC REGRESSION
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "MULTIVARIABLE LOGISTIC REGRESSION"
)

print(
    "=" * 80
)


clinical_covariates = [
    "aggressiveness_bin",
    "bone_marrow_bin",
    "lesion_bin"
]


multivariable_results = []

multivariable_failed = []


for pathway in available_terms:

    predictor = (
        f"{pathway}__z"
    )


    if predictor not in regression_df:

        multivariable_failed.append({

            "pathway": pathway,

            "reason":
                "standardized score unavailable"

        })

        continue


    result, error = run_logistic(
        regression_df,
        predictor=predictor,
        covariates=clinical_covariates,
        pathway_name=pathway
    )


    if result is not None:

        multivariable_results.append(
            result
        )


    else:

        multivariable_failed.append({

            "pathway": pathway,

            "reason": error

        })


if len(multivariable_results) == 0:

    raise RuntimeError(
        "No multivariable logistic models "
        "could be fitted."
    )


logistic_multivariable = pd.concat(
    multivariable_results,
    ignore_index=True
)


# ============================================================
# 20. HUMAN-READABLE PREDICTOR LABELS
# ============================================================

predictor_labels = {

    "aggressiveness_bin":
        "Aggressive vs indolent",

    "bone_marrow_bin":
        "Bone marrow positive vs negative",

    "lesion_bin":
        "Widespread vs localized"
}


logistic_multivariable[
    "predictor_label"
] = (
    logistic_multivariable[
        "predictor"
    ]
    .replace(
        predictor_labels
    )
)


# For pathway predictor itself:
logistic_multivariable.loc[
    logistic_multivariable[
        "predictor"
    ].str.endswith("__z"),
    "predictor_label"
] = (
    logistic_multivariable.loc[
        logistic_multivariable[
            "predictor"
        ].str.endswith("__z"),
        "pathway"
    ]
)


# ============================================================
# 21. SEPARATE FDR CORRECTION BY TEST FAMILY
# ============================================================

# IMPORTANT:
#
# Pathway effects are one hypothesis family:
#   pathway score coefficients across pathways
#
# Clinical effects are different:
#   aggressiveness across models
#   bone marrow across models
#   lesion across models
#
# Therefore we do NOT mix all 52 p-values together.


logistic_multivariable[
    "padj"
] = np.nan


# ------------------------------------------------------------
# 21.1 Pathway FDR
# ------------------------------------------------------------

is_pathway_test = (
    logistic_multivariable[
        "predictor"
    ].str.endswith("__z")
)


logistic_multivariable.loc[
    is_pathway_test,
    "padj"
] = multipletests(
    logistic_multivariable.loc[
        is_pathway_test,
        "p_value"
    ],
    method="fdr_bh"
)[1]


# ------------------------------------------------------------
# 21.2 Clinical FDR
# ------------------------------------------------------------

for clinical_predictor in clinical_covariates:

    mask = (
        logistic_multivariable[
            "predictor"
        ]
        ==
        clinical_predictor
    )


    if mask.sum() > 0:

        logistic_multivariable.loc[
            mask,
            "padj"
        ] = multipletests(
            logistic_multivariable.loc[
                mask,
                "p_value"
            ],
            method="fdr_bh"
        )[1]


# ============================================================
# 22. SAVE MULTIVARIABLE RESULTS
# ============================================================

logistic_multivariable = (
    logistic_multivariable
    .sort_values(
        [
            "pathway",
            "p_value"
        ]
    )
)


logistic_multivariable.to_csv(
    os.path.join(
        args.output_dir,
        "logistic_multivariable.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 23. SAVE FAILED MODELS
# ============================================================

if len(univariate_failed) > 0:

    pd.DataFrame(
        univariate_failed
    ).to_csv(
        os.path.join(
            args.output_dir,
            "logistic_univariate_failed.tsv"
        ),
        sep="\t",
        index=False
    )


if len(multivariable_failed) > 0:

    pd.DataFrame(
        multivariable_failed
    ).to_csv(
        os.path.join(
            args.output_dir,
            "logistic_multivariable_failed.tsv"
        ),
        sep="\t",
        index=False
    )


# ============================================================
# 24. MODEL SUMMARY
# ============================================================

model_summary = []


for pathway in available_terms:

    pathway_rows = logistic_multivariable[
        logistic_multivariable[
            "pathway"
        ]
        ==
        pathway
    ]


    if pathway_rows.empty:

        continue


    pathway_row = pathway_rows[
        pathway_rows[
            "predictor"
        ]
        ==
        f"{pathway}__z"
    ]


    if pathway_row.empty:

        continue


    pathway_row = (
        pathway_row
        .iloc[0]
    )


    model_summary.append({

        "pathway":
            pathway,

        "n":
            pathway_row["n"],

        "pathway_OR_per_SD":
            pathway_row["odds_ratio"],

        "pathway_CI95_low":
            pathway_row["ci95_low"],

        "pathway_CI95_high":
            pathway_row["ci95_high"],

        "pathway_p":
            pathway_row["p_value"],

        "pathway_padj":
            pathway_row["padj"],

        "converged":
            pathway_row["converged"],

        "extreme_probability_fraction":
            pathway_row[
                "extreme_probability_fraction"
            ]

    })


model_summary = pd.DataFrame(
    model_summary
)


model_summary.to_csv(
    os.path.join(
        args.output_dir,
        "multivariable_pathway_summary.tsv"
    ),
    sep="\t",
    index=False
)


