
# Longitudinal ssGSEA analysis for multiple comparisons
#
# Comparisons:
#   response
#   aggressiveness
#   bone_marrow
#   lesion
#
# For each comparison:
#   1. Mixed-effects model
#   2. Interaction analysis
#   3. Delta analysis
#   4. Delta effect heatmap
#   5. Forest plots
#   6. Trajectory plots
#   7. Final summary

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf

from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu

#  Arguments

parser = argparse.ArgumentParser(
    description="Longitudinal ssGSEA analysis for multiple comparisons"
)

parser.add_argument(
    "--comparison",
    required=True,
    choices=[
        "group_analysis_1",
        "response",
        "aggressiveness",
        "bone_marrow",
        "lesion"
    ]
)

parser.add_argument(
    "--metadata",
    required=True
)

parser.add_argument(
    "--scores_all",
    required=True
)

parser.add_argument(
    "--output_dir",
    required=True
)

args = parser.parse_args()

comparison = args.comparison

os.makedirs(
    args.output_dir,
    exist_ok=True
)

# COMPARISON CONFIGURATION

COMPARISONS = {

    "group_analysis_1": {
            "column": "group_analysis_1",
            "positive": "responder",
            "negative": "nonresponder"
        },

    "response": {
        "column": "response",
        "positive": "responder",
        "negative": "nonresponder"
    },

    "aggressiveness": {
        "column": "aggressiveness",
        "positive": "aggressive",
        "negative": "indolent"
    },

    "bone_marrow": {
        "column": "bone_marrow",
        "positive": "bone_marrow_positive",
        "negative": "bone_marrow_negative"
    },

    "lesion": {
        "column": "lesion",
        "positive": "widespread",
        "negative": "localized"
    }
}


config = COMPARISONS[comparison]

comparison_column = config["column"]
positive = config["positive"]
negative = config["negative"]

# CONSTANTS

patient_col = "patient_id"
timepoint_col = "timepoint"

product = "product"
d7 = "d7 after"
d28 = "d28 after"

timepoint_order = [
    product,
    d7,
    d28
]

group_order = [
    negative,
    positive
]

# LOAD DATA

scores_all = pd.read_csv(
    args.scores_all,
    sep=",",
    index_col=0
)

meta_all = pd.read_csv(
    args.metadata,
    sep="\t",
    index_col=0
)

# MATCH SAMPLES

common_samples = (
    scores_all.index
    .intersection(
        meta_all.index
    )
)

scores = scores_all.loc[
    common_samples
].copy()

meta = meta_all.loc[
    common_samples
].copy()


# KEEP ONLY TWO GROUPS FOR THIS COMPARISON

comparison_mask = meta[
    comparison_column
].isin(
    [
        positive,
        negative
    ]
)

meta = meta.loc[
    comparison_mask
].copy()

scores = scores.loc[
    meta.index
].copy()


print("\nSamples used:")
print(
    meta[
        comparison_column
    ].value_counts()
)


# SIGNATURES
# ============================================================

signature_cols = list(
    scores.columns
)


# ============================================================
# PREPARE DATA
# ============================================================

df = meta[
    [
        patient_col,
        timepoint_col,
        comparison_column
    ]
].join(
    scores
)


df[timepoint_col] = (
    df[timepoint_col]
    .astype(str)
    .str.strip()
)


df = df[
    df[timepoint_col].isin(
        timepoint_order
    )
].copy()


df["group"] = df[
    comparison_column
]


df = df[
    df["group"].isin(
        group_order
    )
].copy()


df["timepoint"] = pd.Categorical(
    df[timepoint_col],
    categories=timepoint_order,
    ordered=True
)


df["group"] = pd.Categorical(
    df["group"],
    categories=group_order,
    ordered=True
)


print("\nSamples:")
print(
    df.shape[0]
)

print("\nPatients:")
print(
    df[patient_col].nunique()
)


print("\nPatients by group and timepoint:")

print(
    df.groupby(
        [
            "group",
            "timepoint"
        ],
        observed=True
    )[patient_col]
    .nunique()
)


# ============================================================
# 11. AGGREGATE DUPLICATE SAMPLES
# ============================================================

df_patient = (
    df
    .groupby(
        [
            patient_col,
            "group",
            "timepoint"
        ],
        observed=True
    )[signature_cols]
    .mean()
    .reset_index()
)


print(
    "\nPatient-level table:"
)

print(
    df_patient.shape
)


# ============================================================
# 12. MIXED-EFFECTS MODEL
# ============================================================

print("\n==========================================")
print("MIXED-EFFECTS MODEL")
print("==========================================\n")


model_results = []


for signature in signature_cols:

    dat = df_patient[
        [
            patient_col,
            "timepoint",
            "group",
            signature
        ]
    ].rename(
        columns={
            signature: "score"
        }
    ).dropna()


    n_patients = (
        dat[patient_col]
        .nunique()
    )


    n_samples = len(dat)


    if n_patients < 6:

        print(
            f"WARNING: {signature} "
            f"only {n_patients} patients"
        )


    # --------------------------------------------------------
    # Mixed model
    #
    # score ~ timepoint * group
    #
    # The interaction tests whether the trajectory over time
    # differs between the two groups.
    # --------------------------------------------------------

    formula = (
        "score ~ "
        "C(timepoint) * C(group)"
    )


    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            fit = smf.mixedlm(
                formula=formula,
                data=dat,
                groups=dat[patient_col],
                re_formula="1"
            ).fit(
                reml=False,
                method="lbfgs",
                maxiter=1000,
                disp=False
            )


    except Exception:

        try:

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                fit = smf.mixedlm(
                    formula=formula,
                    data=dat,
                    groups=dat[patient_col],
                    re_formula="1"
                ).fit(
                    reml=False,
                    method="powell",
                    maxiter=1000,
                    disp=False
                )

        except Exception as e:

            print(
                f"FAILED: {signature}"
            )

            continue


    # --------------------------------------------------------
    # Save all model terms
    # --------------------------------------------------------

    for term in fit.params.index:

        model_results.append(
            {
                "signature": signature,
                "term": term,
                "coef": fit.params[term],
                "se": fit.bse[term],
                "p_value": fit.pvalues[term],
                "n_patients": n_patients,
                "n_samples": n_samples
            }
        )


model_results = pd.DataFrame(
    model_results
)


# ============================================================
# 13. FDR FOR ALL MODEL TERMS
# ============================================================

model_results["p_adj"] = np.nan


for term in model_results["term"].unique():

    idx = (
        model_results["term"] == term
    )

    pvalues = (
        model_results.loc[
            idx,
            "p_value"
        ]
    )

    model_results.loc[
        idx,
        "p_adj"
    ] = multipletests(
        pvalues,
        method="fdr_bh"
    )[1]


model_results.to_csv(
    os.path.join(
        args.output_dir,
        "mixed_model_all_terms.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 14. INTERACTION TERMS
# ============================================================

interaction = model_results[
    model_results["term"]
    .str.contains(
        ":",
        regex=False
    )
].copy()


# FDR specifically across interaction tests

interaction["p_adj_interaction"] = (
    multipletests(
        interaction["p_value"],
        method="fdr_bh"
    )[1]
)


interaction["significant_FDR_0.05"] = (
    interaction[
        "p_adj_interaction"
    ] < 0.05
)


interaction["significant_FDR_0.10"] = (
    interaction[
        "p_adj_interaction"
    ] < 0.10
)


interaction = interaction.sort_values(
    "p_adj_interaction"
)


interaction.to_csv(
    os.path.join(
        args.output_dir,
        "mixed_model_interactions.tsv"
    ),
    sep="\t",
    index=False
)


print(
    "\nSignificant interactions FDR < 0.05:"
)

print(
    interaction[
        interaction[
            "significant_FDR_0.05"
        ]
    ]
)


# ============================================================
# 15. FOREST PLOT
# ============================================================

def forest_plot_mixed_term(
    model_results,
    term,
    outdir,
    top_n=20,
    title=None
):

    plot_df = model_results[
        model_results["term"] == term
    ].copy()


    if plot_df.empty:

        print(
            f"No results for {term}"
        )

        return


    plot_df = (
        plot_df
        .sort_values(
            "p_adj_interaction"
        )
        .head(top_n)
        .copy()
    )


    plot_df["ci_low"] = (
        plot_df["coef"]
        -
        1.96 * plot_df["se"]
    )


    plot_df["ci_high"] = (
        plot_df["coef"]
        +
        1.96 * plot_df["se"]
    )


    plot_df["significance"] = "NS"


    plot_df.loc[
        plot_df[
            "p_adj_interaction"
        ] < 0.10,
        "significance"
    ] = "FDR < 0.10"


    plot_df.loc[
        plot_df[
            "p_adj_interaction"
        ] < 0.05,
        "significance"
    ] = "FDR < 0.05"


    colors = {
        "NS": "gray",
        "FDR < 0.10": "orange",
        "FDR < 0.05": "red"
    }


    fig, ax = plt.subplots(
        figsize=(
            8,
            max(
                5,
                len(plot_df) * 0.35
            )
        )
    )


    y = np.arange(
        len(plot_df)
    )


    for i, (_, row) in enumerate(
        plot_df.iterrows()
    ):

        ax.plot(
            [
                row["ci_low"],
                row["ci_high"]
            ],
            [
                i,
                i
            ],
            color="gray",
            linewidth=2
        )


        ax.scatter(
            row["coef"],
            i,
            s=80,
            color=colors[
                row["significance"]
            ],
            zorder=3
        )


    ax.axvline(
        0,
        linestyle="--",
        color="black"
    )


    ax.set_yticks(y)

    ax.set_yticklabels(
        plot_df["signature"]
    )


    ax.set_xlabel(
        "Interaction coefficient\n"
        f"({positive} vs {negative})"
    )


    if title is None:
        title = term


    ax.set_title(
        title
    )


    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=k,
            markerfacecolor=v,
            markersize=8
        )
        for k, v in colors.items()
    ]


    ax.legend(
        handles=handles,
        title="Significance",
        bbox_to_anchor=(
            1.02,
            1
        ),
        loc="upper left"
    )


    plt.tight_layout()


    filename = (
        term
        .replace("[", "")
        .replace("]", "")
        .replace(":", "_")
        .replace(" ", "_")
        +
        "_forest.png"
    )


    plt.savefig(
        os.path.join(
            outdir,
            filename
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ------------------------------------------------------------
# Determine exact interaction terms
# ------------------------------------------------------------

interaction_d7_term = (
    "C(timepoint)[T.d7 after]:"
    "C(group)[T."
    + positive
    + "]"
)


interaction_d28_term = (
    "C(timepoint)[T.d28 after]:"
    "C(group)[T."
    + positive
    + "]"
)


forest_plot_mixed_term(
    interaction,
    interaction_d7_term,
    args.output_dir,
    title=(
        f"Product → d7 trajectory\n"
        f"{positive} vs {negative}"
    )
)


forest_plot_mixed_term(
    interaction,
    interaction_d28_term,
    args.output_dir,
    title=(
        f"Product → d28 trajectory\n"
        f"{positive} vs {negative}"
    )
)


# ============================================================
# 16. DELTA ANALYSIS
# ============================================================

print("\n==========================================")
print("DELTA ANALYSIS")
print("==========================================\n")


delta_results = []


for signature in signature_cols:

    tmp = df_patient[
        [
            patient_col,
            "group",
            "timepoint",
            signature
        ]
    ].copy()


    tmp = tmp.pivot_table(
        index=[
            patient_col,
            "group"
        ],
        columns="timepoint",
        values=signature
    )


    if product not in tmp.columns:
        continue


    if d7 in tmp.columns:

        tmp[
            "delta_d7"
        ] = (
            tmp[d7]
            -
            tmp[product]
        )


    if d28 in tmp.columns:

        tmp[
            "delta_d28"
        ] = (
            tmp[d28]
            -
            tmp[product]
        )


    tmp = tmp.reset_index()


    # --------------------------------------------------------
    # Compare delta between groups
    # --------------------------------------------------------

    for delta_name in [
        "delta_d7",
        "delta_d28"
    ]:

        if delta_name not in tmp.columns:
            continue


        positive_values = tmp.loc[
            tmp["group"] == positive,
            delta_name
        ].dropna()


        negative_values = tmp.loc[
            tmp["group"] == negative,
            delta_name
        ].dropna()


        if (
            len(positive_values) < 3
            or
            len(negative_values) < 3
        ):
            continue


        stat, p = mannwhitneyu(
            positive_values,
            negative_values,
            alternative="two-sided"
        )


        delta_results.append(
            {
                "signature": signature,
                "delta": delta_name,
                "p_value": p,

                "positive_mean":
                    positive_values.mean(),

                "negative_mean":
                    negative_values.mean(),

                "positive_median":
                    positive_values.median(),

                "negative_median":
                    negative_values.median(),

                "effect_size":
                    (
                        positive_values.mean()
                        -
                        negative_values.mean()
                    ),

                "n_positive":
                    len(positive_values),

                "n_negative":
                    len(negative_values)
            }
        )


delta_results = pd.DataFrame(
    delta_results
)


if not delta_results.empty:

    delta_results["p_adj"] = (
        multipletests(
            delta_results[
                "p_value"
            ],
            method="fdr_bh"
        )[1]
    )


    delta_results = (
        delta_results
        .sort_values(
            "p_adj"
        )
    )


    delta_results.to_csv(
        os.path.join(
            args.output_dir,
            "delta_analysis.tsv"
        ),
        sep="\t",
        index=False
    )


else:

    print(
        "No delta results."
    )


# ============================================================
# 17. DELTA EFFECT HEATMAP
# ============================================================

def delta_effect_heatmap(
    delta_results,
    outdir
):

    if delta_results.empty:
        return


    mat = delta_results.pivot_table(
        index="signature",
        columns="delta",
        values="effect_size"
    )


    pmat = delta_results.pivot_table(
        index="signature",
        columns="delta",
        values="p_adj"
    )


    mat = mat.reindex(
        columns=[
            "delta_d7",
            "delta_d28"
        ]
    )


    pmat = pmat.reindex(
        columns=[
            "delta_d7",
            "delta_d28"
        ]
    )


    # --------------------------------------------------------
    # Panel information
    # --------------------------------------------------------

    panel_info = pd.DataFrame(
        index=mat.index
    )


    panel_info["panel"] = (
        panel_info.index
        .astype(str)
        .str.extract(
            r"^(P[1-4])_",
            expand=False
        )
    )


    panel_order = {
        "P1": 1,
        "P2": 2,
        "P3": 3,
        "P4": 4
    }


    panel_info["panel_order"] = (
        panel_info["panel"]
        .map(panel_order)
    )


    panel_info["max_abs_effect"] = (
        mat
        .abs()
        .max(
            axis=1,
            skipna=True
        )
    )


    panel_info = (
        panel_info
        .sort_values(
            [
                "panel_order",
                "max_abs_effect"
            ],
            ascending=[
                True,
                False
            ]
        )
    )


    ordered = panel_info.index


    mat = mat.loc[
        ordered
    ]

    pmat = pmat.reindex(
        ordered
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            7,
            max(
                8,
                len(mat) * 0.30
            )
        )
    )


    sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label":
            f"Effect size\n"
            f"{positive} - {negative}"
        },
        ax=ax
    )


    # Significance stars

    for i in range(
        pmat.shape[0]
    ):

        for j in range(
            pmat.shape[1]
        ):

            p = pmat.iloc[
                i,
                j
            ]


            if pd.isna(p):
                continue


            if p < 0.05:
                star = "**"

            elif p < 0.10:
                star = "*"

            else:
                continue


            ax.text(
                j + 0.5,
                i + 0.5,
                star,
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )


    # Panel boundaries

    panel_sequence = (
        panel_info["panel"]
        .values
    )


    boundaries = np.where(
        panel_sequence[1:]
        !=
        panel_sequence[:-1]
    )[0] + 1


    for boundary in boundaries:

        ax.axhline(
            boundary,
            color="black",
            linewidth=1.5
        )


    ax.set_title(
        f"Delta effect size\n"
        f"{positive} - {negative}"
    )


    ax.set_xlabel("")
    ax.set_ylabel("")


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            outdir,
            "Delta_effect_size_heatmap.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


delta_effect_heatmap(
    delta_results,
    args.output_dir
)


# ============================================================
# 18. TRAJECTORY PLOTS
# ============================================================

print("\n==========================================")
print("TRAJECTORY PLOTS")
print("==========================================\n")


palette = {
    positive: "#D62728",
    negative: "#1F77B4"
}


sns.set_theme(
    style="whitegrid",
    context="talk"
)


# ------------------------------------------------------------
# Signatures selected from either analysis
# ------------------------------------------------------------

interesting_signatures = set()


# Interaction

if not interaction.empty:

    interesting_signatures.update(
        interaction.loc[
            interaction[
                "p_adj_interaction"
            ] < 0.10,
            "signature"
        ]
    )


# Delta

if not delta_results.empty:

    interesting_signatures.update(
        delta_results.loc[
            delta_results[
                "p_adj"
            ] < 0.10,
            "signature"
        ]
    )


# If nothing significant — top 20 delta signatures

if len(
    interesting_signatures
) == 0:

    if not delta_results.empty:

        interesting_signatures = set(
            delta_results
            .sort_values(
                "p_adj"
            )
            .head(20)[
                "signature"
            ]
        )


print(
    f"Signatures for trajectory plots: "
    f"{len(interesting_signatures)}"
)


# ------------------------------------------------------------
# Plot trajectories
# ------------------------------------------------------------

for signature in (
    interesting_signatures
):

    plot_dat = df_patient[
        [
            patient_col,
            "group",
            "timepoint",
            signature
        ]
    ].dropna()


    if plot_dat.empty:
        continue


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    # Individual patient trajectories

    sns.lineplot(
        data=plot_dat,
        x="timepoint",
        y=signature,
        hue="group",
        units=patient_col,
        estimator=None,
        hue_order=group_order,
        palette=palette,
        alpha=0.25,
        linewidth=1.2,
        legend=False,
        ax=ax
    )


    # Group mean trajectory

    sns.pointplot(
        data=plot_dat,
        x="timepoint",
        y=signature,
        hue="group",
        order=timepoint_order,
        hue_order=group_order,
        palette=palette,
        errorbar="se",
        dodge=0.18,
        markers="o",
        linestyles="-",
        linewidth=2.2,
        ax=ax
    )


    ax.set_title(
        f"{signature}\n"
        f"{positive} vs {negative}"
    )


    ax.set_xlabel("")
    ax.set_ylabel(
        "ssGSEA score"
    )


    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(
            1.02,
            1
        ),
        title="Group"
    )


    sns.despine()


    plt.tight_layout()


    safe_signature = (
        str(signature)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


    plt.savefig(
        os.path.join(
            args.output_dir,
            f"trajectory_{safe_signature}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# 19. FINAL SUMMARY TABLE
# ============================================================

print("\n==========================================")
print("FINAL SUMMARY")
print("==========================================\n")


# ------------------------------------------------------------
# Interaction d7
# ------------------------------------------------------------

interaction_d7 = interaction[
    interaction["term"]
    == interaction_d7_term
].copy()


interaction_d7 = interaction_d7[
    [
        "signature",
        "coef",
        "p_adj_interaction"
    ]
].rename(
    columns={
        "coef":
            "interaction_d7",

        "p_adj_interaction":
            "FDR_interaction_d7"
    }
)


# ------------------------------------------------------------
# Interaction d28
# ------------------------------------------------------------

interaction_d28 = interaction[
    interaction["term"]
    == interaction_d28_term
].copy()


interaction_d28 = interaction_d28[
    [
        "signature",
        "coef",
        "p_adj_interaction"
    ]
].rename(
    columns={
        "coef":
            "interaction_d28",

        "p_adj_interaction":
            "FDR_interaction_d28"
    }
)


# ------------------------------------------------------------
# Delta d7
# ------------------------------------------------------------

if not delta_results.empty:

    delta_d7 = delta_results[
        delta_results[
            "delta"
        ] == "delta_d7"
    ].copy()


    delta_d7 = delta_d7[
        [
            "signature",
            "positive_mean",
            "negative_mean",
            "effect_size",
            "p_adj"
        ]
    ].rename(
        columns={
            "positive_mean":
                "delta_positive_d7",

            "negative_mean":
                "delta_negative_d7",

            "effect_size":
                "delta_effect_d7",

            "p_adj":
                "FDR_delta_d7"
        }
    )

else:

    delta_d7 = pd.DataFrame(
        columns=[
            "signature",
            "delta_positive_d7",
            "delta_negative_d7",
            "delta_effect_d7",
            "FDR_delta_d7"
        ]
    )


# ------------------------------------------------------------
# Delta d28
# ------------------------------------------------------------

if not delta_results.empty:

    delta_d28 = delta_results[
        delta_results[
            "delta"
        ] == "delta_d28"
    ].copy()


    delta_d28 = delta_d28[
        [
            "signature",
            "positive_mean",
            "negative_mean",
            "effect_size",
            "p_adj"
        ]
    ].rename(
        columns={
            "positive_mean":
                "delta_positive_d28",

            "negative_mean":
                "delta_negative_d28",

            "effect_size":
                "delta_effect_d28",

            "p_adj":
                "FDR_delta_d28"
        }
    )

else:

    delta_d28 = pd.DataFrame(
        columns=[
            "signature",
            "delta_positive_d28",
            "delta_negative_d28",
            "delta_effect_d28",
            "FDR_delta_d28"
        ]
    )


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

final_summary = (
    interaction_d7
    .merge(
        interaction_d28,
        on="signature",
        how="outer"
    )
    .merge(
        delta_d7,
        on="signature",
        how="outer"
    )
    .merge(
        delta_d28,
        on="signature",
        how="outer"
    )
)


# ------------------------------------------------------------
# Best FDR
# ------------------------------------------------------------

fdr_cols = [
    "FDR_interaction_d7",
    "FDR_interaction_d28",
    "FDR_delta_d7",
    "FDR_delta_d28"
]


final_summary["best_FDR"] = (
    final_summary[
        fdr_cols
    ].min(
        axis=1
    )
)


final_summary = (
    final_summary
    .sort_values(
        "best_FDR"
    )
)


# ------------------------------------------------------------
# Significance flags
# ------------------------------------------------------------

final_summary[
    "significant_d7"
] = (
    (
        final_summary[
            "FDR_interaction_d7"
        ] < 0.05
    )
    |
    (
        final_summary[
            "FDR_delta_d7"
        ] < 0.05
    )
)


final_summary[
    "significant_d28"
] = (
    (
        final_summary[
            "FDR_interaction_d28"
        ] < 0.05
    )
    |
    (
        final_summary[
            "FDR_delta_d28"
        ] < 0.05
    )
)


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

final_summary.to_csv(
    os.path.join(
        args.output_dir,
        "FINAL_signature_longitudinal_summary.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# 20. INTERPRETATION
# ============================================================

def classify_signal(row):

    # Strong interaction at d28

    if (
        pd.notna(
            row[
                "FDR_interaction_d28"
            ]
        )
        and
        row[
            "FDR_interaction_d28"
        ] < 0.05
    ):

        if (
            row[
                "interaction_d28"
            ] > 0
        ):

            return (
                f"Higher trajectory in "
                f"{positive}"
            )

        else:

            return (
                f"Higher trajectory in "
                f"{negative}"
            )


    # Strong delta at d28

    if (
        pd.notna(
            row[
                "FDR_delta_d28"
            ]
        )
        and
        row[
            "FDR_delta_d28"
        ] < 0.05
    ):

        if (
            row[
                "delta_effect_d28"
            ] > 0
        ):

            return (
                f"Greater increase in "
                f"{positive}"
            )

        else:

            return (
                f"Greater increase in "
                f"{negative}"
            )


    # Trend

    if (
        pd.notna(
            row[
                "FDR_interaction_d28"
            ]
        )
        and
        row[
            "FDR_interaction_d28"
        ] < 0.10
    ):

        if (
            row[
                "interaction_d28"
            ] > 0
        ):

            return (
                f"Trend: higher trajectory "
                f"in {positive}"
            )

        else:

            return (
                f"Trend: higher trajectory "
                f"in {negative}"
            )


    return (
        "No significant "
        "longitudinal difference"
    )


final_summary[
    "interpretation"
] = final_summary.apply(
    classify_signal,
    axis=1
)


# Save final version again

final_summary.to_csv(
    os.path.join(
        args.output_dir,
        "FINAL_signature_longitudinal_summary.tsv"
    ),
    sep="\t",
    index=False
)

