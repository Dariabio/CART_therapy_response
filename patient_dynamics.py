#%%
import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu

#%%
project_dir = "/Users/darakuzmina/Desktop/CAR_T"

results_dir = os.path.join(project_dir, "results")
figures_dir = os.path.join(
    project_dir,
    "figures",
    "figures_dynamics"
)
tables_dir = os.path.join(
    project_dir,
    "results",
    "tables_dynamics"
)

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
#%%

patient_col = "patient_id"
timepoint_col = "timepoint"
response_col = "group_analysis_1"
original_group_col = "group_analysis_2"

product = "product"
d7 = "d7 after"
d28 = "d28 after"

responder = "responder"
nonresponder = "nonresponder"

timepoint_order = [product, d7, d28]
model_groups = [nonresponder, responder]

plot_groups = [
    nonresponder,
    responder,
    "group_3",
    "group_4"
]
#%%

# 1. Load data


scores_all = pd.read_csv(
    os.path.join(
        results_dir,
        "results_ssgsea",
        "scores_all.csv"
    ),
    index_col=0
)

meta_all = pd.read_csv(
    os.path.join(
        project_dir,
        "data",
        "processed",
        "meta_all.csv"
    ),
    index_col=0
)

common_samples = (
    scores_all.index
    .intersection(meta_all.index)
)

scores = scores_all.loc[common_samples]
meta = meta_all.loc[common_samples]

signature_cols = list(scores.columns)


# Prepare table


df = meta[
    [
        patient_col,
        timepoint_col,
        response_col,
        original_group_col
    ]
].join(scores)

df[timepoint_col] = (
    df[timepoint_col]
    .astype(str)
    .str.strip()
)

df["plot_group"] = pd.NA

df.loc[
    df[response_col] == responder,
    "plot_group"
] = responder

df.loc[
    df[response_col] == nonresponder,
    "plot_group"
] = nonresponder

df.loc[
    df[original_group_col] == 3,
    "plot_group"
] = "group_3"

df.loc[
    df[original_group_col] == 4,
    "plot_group"
] = "group_4"

df = df[
    df[timepoint_col].isin(timepoint_order) &
    df["plot_group"].isin(plot_groups)
].copy()

df["timepoint"] = pd.Categorical(
    df[timepoint_col],
    categories=timepoint_order,
    ordered=True
)

df["plot_group"] = pd.Categorical(
    df["plot_group"],
    categories=plot_groups,
    ordered=True
)

print("Samples:", df.shape[0])
print("Patients:", df[patient_col].nunique())

print("\nSamples by group and timepoint:")
print(
    df.groupby(
        ["plot_group", "timepoint"],
        observed=True
    )[patient_col].nunique()
)


# Aggregate possible duplicate samples


df_patient = (
    df
    .groupby(
        [
            patient_col,
            "plot_group",
            "timepoint"
        ],
        observed=True
    )[signature_cols]
    .mean()
    .reset_index()
)
#%%
#3. Mixed-effects models


model_df = df_patient[
    df_patient["plot_group"].isin(model_groups)
].copy()

model_df["plot_group"] = pd.Categorical(
    model_df["plot_group"],
    categories=model_groups,
    ordered=True
)

model_results = []

for signature in signature_cols:

    dat = model_df[
        [
            patient_col,
            "timepoint",
            "plot_group",
            signature
        ]
    ].rename(
        columns={signature: "score"}
    ).dropna()

    if dat[patient_col].nunique() < 8:
        print(
        f"WARNING: {signature} "
        f"only {dat[patient_col].nunique()} patients"
    )

    # Does this signature change over time across all patients?
    # Is this signature higher/lower) in respondents than in non-respondents, on average across all three points?

    formula = (
        "score ~ C(timepoint) * C(plot_group)"
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

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

    for term in fit.params.index:

        model_results.append({
    "signature": signature,
    "term": term,
    "coef": fit.params[term],
    "se": fit.bse[term],
    "p_value": fit.pvalues[term],
    "n_patients": dat[patient_col].nunique(),
    "n_samples": len(dat)
})


model_results = pd.DataFrame(model_results)

# FDR correction for all terms separately

model_results["p_adj"] = np.nan

for term in model_results["term"].unique():

    idx = model_results["term"] == term

    model_results.loc[idx, "p_adj"] = multipletests(
        model_results.loc[idx, "p_value"],
        method="fdr_bh"
    )[1]

model_results.to_csv(
    os.path.join(
        tables_dir,
        "mixed_model_all_terms.tsv"
    ),
    sep="\t",
    index=False
)


# only interaction terms:
# timepoint × responder
# Does the signature change differently over time?
interaction = model_results[
    model_results["term"].str.contains(
        ":",
        regex=False
    )
].copy()

interaction["p_adj"] = multipletests(
    interaction["p_value"],
    method="fdr_bh"
)[1]

interaction["significant"] = (
    interaction["p_adj"] < 0.05
)

interaction = interaction.sort_values(
    "p_adj"
)

interaction.to_csv(
    os.path.join(
        tables_dir,
        "mixed_model_timepoint_group_interactions.tsv"
    ),
    sep="\t",
    index=False
)

print("\nDynamic interaction results:")
print(interaction)

print("\nFDR-significant interactions:")
print(
    interaction[
        interaction["significant"]
    ]
)
#%%
def forest_plot_mixed_term(
    model_results,
    term,
    outdir,
    top_n=20,
    title=None
):

    df = model_results[
        model_results["term"] == term
    ].copy()


    if df.empty:
        print("No results for", term)
        return


    df = (
        df
        .sort_values("p_adj")
        .head(top_n)
        .copy()
    )


    # 95% CI
    df["ci_low"] = df["coef"] - 1.96 * df["se"]
    df["ci_high"] = df["coef"] + 1.96 * df["se"]


    # категории значимости
    df["significance"] = "NS"

    df.loc[
        df["p_adj"] < 0.1,
        "significance"
    ] = "FDR < 0.1"

    df.loc[
        df["p_adj"] < 0.05,
        "significance"
    ] = "FDR < 0.05"


    colors = {
        "NS": "gray",
        "FDR < 0.1": "orange",
        "FDR < 0.05": "red"
    }


    fig, ax = plt.subplots(
        figsize=(8, max(5, len(df)*0.35))
    )


    y = np.arange(len(df))


    for i, (_, row) in enumerate(df.iterrows()):

        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [i, i],
            color="gray",
            linewidth=2
        )


        ax.scatter(
            row["coef"],
            i,
            s=80,
            color=colors[row["significance"]],
            zorder=3
        )


    ax.axvline(
        0,
        linestyle="--",
        color="black"
    )


    ax.set_yticks(y)

    ax.set_yticklabels(
        df["signature"]
    )


    ax.set_xlabel(
        "Interaction coefficient\n"
        "(difference in trajectory between responder and nonresponder)"
    )


    if title is None:
        title = term

    ax.set_title(title)


    # легенда
    handles = [
        plt.Line2D(
            [0],[0],
            marker="o",
            color="w",
            label=k,
            markerfacecolor=v,
            markersize=8
        )
        for k,v in colors.items()
    ]

    ax.legend(
        handles=handles,
        title="Significance",
        bbox_to_anchor=(1.02,1),
        loc="upper left"
    )

    plt.yticks(fontsize=10)
    plt.tight_layout()


    filename = (
        term
        .replace("[","")
        .replace("]","")
        .replace(":","_")
        .replace(" ","_")
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

forest_plot_mixed_term(
    interaction,
    "C(timepoint)[T.d7 after]:C(plot_group)[T.responder]",
    figures_dir,
    top_n=20,
    title="Product → d7 responder trajectory"
)

forest_plot_mixed_term(
    interaction,
    "C(timepoint)[T.d28 after]:C(plot_group)[T.responder]",
    figures_dir,
    top_n=20,
    title="Product → d28 responder trajectory"
)
#%%
# 4. Delta analysis
# which signatures change the most?
delta_results = []

for signature in signature_cols:

    tmp = df_patient[
        [
            patient_col,
            "plot_group",
            "timepoint",
            signature
        ]
    ].copy()

    tmp = tmp.pivot_table(
        index=[patient_col, "plot_group"],
        columns="timepoint",
        values=signature
    )

    if product not in tmp.columns:
        continue

    if d7 in tmp.columns:
        tmp["delta_d7"] = tmp[d7] - tmp[product]

    if d28 in tmp.columns:
        tmp["delta_d28"] = tmp[d28] - tmp[product]

    tmp = tmp.reset_index()


    for delta_name in ["delta_d7", "delta_d28"]:
        if delta_name not in tmp.columns:
            continue

        resp = tmp.loc[
            tmp["plot_group"] == responder,
            delta_name
        ].dropna()

        nonresp = tmp.loc[
            tmp["plot_group"] == nonresponder,
            delta_name
        ].dropna()

        if len(resp) < 3 or len(nonresp) < 3:
            continue

        stat, p = mannwhitneyu(
            resp,
            nonresp,
            alternative="two-sided"
        )

        delta_results.append({
            "signature": signature,
            "delta": delta_name,
            "p_value": p,
            "resp_mean": resp.mean(),
            "nonresp_mean": nonresp.mean(),
            "effect_size":
                resp.mean() - nonresp.mean()
        })
delta_results = pd.DataFrame(delta_results)

delta_results["p_adj"] = multipletests(
    delta_results["p_value"],
    method="fdr_bh"
)[1]

delta_results = delta_results.sort_values(
    "p_adj"
)

delta_results.to_csv(
    os.path.join(
        tables_dir,
        "delta_analysis.tsv"
    ),
    sep="\t",
    index=False
)


delta_results.head()

#%%
def delta_effect_heatmap(
    delta_results,
    outdir
):

    df = delta_results.copy()


    mat = df.pivot_table(
        index="signature",
        columns="delta",
        values="effect_size"
    )


    pmat = df.pivot_table(
        index="signature",
        columns="delta",
        values="p_adj"
    )

    '''
    # сортировка по силе эффекта
    order = (
        mat.abs()
        .max(axis=1)
        .sort_values(ascending=False)
        .index
    )

    mat = mat.loc[order]
    pmat = pmat.loc[order]
    '''


    cols = ["delta_d7", "delta_d28"]

    mat = mat.reindex(columns=cols)
    pmat = pmat.reindex(columns=cols)

    # Determine panel for each signature
    panel_info = pd.DataFrame(index=mat.index)

    panel_info["panel"] = (
        panel_info.index
        .astype(str)
        .str.extract(r"^(P[1-4])_", expand=False)
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

    # sort signatures
    panel_info["max_abs_effect"] = (
        mat
        .abs()
        .max(axis=1, skipna=True)
    )

    panel_info = panel_info.sort_values(
        ["panel_order", "max_abs_effect"],
        ascending=[True, False]
    )

    ordered_signatures = panel_info.index

    mat = mat.loc[ordered_signatures]
    pmat = pmat.reindex(ordered_signatures)


    fig, ax = plt.subplots(
        figsize=(7, max(8, len(mat)*0.30))
    )


    sns.heatmap(
        mat,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "label": 
            "Effect size\nResponder - Nonresponder"
        },
        ax=ax
    )


    # добавляем звёздочки вручную

    for i in range(pmat.shape[0]):
        for j in range(pmat.shape[1]):

            p = pmat.iloc[i,j]

            if pd.isna(p):
                continue

            if p < 0.05:
                star = "**"

            elif p < 0.1:
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

    # Add horizontal lines between panels
    

    panel_sequence = panel_info["panel"].values

    boundaries = np.where(
        panel_sequence[1:] != panel_sequence[:-1]
    )[0] + 1

    for boundary in boundaries:
        ax.axhline(
            boundary,
            color="black",
            linewidth=1.5
        )
    
    ax.set_title(
        "Delta effect size\nResponder - Nonresponder"
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
    figures_dir
)
#%%


sigs = delta_results.query("p_adj < 0.1")["signature"].unique()

if len(sigs) == 0:
    sigs = (
        delta_results
        .sort_values("p_adj")
        .head(10)["signature"]
    )

palette = {
    responder: "#D62728",
    nonresponder: "#1F77B4",
    "group_3": "#2CA02C",
    "group_4": "#9467BD"
}

for signature in sigs:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11,4),
        gridspec_kw={"width_ratios":[1.3,1]}
    )


    sns.pointplot(
        data=df_patient,
        x="timepoint",
        y=signature,
        hue="plot_group",
        order=timepoint_order,
        hue_order=[nonresponder,responder],
        palette=palette,
        errorbar="se",
        dodge=0.15,
        markers="o",
        linewidth=2,
        ax=axes[0]
    )

    axes[0].set_title("Trajectory")

    axes[0].set_ylabel("ssGSEA score")

    axes[0].set_xlabel("")


    tmp = df_patient[
        [
            patient_col,
            "plot_group",
            "timepoint",
            signature
        ]
    ]

    tmp = tmp.pivot_table(
        index=[patient_col,"plot_group"],
        columns="timepoint",
        values=signature
    )

    if (
        product not in tmp.columns or
        d7 not in tmp.columns
    ):
        plt.close()
        continue

    tmp["Delta"] = tmp[d28]-tmp[product]

    tmp = tmp.reset_index()

    sns.boxplot(
        data=tmp,
        x="plot_group",
        y="Delta",
        order=[nonresponder,responder],
        palette=[
            palette[nonresponder],
            palette[responder]
        ],
        ax=axes[1]
    )

    sns.stripplot(
        data=tmp,
        x="plot_group",
        y="Delta",
        order=[nonresponder,responder],
        color="black",
        size=5,
        ax=axes[1]
    )

    resp = tmp.loc[
        tmp["plot_group"]==responder,
        "Delta"
    ].dropna()

    non = tmp.loc[
        tmp["plot_group"]==nonresponder,
        "Delta"
    ].dropna()

    if len(resp)>2 and len(non)>2:

        _,p = mannwhitneyu(
            resp,
            non
        )

        axes[1].set_title(
            f"Δ product→d28\np={p:.3f}"
        )

    axes[1].set_ylabel("Δ ssGSEA")

    axes[1].set_xlabel("")

    sns.despine()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            f"{signature}_trajectory_delta.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()





#%%
# ============================================================
# Trajectory plots
# ============================================================

palette = {
    responder: "#D62728",
    nonresponder: "#1F77B4",
    "group_3": "#2CA02C",
    "group_4": "#9467BD"
}

sns.set_theme(
    style="whitegrid",
    context="talk"
)

interesting_signatures = set(
    interaction
    .query("p_adj < 0.1")
    ["signature"]
)

interesting_signatures.update(
    delta_results
    .query("p_adj < 0.1")
    ["signature"]
)

if len(interesting_signatures) == 0:

    interesting_signatures = set(
        delta_results
        .sort_values("p_adj")
        .head(20)
        ["signature"]
    )

for signature in interesting_signatures:

    plot_dat = df_patient[
        [
            patient_col,
            "plot_group",
            "timepoint",
            signature
        ]
    ].dropna()

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    # Индивидуальные линии пациентов
    sns.lineplot(
        data=plot_dat,
        x="timepoint",
        y=signature,
        hue="plot_group",
        units=patient_col,
        estimator=None,
        palette=palette,
        alpha=0.25,
        linewidth=1.2,
        legend=False,
        ax=ax
    )

    # Средняя динамика с 95% CI
    sns.pointplot(
        data=plot_dat,
        x="timepoint",
        y=signature,
        hue="plot_group",
        order=timepoint_order,
        hue_order=plot_groups,
        palette=palette,
        errorbar="se",
        dodge=0.18,
        markers="o",
        linestyles="-",
        linewidth=2.2,
        ax=ax
    )

    ax.set_title(signature)
    ax.set_xlabel("")
    ax.set_ylabel("ssGSEA score")

    sns.move_legend(
        ax,
        "upper left",
        bbox_to_anchor=(1.02, 1),
        title="Group"
    )

    sns.despine()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            figures_dir,
            f"trajectory_{signature}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()
# %%
# ============================================================
# FINAL SUMMARY TABLE
# ============================================================

# ------------------------------------------------------------
# 1. Interaction results
# ------------------------------------------------------------

interaction_summary = interaction.copy()

# Берем только два нужных interaction:
# product -> d7
# product -> d28

interaction_d7 = interaction_summary[
    interaction_summary["term"] ==
    "C(timepoint)[T.d7 after]:C(plot_group)[T.responder]"
].copy()

interaction_d28 = interaction_summary[
    interaction_summary["term"] ==
    "C(timepoint)[T.d28 after]:C(plot_group)[T.responder]"
].copy()


interaction_d7 = interaction_d7[
    ["signature", "coef", "p_adj"]
].rename(
    columns={
        "coef": "interaction_d7",
        "p_adj": "FDR_interaction_d7"
    }
)

interaction_d28 = interaction_d28[
    ["signature", "coef", "p_adj"]
].rename(
    columns={
        "coef": "interaction_d28",
        "p_adj": "FDR_interaction_d28"
    }
)


# ------------------------------------------------------------
# 2. Delta results
# ------------------------------------------------------------

delta_summary = delta_results.copy()

delta_d7 = delta_summary[
    delta_summary["delta"] == "delta_d7"
].copy()

delta_d28 = delta_summary[
    delta_summary["delta"] == "delta_d28"
].copy()


# delta_results уже содержит:
# resp_mean
# nonresp_mean
# effect_size
# p_adj

delta_d7 = delta_d7[
    [
        "signature",
        "resp_mean",
        "nonresp_mean",
        "effect_size",
        "p_adj"
    ]
].rename(
    columns={
        "resp_mean": "delta_resp_d7",
        "nonresp_mean": "delta_nonresp_d7",
        "effect_size": "delta_effect_d7",
        "p_adj": "FDR_delta_d7"
    }
)

delta_d28 = delta_d28[
    [
        "signature",
        "resp_mean",
        "nonresp_mean",
        "effect_size",
        "p_adj"
    ]
].rename(
    columns={
        "resp_mean": "delta_resp_d28",
        "nonresp_mean": "delta_nonresp_d28",
        "effect_size": "delta_effect_d28",
        "p_adj": "FDR_delta_d28"
    }
)


# ------------------------------------------------------------
# 3. Merge everything
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
# 4. Sort by most interesting signals
# ------------------------------------------------------------

# Минимальный FDR из всех longitudinal tests

fdr_cols = [
    "FDR_interaction_d7",
    "FDR_interaction_d28",
    "FDR_delta_d7",
    "FDR_delta_d28"
]

final_summary["best_FDR"] = (
    final_summary[fdr_cols]
    .min(axis=1)
)

final_summary = final_summary.sort_values(
    "best_FDR"
)


# ------------------------------------------------------------
# 5. Add significance flags
# ------------------------------------------------------------

final_summary["significant_d7"] = (
    (final_summary["FDR_interaction_d7"] < 0.05) |
    (final_summary["FDR_delta_d7"] < 0.05)
)

final_summary["significant_d28"] = (
    (final_summary["FDR_interaction_d28"] < 0.05) |
    (final_summary["FDR_delta_d28"] < 0.05)
)


# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------

final_summary.to_csv(
    os.path.join(
        tables_dir,
        "FINAL_signature_longitudinal_summary.tsv"
    ),
    sep="\t",
    index=False
)


print("\n==========================================")
print("FINAL LONGITUDINAL SUMMARY")
print("==========================================")

print(
    final_summary[
        [
            "signature",
            "interaction_d7",
            "FDR_interaction_d7",
            "delta_resp_d7",
            "delta_nonresp_d7",
            "delta_effect_d7",
            "FDR_delta_d7",
            "interaction_d28",
            "FDR_interaction_d28",
            "delta_resp_d28",
            "delta_nonresp_d28",
            "delta_effect_d28",
            "FDR_delta_d28"
        ]
    ].to_string(index=False)
)

def classify_signal(row):

    # Сильный longitudinal signal
    if (
        pd.notna(row["FDR_interaction_d28"])
        and row["FDR_interaction_d28"] < 0.05
    ):
        if row["interaction_d28"] > 0:
            return "Higher trajectory in responders"
        else:
            return "Higher trajectory in nonresponders"

    # Если interaction не значим, смотрим delta
    if (
        pd.notna(row["FDR_delta_d28"])
        and row["FDR_delta_d28"] < 0.05
    ):
        if row["delta_effect_d28"] > 0:
            return "Greater increase in responders"
        else:
            return "Greater increase in nonresponders"

    # Тренд
    if (
        pd.notna(row["FDR_interaction_d28"])
        and row["FDR_interaction_d28"] < 0.10
    ):
        if row["interaction_d28"] > 0:
            return "Trend: higher trajectory in responders"
        else:
            return "Trend: higher trajectory in nonresponders"

    return "No significant longitudinal difference"


final_summary["interpretation"] = final_summary.apply(
    classify_signal,
    axis=1
)