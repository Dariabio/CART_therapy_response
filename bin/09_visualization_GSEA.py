import os
import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from scipy.stats import zscore
from matplotlib.colors import TwoSlopeNorm, Normalize

#  Arguments

parser = argparse.ArgumentParser(
    description="ssGSEA and ranked GSEA visualization"
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
    "--stat_ssgsea_dir",
    required=True
)

parser.add_argument(
    "--ranked_gsea_dir",
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


# LOAD METADATA


meta = pd.read_csv(
    args.metadata,
    sep="\t",
    index_col=0
)

scores = pd.read_csv(
    args.scores_all,
    sep=",",
    index_col=0
)

#  1. MATCH SAMPLES

common_samples = (
    meta.index
    .intersection(scores.index)
)

meta = meta.loc[
    common_samples
].copy()

scores = scores.loc[
    common_samples
].copy()

# KEEP ONLY SAMPLES FOR CURRENT COMPARISON


comparison_mask = meta[
    comparison_column
].isin([
    positive,
    negative
])

meta = meta.loc[
    comparison_mask
].copy()

scores = scores.loc[
    meta.index
].copy()


# TIMEPOINTS


TIMEPOINTS = {
    "product": "product",
    "d7": "d7 after",
    "d28": "d28 after"
}


# PREPARE DATA BY TIMEPOINT

scores_by_tp = {}
meta_by_tp = {}


for short_name, tp in TIMEPOINTS.items():

    meta_tp = meta[
        meta["timepoint"] == tp
    ].copy()

    common_tp = (
        meta_tp.index
        .intersection(
            scores.index
        )
    )

    meta_tp = meta_tp.loc[
        common_tp
    ]

    scores_tp = scores.loc[
        common_tp
    ].copy()

    scores_by_tp[short_name] = scores_tp
    meta_by_tp[short_name] = meta_tp

    print(
        f"{short_name}: "
        f"{len(meta_tp)} samples"
    )

    print(
        meta_tp[
            comparison_column
        ].value_counts()
    )

    print()


# 4. LOAD ssGSEA STATISTICAL RESULTS

def load_ssgsea_results(
    comparison,
    timepoint
):
    
    path = os.path.join(
        args.stat_ssgsea_dir,
        f"ssGSEA_{timepoint}.tsv"
    )

    if not os.path.exists(path):

        print(
            f"Missing ssGSEA results: "
            f"{path}"
        )

        return None

    return pd.read_csv(
        path,
        sep="\t"
    )


# ============================================================
# 5. ssGSEA BOXPLOTS
# ============================================================

def make_ssgsea_boxplots(
    p_cutoff=0.05
):

    boxplot_dir = os.path.join(
        args.output_dir,
        "ssGSEA_boxplots"
    )

    os.makedirs(
        boxplot_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load ssGSEA statistics for all three timepoints
    # --------------------------------------------------------

    gsea_results = {}

    for short_name in TIMEPOINTS:

        res = load_ssgsea_results(
            comparison,
            short_name
        )

        if res is not None and not res.empty:

            gsea_results[
                short_name
            ] = res.copy()

    if not gsea_results:

        print(
            "No ssGSEA result files found."
        )

        return

    # --------------------------------------------------------
    # Process EACH timepoint separately
    # --------------------------------------------------------

    plots_log = []

    for short_name in TIMEPOINTS:

        current_res = gsea_results.get(
            short_name
        )

        if (
            current_res is None
            or current_res.empty
        ):
            print(
                f"No ssGSEA results for "
                f"{short_name}"
            )
            continue

        # ----------------------------------------------------
        # Significant signatures at this timepoint
        # ----------------------------------------------------

        current_res = current_res.copy()

        current_res["p_value"] = pd.to_numeric(
            current_res["p_value"],
            errors="coerce"
        )

        current_res["FDR"] = pd.to_numeric(
            current_res["FDR"],
            errors="coerce"
        )

        significant = current_res[
            current_res["p_value"] < p_cutoff
        ].copy()

        # Remove duplicated signatures if present
        significant = significant.drop_duplicates(
            subset="subpopulation"
        )

        # Sort by p-value
        significant = significant.sort_values(
            "p_value"
        )

        print(
            f"\n{short_name}: "
            f"{len(significant)} significant "
            f"ssGSEA signatures "
            f"(p < {p_cutoff})"
        )

        if significant.empty:

            print(
                f"No significant signatures "
                f"for {short_name}"
            )

            continue

        # ----------------------------------------------------
        # Metadata and scores for this timepoint
        # ----------------------------------------------------

        meta_tp = meta_by_tp.get(
            short_name
        )

        scores_tp = scores_by_tp.get(
            short_name
        )

        if (
            meta_tp is None
            or scores_tp is None
        ):
            print(
                f"Missing metadata or scores "
                f"for {short_name}"
            )
            continue

        # ----------------------------------------------------
        # Plot EVERY significant signature
        # ----------------------------------------------------

        for _, sig_stats in significant.iterrows():

            signature = sig_stats[
                "subpopulation"
            ]

            p_value = float(
                sig_stats["p_value"]
            )

            fdr = float(
                sig_stats["FDR"]
            )

            # ------------------------------------------------
            # Check signature exists in score matrix
            # ------------------------------------------------

            if signature not in scores_tp.columns:

                print(
                    f"Signature not found in "
                    f"scores: {signature}"
                )

                continue

            # ------------------------------------------------
            # Prepare plotting dataframe
            # ------------------------------------------------

            plot_df = pd.DataFrame(
                {
                    "score":
                    scores_tp[
                        signature
                    ],

                    "group":
                    meta_tp[
                        comparison_column
                    ]
                }
            )

            # Remove missing values
            plot_df = plot_df.dropna(
                subset=[
                    "score",
                    "group"
                ]
            )

            # Keep only comparison groups
            plot_df = plot_df[
                plot_df["group"].isin(
                    [
                        positive,
                        negative
                    ]
                )
            ]

            if plot_df.empty:

                print(
                    f"No data for "
                    f"{signature} — "
                    f"{short_name}"
                )

                continue

            # ------------------------------------------------
            # Boxplot
            # ------------------------------------------------

            plt.figure(
                figsize=(5, 4)
            )

            order = [
                positive,
                negative
            ]

            sns.boxplot(
                data=plot_df,
                x="group",
                y="score",
                order=order
            )

            sns.stripplot(
                data=plot_df,
                x="group",
                y="score",
                order=order,
                color="black",
                size=6,
                jitter=0.2
            )

            # ------------------------------------------------
            # Title
            # ------------------------------------------------

            plt.title(
                f"{signature}\n"
                f"{short_name} "
                f"(p = {p_value:.4g}, "
                f"FDR = {fdr:.4g})"
            )

            plt.xlabel(
                comparison
            )

            plt.ylabel(
                "ssGSEA score"
            )

            plt.tight_layout()

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            filename = (
                f"Boxplot_"
                f"{signature}_"
                f"{short_name}.png"
            )

            output_file = os.path.join(
                boxplot_dir,
                filename
            )

            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            plots_log.append(
                {
                    "Signature":
                    signature,

                    "Timepoint":
                    short_name,

                    "p_value":
                    p_value,

                    "FDR":
                    fdr,

                    "File_Name":
                    filename
                }
            )

            print(
                f"Saved: {output_file}"
            )

    # --------------------------------------------------------
    # Save log of all generated plots
    # --------------------------------------------------------

    if plots_log:

        plots_log_df = pd.DataFrame(
            plots_log
        )

        log_file = os.path.join(
            boxplot_dir,
            "ssGSEA_boxplots_summary.tsv"
        )

        plots_log_df.to_csv(
            log_file,
            sep="\t",
            index=False
        )

        print(
            f"\nBoxplot summary saved to: "
            f"{log_file}"
        )

    else:

        print(
            "\nNo significant ssGSEA "
            "boxplots were generated."
        )


# ============================================================
# RUN ssGSEA BOXPLOTS
# ============================================================

make_ssgsea_boxplots(
    p_cutoff=0.05
)

# ============================================================
# 6. LOAD RANKED GSEA RESULTS

def load_ranked_gsea_results(comparison):

    result_files = {
        "product": os.path.join(
            args.ranked_gsea_dir,
            "product",
            "gseapy.gene_set.prerank.report.csv"
        ),

        "d7": os.path.join(
            args.ranked_gsea_dir,
            "d7",
            "gseapy.gene_set.prerank.report.csv"
        ),

        "d28": os.path.join(
            args.ranked_gsea_dir,
            "d28",
            "gseapy.gene_set.prerank.report.csv"
        )
    }

    results = {}

    for timepoint, path in result_files.items():

        print(f"\nChecking ranked GSEA:")
        print(path)

        if not os.path.exists(path):
            print("  -> NOT FOUND")
            continue

        print("  -> FOUND")

        try:

            df = pd.read_csv(
                path,
                sep=",",
                engine="python",
                on_bad_lines="warn"
            )

        except Exception as e:

            print(
                f"  -> ERROR reading file: {e}"
            )

            continue

        # Проверяем необходимые колонки

        required_columns = [
            "Term",
            "NES",
            "FDR q-val"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:

            print(
                f"  -> Missing columns: "
                f"{missing_columns}"
            )

            print(
                "  -> Available columns:"
            )

            print(
                df.columns.tolist()
            )

            continue

        # Оставляем только нужные колонки

        df = df[
            [
                "Term",
                "NES",
                "FDR q-val"
            ]
        ].copy()

        # Приводим числовые колонки к numeric

        df["NES"] = pd.to_numeric(
            df["NES"],
            errors="coerce"
        )

        df["FDR q-val"] = pd.to_numeric(
            df["FDR q-val"],
            errors="coerce"
        )

        # Убираем строки без NES

        df = df.dropna(
            subset=["NES"]
        )

        results[timepoint] = df

        print(
            f"  -> Loaded {len(df)} pathways"
        )

    return results



ranked_gsea_results = load_ranked_gsea_results(comparison)


# ============================================================
# 7. COMMON RANKED GSEA NES HEATMAP
# ============================================================

def ranked_gsea_heatmap(
    gsea_results
):

    if not gsea_results:

        print(
            "No ranked GSEA results "
            "for heatmap."
        )

        return

    # --------------------------------------------------------
    # Extract NES and FDR
    # --------------------------------------------------------

    nes_list = []
    fdr_list = []

    '''
    for timepoint in [
        "product",
        "d7",
        "d28"
    ]:
    '''

    for timepoint in [
            "product"
        ]:

        if timepoint not in gsea_results:
            continue

        res = gsea_results[
            timepoint
        ].copy()

        if "Term" not in res.columns:
            print(
                f"No 'Term' column in "
                f"{timepoint} GSEA result."
            )
            continue

        res = res.set_index(
            "Term"
        )

        nes = pd.to_numeric(
            res["NES"],
            errors="coerce"
        )

        fdr = pd.to_numeric(
            res["FDR q-val"],
            errors="coerce"
        )

        nes_list.append(
            nes.rename(
                timepoint
            )
        )

        fdr_list.append(
            fdr.rename(
                timepoint
            )
        )

    if not nes_list:

        print(
            "No NES values available."
        )

        return

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    nes_mat = pd.concat(
        nes_list,
        axis=1
    )

    fdr_mat = pd.concat(
        fdr_list,
        axis=1
    )


    # --------------------------------------------------------
# Signature order for heatmap
# --------------------------------------------------------

    signature_order = [

    # Naive / memory
    "Naive_Like",
    "T_EarlyMem",
    "Central_Memory",
    "T_NonExhausted",

    # Effector-memory
    "Effector_Memory",
    "CD8_TEM_circulating",
    "Tissue_resident_memory",

    # Effector
    "Effector",
    "Cytotoxic",

    # Activation / functional states
    "BTM_T cell_activation",
    "Th1",
    "TfH",
    "Treg",

    # Exhaustion trajectory
    "Pre_Exhausted",
    "TPEX_up",
    "TEX_int_up",
    "TEX_term_up",
    "Exhaustion_Wherry",
    "Exhaustion_Long",
    "Exhausted_T",
]




    # Remove P1-P4 prefix from signature names

    def clean_signature_names(index):
        return (
        index
        .astype(str)
        .str.strip()
        .str.replace(
            r"^P[1-4][-_: ]",
            "",
            regex=True
        )
        .str.strip()
    )


    nes_mat = nes_mat.copy()
    fdr_mat = fdr_mat.copy()

    nes_mat.index = clean_signature_names(
    nes_mat.index
)

    fdr_mat.index = clean_signature_names(
    fdr_mat.index
)


# --------------------------------------------------------
# Keep only signatures that exist in GSEA results
# and follow custom biological order
# --------------------------------------------------------

    available_signatures = [
    term
    for term in signature_order
    if term in nes_mat.index
]



# --------------------------------------------------------
# Reorder NES and FDR matrices
# --------------------------------------------------------

    nes_mat = nes_mat.loc[
    available_signatures
]

    fdr_mat = fdr_mat.loc[
    available_signatures
]
    # --------------------------------------------------------
    # Significance markers
    # --------------------------------------------------------

    significance = fdr_mat.apply(
        lambda col: col.map(
            lambda x:
            "*" if (
                pd.notna(x)
                and x < 0.05
            )
            else ""
        )
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    g = sns.clustermap(
        nes_mat,
        cmap="RdBu_r",
        center=0,

        figsize=(
            10,
            max(
                8,
                len(nes_mat) * 0.30
            )
        ),

        col_cluster=False,
        row_cluster=False,

        linewidths=0.5,
        linecolor="white",

        annot=significance,
        fmt="",

        annot_kws={
            "fontsize": 9,
            "fontweight": "bold"
        }
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    g.ax_heatmap.set_xlabel(
        "Timepoint"
    )

    g.ax_heatmap.set_ylabel(
        "Gene signature"
    )


    label_map = {
            "product": "Product"
        }

    existing_columns = (
        nes_mat.columns
    )

    g.ax_heatmap.set_xticklabels(
        [
            label_map[x]
            for x in existing_columns
        ],
        rotation=0
    )

    g.ax_heatmap.set_title(
        f"Ranked GSEA — "
        f"{comparison} — "
        f"NES across timepoints",
        pad=20
    )

   

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        args.output_dir,
        "Ranked_GSEA_NES_heatmap.png"
    )

    g.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        g.fig
    )


ranked_gsea_heatmap(
    ranked_gsea_results
)

