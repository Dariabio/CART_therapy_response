import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu


# ============================================================
# PATHS
# ============================================================

H5AD = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "data/scRNAseq/CART_fulldataset_clustered.h5ad"
)

SIGNATURE_FILE = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/13_signature/"
    "CAR_T_response_signature.tsv"
)

OUTPUT_DIR = (
    "/Users/darakuzmina/Desktop/CAR_T/"
    "results/15_signature_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

MIN_CELLS_PER_PATIENT = 20

MIN_CELLS_THRESHOLDS = [
    20,
    50,
    100,
    200
]

N_PERMUTATIONS = 10000
N_BOOTSTRAPS = 5000

RANDOM_SEED = 42


# ============================================================
# LOAD SIGNATURE
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE,
    sep="\t"
)

print()
print("Signature columns:")
print(signature.columns.tolist())

available_genes = (
    signature.loc[
        signature["direction"] == "responder_UP",
        "gene"
    ]
    .astype(str)
    .str.strip()
    .tolist()
)

# remove duplicates while preserving order

available_genes = list(
    dict.fromkeys(
        available_genes
    )
)

print()
print(
    f"Genes in bulk signature: "
    f"{len(available_genes)}"
)

print(
    available_genes
)


# ============================================================
# LOAD H5AD
# ============================================================

print()
print("Loading h5ad in backed mode...")

adata = sc.read_h5ad(
    H5AD,
    backed="r"
)

print(
    "adata shape:",
    adata.shape
)

print(
    "adata.raw shape:",
    adata.raw.shape
)


# ============================================================
# CHECK REQUIRED OBS COLUMNS
# ============================================================

required_obs_columns = [
    "timepoint",
    "barcode",
    "cell_type",
    "response"
]

missing_columns = [
    col
    for col in required_obs_columns
    if col not in adata.obs.columns
]

if missing_columns:

    raise ValueError(
        "Missing required obs columns: "
        + str(missing_columns)
    )


# ============================================================
# CHECK SIGNATURE GENES IN RAW
# ============================================================

raw_genes = set(
    adata.raw.var_names
)

missing_genes = [
    gene
    for gene in available_genes
    if gene not in raw_genes
]

if missing_genes:

    print()
    print(
        "WARNING: signature genes "
        "missing from adata.raw:"
    )

    print(missing_genes)


available_genes = [
    gene
    for gene in available_genes
    if gene in raw_genes
]

print()
print(
    f"Signature genes available "
    f"in adata.raw: {len(available_genes)}"
)

print(
    available_genes
)

if len(available_genes) < 5:

    raise ValueError(
        "Too few signature genes available "
        "in adata.raw."
    )


# ============================================================
# SELECT CELLS
# ============================================================

print()
print("Selecting Infusion T cells...")


timepoint_mask = (
    adata.obs["timepoint"].astype(str)
    == "Infusion"
)

T_cell_types = [
    "Infusion T",
    "CD4 T",
    "CD8 T"
]

cell_type_mask = (
    adata.obs["cell_type"]
    .astype(str)
    .isin(T_cell_types)
)

mask = (
    timepoint_mask
    & cell_type_mask
)

obs = adata.obs.loc[
    mask
].copy()


print(
    f"Selected cells: {len(obs)}"
)

print()
print(
    "Cell types:"
)

print(
    obs["cell_type"]
    .value_counts()
)


# ============================================================
# GET RAW EXPRESSION FOR SIGNATURE GENES
# ============================================================

print()
print(
    "Extracting signature expression "
    "from adata.raw..."
)

gene_indices = [
    adata.raw.var_names.get_loc(
        gene
    )
    for gene in available_genes
]

selected_cell_indices = np.where(
    mask.values
)[0]

X = adata.raw.X[
    selected_cell_indices,
    :
][:, gene_indices]

print(
    "Expression matrix shape:",
    X.shape
)


# ============================================================
# PATIENT METADATA
# ============================================================

obs_patient = (
    obs[
        [
            "barcode",
            "response"
        ]
    ]
    .drop_duplicates(
        "barcode"
    )
    .set_index(
        "barcode"
    )
)

obs_patient["response"] = (
    obs_patient["response"]
    .astype(str)
    .str.strip()
)

print()
print(
    "Patient response counts:"
)

print(
    obs_patient["response"]
    .value_counts()
)


# ============================================================
# BUILD PATIENT-LEVEL EXPRESSION
# ============================================================

def build_patient_expression(
    X,
    obs,
    genes,
    cell_type=None
):

    # --------------------------------------------------------
    # Cell type filter
    # --------------------------------------------------------

    if cell_type is not None:

        mask = (
            obs["cell_type"].astype(str)
            == cell_type
        )

        obs_use = obs.loc[
            mask
        ].copy()

        X_use = X[mask.values]

    else:

        obs_use = obs.copy()

        X_use = X

    # --------------------------------------------------------
    # Patient IDs
    # --------------------------------------------------------

    patient_ids = (
        obs_use["barcode"]
        .astype(str)
        .values
    )

    # --------------------------------------------------------
    # Convert expression to DataFrame
    # --------------------------------------------------------

    if hasattr(
        X_use,
        "toarray"
    ):

        X_dense = X_use.toarray()

    else:

        X_dense = np.asarray(
            X_use
        )

    expression = pd.DataFrame(
        X_dense,
        columns=genes,
        index=patient_ids
    )

    # --------------------------------------------------------
    # Patient-level mean expression
    #
    # This is pseudobulk-like aggregation.
    # It is NOT count-sum pseudobulk.
    # --------------------------------------------------------

    patient_expr = (
        expression
        .groupby(level=0)
        .mean()
    )

    # --------------------------------------------------------
    # Number of cells per patient
    # --------------------------------------------------------

    cell_counts = (
        pd.Series(
            patient_ids
        )
        .value_counts()
        .sort_index()
    )

    return (
        patient_expr,
        cell_counts
    )


# ============================================================
# BUILD TWO ANALYSES
# ============================================================

print()
print(
    "Building All T patient profiles..."
)

patient_expr_all_T, cell_counts_all_T = (
    build_patient_expression(
        X,
        obs,
        available_genes,
        cell_type=None
    )
)


print()
print(
    "Building Infusion T patient profiles..."
)

patient_expr_infusion_T, cell_counts_infusion_T = (
    build_patient_expression(
        X,
        obs,
        available_genes,
        cell_type="Infusion T"
    )
)


# ============================================================
# PRINT PATIENT SUMMARY
# ============================================================

def print_patient_summary(
    patient_expr,
    cell_counts,
    label
):

    print()
    print(
        "=" * 60
    )

    print(
        label
    )

    print(
        "=" * 60
    )

    summary = pd.DataFrame({

        "n_cells": cell_counts,

        "response": (
            obs_patient
            .loc[
                cell_counts.index,
                "response"
            ]
        )
    })

    print(
        summary
        .sort_values(
            "n_cells"
        )
        .to_string()
    )

    print()

    print(
        "Number of patients:",
        len(summary)
    )

    print(
        "Responders:",
        (summary["response"] == "R").sum()
    )

    print(
        "Nonresponders:",
        (summary["response"] == "NR").sum()
    )

    print(
        "Patients >= "
        f"{MIN_CELLS_PER_PATIENT} cells:",
        (
            summary["n_cells"]
            >= MIN_CELLS_PER_PATIENT
        ).sum()
    )


print_patient_summary(
    patient_expr_all_T,
    cell_counts_all_T,
    "ALL T"
)

print_patient_summary(
    patient_expr_infusion_T,
    cell_counts_infusion_T,
    "INFUSION T"
)


# ============================================================
# FILTER PATIENTS
# ============================================================

def filter_patients_by_cell_count(
    patient_expr,
    cell_counts,
    min_cells
):

    keep = (
        cell_counts[
            cell_counts >= min_cells
        ]
        .index
    )

    filtered_expr = (
        patient_expr
        .loc[
            patient_expr.index.isin(
                keep
            )
        ]
        .copy()
    )

    return filtered_expr


patient_expr_all_T = (
    filter_patients_by_cell_count(
        patient_expr_all_T,
        cell_counts_all_T,
        MIN_CELLS_PER_PATIENT
    )
)

patient_expr_infusion_T = (
    filter_patients_by_cell_count(
        patient_expr_infusion_T,
        cell_counts_infusion_T,
        MIN_CELLS_PER_PATIENT
    )
)


# ============================================================
# HEATMAP
# ============================================================

def plot_signature_heatmap(
    patient_expr,
    label
):

    # --------------------------------------------------------
    # Genes x patients
    # --------------------------------------------------------

    heatmap_data = (
        patient_expr[
            available_genes
        ]
        .T
    )

    # --------------------------------------------------------
    # Gene-wise z-score
    # --------------------------------------------------------

    z_data = pd.DataFrame(
        index=heatmap_data.index,
        columns=heatmap_data.columns,
        dtype=float
    )

    for gene in heatmap_data.index:

        values = (
            heatmap_data.loc[
                gene
            ]
            .astype(float)
        )

        sd = values.std(
            ddof=1
        )

        if (
            pd.isna(sd)
            or sd == 0
        ):

            z_data.loc[
                gene
            ] = values

        else:

            z_data.loc[
                gene
            ] = (
                values - values.mean()
            ) / sd

    # --------------------------------------------------------
    # Patient order
    # --------------------------------------------------------

    response = (
        obs_patient
        .loc[
            z_data.columns,
            "response"
        ]
    )

    patient_order = (
        response
        .sort_values()
        .index
    )

    z_data = z_data[
        patient_order
    ]

    # --------------------------------------------------------
    # Verify genes
    # --------------------------------------------------------

    missing_heatmap_genes = [
        gene
        for gene in available_genes
        if gene not in z_data.index
    ]

    if missing_heatmap_genes:

        raise ValueError(
            "Missing genes in heatmap: "
            + str(
                missing_heatmap_genes
            )
        )

    # --------------------------------------------------------
    # Response annotation
    # --------------------------------------------------------

    response_colors = pd.DataFrame(
        {
            "Response": [
                "R"
                if response[p] == "R"
                else "NR"
                for p in patient_order
            ]
        },
        index=patient_order
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    g = sns.clustermap(
        z_data,
        row_cluster=False,
        col_cluster=False,
        cmap="vlag",
        center=0,
        figsize=(
            max(
                12,
                len(patient_order) * 0.45
            ),
            8
        ),
        xticklabels=True,
        yticklabels=True,
        cbar_kws={
            "label": "Gene-wise z-score"
        }
    )

    g.ax_heatmap.set_xlabel(
        "Patient"
    )

    g.ax_heatmap.set_ylabel(
        "Signature gene"
    )

    g.ax_heatmap.set_title(
        f"CAR-T response signature — {label}"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"signature_heatmap_{label.replace(' ', '_')}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved:",
        output_file
    )


plot_signature_heatmap(
    patient_expr_all_T,
    "All T"
)

plot_signature_heatmap(
    patient_expr_infusion_T,
    "Infusion T"
)


# ============================================================
# SIGNATURE SCORE
# ============================================================

def calculate_signature_score(
    patient_expr,
    genes
):

    expression = (
        patient_expr[
            genes
        ]
        .astype(float)
    )

    z_expression = pd.DataFrame(
        index=expression.index,
        columns=expression.columns,
        dtype=float
    )

    for gene in genes:

        values = (
            expression[gene]
        )

        sd = values.std(
            ddof=1
        )

        if (
            pd.isna(sd)
            or sd == 0
        ):

            z_expression[
                gene
            ] = 0.0

        else:

            z_expression[
                gene
            ] = (
                values - values.mean()
            ) / sd

    score = (
        z_expression
        .mean(axis=1)
    )

    return score


# ============================================================
# RANK-BISERIAL EFFECT
# ============================================================

def rank_biserial_effect(
    scores_R,
    scores_NR
):

    n_R = len(
        scores_R
    )

    n_NR = len(
        scores_NR
    )

    U, _ = mannwhitneyu(
        scores_R,
        scores_NR,
        alternative="two-sided"
    )

    effect = (
        2 * U
        / (
            n_R * n_NR
        )
        - 1
    )

    return effect


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(
    scores,
    labels,
    n_permutations=10000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    labels = np.asarray(
        labels
    )

    observed = (
        np.median(
            scores[
                labels == "R"
            ]
        )
        -
        np.median(
            scores[
                labels == "NR"
            ]
        )
    )

    null_distribution = np.zeros(
        n_permutations
    )

    for i in range(
        n_permutations
    ):

        shuffled_labels = (
            rng.permutation(
                labels
            )
        )

        null_distribution[i] = (
            np.median(
                scores[
                    shuffled_labels == "R"
                ]
            )
            -
            np.median(
                scores[
                    shuffled_labels == "NR"
                ]
            )
        )

    p_value = (
        (
            np.sum(
                np.abs(
                    null_distribution
                )
                >= abs(observed)
            )
            + 1
        )
        /
        (
            n_permutations
            + 1
        )
    )

    return (
        observed,
        p_value,
        null_distribution
    )


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_median_difference(
    scores_R,
    scores_NR,
    n_bootstraps=5000,
    seed=42
):

    rng = np.random.default_rng(
        seed
    )

    scores_R = np.asarray(
        scores_R,
        dtype=float
    )

    scores_NR = np.asarray(
        scores_NR,
        dtype=float
    )

    differences = np.zeros(
        n_bootstraps
    )

    for i in range(
        n_bootstraps
    ):

        boot_R = rng.choice(
            scores_R,
            size=len(scores_R),
            replace=True
        )

        boot_NR = rng.choice(
            scores_NR,
            size=len(scores_NR),
            replace=True
        )

        differences[i] = (
            np.median(
                boot_R
            )
            -
            np.median(
                boot_NR
            )
        )

    ci_low, ci_high = (
        np.percentile(
            differences,
            [2.5, 97.5]
        )
    )

    return (
        ci_low,
        ci_high
    )


# ============================================================
# MAIN STATISTICAL VALIDATION
# ============================================================

def validate_signature(
    patient_expr,
    label
):

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    scores = (
        calculate_signature_score(
            patient_expr,
            available_genes
        )
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = (
        obs_patient
        .loc[
            scores.index,
            "response"
        ]
        .astype(str)
    )

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    scores_R = scores[
        response == "R"
    ]

    scores_NR = scores[
        response == "NR"
    ]

    if (
        len(scores_R) == 0
        or
        len(scores_NR) == 0
    ):

        raise ValueError(
            f"{label}: "
            "one response group is empty."
        )

    # --------------------------------------------------------
    # Permutation test
    # --------------------------------------------------------

    (
        observed_difference,
        permutation_p,
        null_distribution
    ) = permutation_test(

        scores.values,

        response.values,

        n_permutations=(
            N_PERMUTATIONS
        ),

        seed=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Mann-Whitney
    # --------------------------------------------------------

    U, mw_p = mannwhitneyu(

        scores_R,

        scores_NR,

        alternative="two-sided"
    )

    # --------------------------------------------------------
    # Effect size
    # --------------------------------------------------------

    effect = (
        rank_biserial_effect(
            scores_R,
            scores_NR
        )
    )

    # --------------------------------------------------------
    # Bootstrap CI
    # --------------------------------------------------------

    (
        ci_low,
        ci_high
    ) = bootstrap_median_difference(

        scores_R,

        scores_NR,

        n_bootstraps=(
            N_BOOTSTRAPS
        ),

        seed=RANDOM_SEED
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    statistics = pd.DataFrame({

        "analysis": [
            label
        ],

        "n_R": [
            len(scores_R)
        ],

        "n_NR": [
            len(scores_NR)
        ],

        "median_R": [
            np.median(
                scores_R
            )
        ],

        "median_NR": [
            np.median(
                scores_NR
            )
        ],

        "median_difference_R_minus_NR": [
            observed_difference
        ],

        "permutation_p": [
            permutation_p
        ],

        "mannwhitney_U": [
            U
        ],

        "mannwhitney_p": [
            mw_p
        ],

        "rank_biserial_effect": [
            effect
        ],

        "bootstrap_CI_low": [
            ci_low
        ],

        "bootstrap_CI_high": [
            ci_high
        ]
    })

    # --------------------------------------------------------
    # Save patient scores
    # --------------------------------------------------------

    scores_output = pd.DataFrame({

        "patient_id": scores.index,

        "signature_score": scores.values,

        "response": response.values
    })

    scores_output.to_csv(

        os.path.join(
            OUTPUT_DIR,
            f"signature_scores_{label}.tsv"
        ),

        sep="\t",

        index=False
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics.to_csv(

        os.path.join(
            OUTPUT_DIR,
            f"signature_statistics_{label}.tsv"
        ),

        sep="\t",

        index=False
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        f"SIGNATURE VALIDATION: {label}"
    )

    print(
        "=" * 60
    )

    print(
        f"R:  n={len(scores_R)}, "
        f"median={np.median(scores_R):.4f}"
    )

    print(
        f"NR: n={len(scores_NR)}, "
        f"median={np.median(scores_NR):.4f}"
    )

    print(
        f"Median difference R-NR: "
        f"{observed_difference:.4f}"
    )

    print(
        f"Permutation p: "
        f"{permutation_p:.5f}"
    )

    print(
        f"Mann-Whitney p: "
        f"{mw_p:.5f}"
    )

    print(
        f"Rank-biserial effect: "
        f"{effect:.4f}"
    )

    print(
        "Bootstrap 95% CI: "
        f"[{ci_low:.4f}, "
        f"{ci_high:.4f}]"
    )

    return {

        "scores": scores,

        "response": response,

        "statistics": statistics,

        "null_distribution": (
            null_distribution
        )
    }


# ============================================================
# RUN MAIN VALIDATION
# ============================================================

validation_all_T = (
    validate_signature(
        patient_expr_all_T,
        "All_T"
    )
)

validation_infusion_T = (
    validate_signature(
        patient_expr_infusion_T,
        "Infusion_T"
    )
)


# ============================================================
# SIGNATURE SCORE PLOTS
# ============================================================

def plot_signature_score(
    validation_result,
    label
):

    scores = (
        validation_result[
            "scores"
        ]
    )

    response = (
        validation_result[
            "response"
        ]
    )

    plot_data = pd.DataFrame({

        "score": scores,

        "response": response
    })

    plt.figure(
        figsize=(6, 5)
    )

    sns.boxplot(
        data=plot_data,
        x="response",
        y="score"
    )

    sns.stripplot(
        data=plot_data,
        x="response",
        y="score",
        color="black",
        alpha=0.7,
        jitter=0.12
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Response"
    )

    plt.ylabel(
        "Signature score"
    )

    plt.title(
        f"CAR-T response signature — {label}"
    )

    plt.tight_layout()

    output_file = os.path.join(

        OUTPUT_DIR,

        f"signature_score_"
        f"{label.replace(' ', '_')}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print(
        "Saved:",
        output_file
    )


plot_signature_score(
    validation_all_T,
    "All T"
)

plot_signature_score(
    validation_infusion_T,
    "Infusion T"
)


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(
    patient_expr,
    cell_counts,
    label,
    thresholds
):

    results = []

    for threshold in thresholds:

        print()
        print(
            f"Running {label} "
            f"with >= {threshold} cells..."
        )

        # ----------------------------------------------------
        # Select patients
        # ----------------------------------------------------

        eligible_patients = (
            cell_counts[
                cell_counts >= threshold
            ]
            .index
        )

        expr = (
            patient_expr
            .loc[
                patient_expr.index.isin(
                    eligible_patients
                )
            ]
            .copy()
        )

        # ----------------------------------------------------
        # Skip if too few patients
        # ----------------------------------------------------

        if len(expr) < 4:

            print(
                "Too few patients. "
                "Skipping."
            )

            continue

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        scores = (
            calculate_signature_score(
                expr,
                available_genes
            )
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response = (
            obs_patient
            .loc[
                scores.index,
                "response"
            ]
            .astype(str)
        )

        scores_R = scores[
            response == "R"
        ]

        scores_NR = scores[
            response == "NR"
        ]

        if (
            len(scores_R) < 2
            or
            len(scores_NR) < 2
        ):

            print(
                "Too few patients in "
                "one response group. "
                "Skipping."
            )

            continue

        # ----------------------------------------------------
        # Permutation
        # ----------------------------------------------------

        (
            observed_difference,
            permutation_p,
            _
        ) = permutation_test(

            scores.values,

            response.values,

            n_permutations=(
                N_PERMUTATIONS
            ),

            seed=RANDOM_SEED
        )

        # ----------------------------------------------------
        # Mann-Whitney
        # ----------------------------------------------------

        U, mw_p = mannwhitneyu(

            scores_R,

            scores_NR,

            alternative="two-sided"
        )

        # ----------------------------------------------------
        # Effect
        # ----------------------------------------------------

        effect = (
            rank_biserial_effect(
                scores_R,
                scores_NR
            )
        )

        # ----------------------------------------------------
        # Bootstrap CI
        # ----------------------------------------------------

        (
            ci_low,
            ci_high
        ) = bootstrap_median_difference(

            scores_R,

            scores_NR,

            n_bootstraps=(
                N_BOOTSTRAPS
            ),

            seed=RANDOM_SEED
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({

            "analysis": label,

            "min_cells": threshold,

            "n_patients": len(scores),

            "n_R": len(scores_R),

            "n_NR": len(scores_NR),

            "median_R": np.median(
                scores_R
            ),

            "median_NR": np.median(
                scores_NR
            ),

            "median_difference_R_minus_NR": (
                observed_difference
            ),

            "permutation_p": (
                permutation_p
            ),

            "mannwhitney_U": U,

            "mannwhitney_p": (
                mw_p
            ),

            "rank_biserial_effect": (
                effect
            ),

            "bootstrap_CI_low": (
                ci_low
            ),

            "bootstrap_CI_high": (
                ci_high
            )
        })

        print(
            f"Patients: {len(scores)}"
        )

        print(
            f"R={len(scores_R)}, "
            f"NR={len(scores_NR)}"
        )

        print(
            f"Permutation p="
            f"{permutation_p:.5f}"
        )

        print(
            f"Effect="
            f"{effect:.4f}"
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------

    results = pd.DataFrame(
        results
    )

    output_file = os.path.join(

        OUTPUT_DIR,

        f"sensitivity_analysis_{label}.tsv"
    )

    results.to_csv(

        output_file,

        sep="\t",

        index=False
    )

    print()
    print(
        "Saved:",
        output_file
    )

    return results


# ============================================================
# RUN SENSITIVITY ANALYSIS
# ============================================================

sensitivity_all_T = (
    sensitivity_analysis(

        patient_expr_all_T,

        cell_counts_all_T,

        "All_T",

        MIN_CELLS_THRESHOLDS
    )
)

sensitivity_infusion_T = (
    sensitivity_analysis(

        patient_expr_infusion_T,

        cell_counts_infusion_T,

        "Infusion_T",

        MIN_CELLS_THRESHOLDS
    )
)


# ============================================================
# PRINT SENSITIVITY TABLES
# ============================================================

print()
print()
print(
    "=" * 80
)

print(
    "SENSITIVITY ANALYSIS — ALL T"
)

print(
    "=" * 80
)

print(
    sensitivity_all_T.to_string(
        index=False
    )
)


print()
print()
print(
    "=" * 80
)

print(
    "SENSITIVITY ANALYSIS — INFUSION T"
)

print(
    "=" * 80
)

print(
    sensitivity_infusion_T.to_string(
        index=False
    )
)


# ============================================================
# FINISHED
# ============================================================

print()
print()
print(
    "=" * 80
)

print(
    "SIGNATURE VALIDATION COMPLETED"
)

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print(
    "=" * 80
)

