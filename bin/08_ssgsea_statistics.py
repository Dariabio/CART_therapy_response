import os
import argparse
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# PATHS

project_dir = "/Users/darakuzmina/Desktop/CAR_T"

results_dir = os.path.join(
    project_dir,
    "results"
)

dir_proc = os.path.join(
    project_dir,
    "data",
    "processed"
)


# COMMAND LINE ARGUMENT


parser = argparse.ArgumentParser(
    description="ssGSEA statistical comparison"
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

args = parser.parse_args()

comparison = args.comparison

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

# LOAD DATA

meta = pd.read_csv(
    os.path.join(
        dir_proc,
        "meta_analysis.tsv"
    ),
    sep="\t",
    index_col=0
)

scores = pd.read_csv(
    os.path.join(
        results_dir,
        "ssgsea",
        "scores_all.csv"
    ),
    sep=",",
    index_col=0
)


print("\nLoaded data:")
print(f"Metadata: {meta.shape}")
print(f"ssGSEA:   {scores.shape}")

# MATCH SAMPLES


common_samples = meta.index.intersection(
    scores.index
)

meta = meta.loc[
    common_samples
].copy()

scores = scores.loc[
    common_samples
].copy()


assert meta.index.equals(
    scores.index
)


# TEST TWO GROUPS

def test_ssgsea_comparison(
    meta_df,
    scores_df,
    comparison_column,
    positive,
    negative,
    timepoint,
    output_dir
):

    meta = meta_df.copy()
    scores = scores_df.copy()

    results = []

    # Test every ssGSEA signature

    for signature in scores.columns:

        positive_values = scores.loc[
            meta[comparison_column] == positive,
            signature
        ].dropna()

        negative_values = scores.loc[
            meta[comparison_column] == negative,
            signature
        ].dropna()


        # Need at least 2 samples per group

        if (
            len(positive_values) < 2
            or
            len(negative_values) < 2
        ):
            continue


        # Skip if both groups have zero variance

        if (
            positive_values.std() == 0
            and
            negative_values.std() == 0
        ):
            continue

        # Mann-Whitney U test

        statistic, p_value = mannwhitneyu(
            positive_values,
            negative_values,
            alternative="two-sided"
        )

        # Store result

        results.append({

            "subpopulation": signature,

            "n_positive":
                len(positive_values),

            "n_negative":
                len(negative_values),

            "mean_positive":
                positive_values.mean(),

            "mean_negative":
                negative_values.mean(),

            "median_positive":
                positive_values.median(),

            "median_negative":
                negative_values.median(),

            "effect_positive_minus_negative":
                (
                    positive_values.mean()
                    -
                    negative_values.mean()
                ),

            "p_value":
                p_value

        })

    # Convert to DataFrame

    result_df = pd.DataFrame(
        results
    )

    # FDR CORRECTION

    reject, fdr_corrected, _, _ = multipletests(
        result_df["p_value"],
        alpha=0.1,
        method="fdr_bh"
    )

    result_df["FDR"] = fdr_corrected

    # SORT RESULTS

    result_df = result_df.sort_values(
        by=[
            "FDR",
            "p_value"
        ]
    )

    # SAVE RESULTS

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    output_file = os.path.join(
        output_dir,
        f"ssGSEA_{timepoint}.tsv"
    )


    result_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )


    print(
        f"\nSaved: {output_file}"
    )


    return result_df

# OUTPUT DIRECTORY FOR CURRENT COMPARISON

output_dir = os.path.join(
    results_dir,
    "ssgsea",
    comparison
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# RUN ANALYSIS FOR EACH TIMEPOINT

TIMEPOINTS = {

    "product": "product",

    "d7": "d7 after",

    "d28": "d28 after"

}


all_results = {}


for short_name, timepoint in TIMEPOINTS.items():


    # Select current timepoint


    meta_tp = meta[
        meta["timepoint"] == timepoint
    ].copy()

    # Keep only requested comparison groups

    meta_tp = meta_tp[
        meta_tp[comparison_column].isin(
            [
                positive,
                negative
            ]
        )
    ].copy()

    # Match scores to metadata

    common_tp = meta_tp.index.intersection(
        scores.index
    )


    meta_tp = meta_tp.loc[
        common_tp
    ]

    scores_tp = scores.loc[
        common_tp
    ]

    # Check sample matching

    assert meta_tp.index.equals(
        scores_tp.index
    )


    # Check both groups

    n_positive = (
        meta_tp[comparison_column]
        .eq(positive)
        .sum()
    )

    n_negative = (
        meta_tp[comparison_column]
        .eq(negative)
        .sum()
    )


    if n_positive < 2:

        print(
            f"SKIP {timepoint}: "
            f"not enough {positive} samples"
        )

        continue


    if n_negative < 2:

        print(
            f"SKIP {timepoint}: "
            f"not enough {negative} samples"
        )

        continue

    # Run statistical analysis

    result = test_ssgsea_comparison(

        meta_df=meta_tp,

        scores_df=scores_tp,

        comparison_column=comparison_column,

        positive=positive,

        negative=negative,

        timepoint=short_name,

        output_dir=output_dir

    )


    all_results[short_name] = result

