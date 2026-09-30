import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve


parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    required=True
) # scRNAseq data


parser.add_argument(
    "--metadata",
    required=True
)

parser.add_argument(
    "--signature",
    required=True
) # cart_response_signature

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

responder_genes = signature.loc[
    signature["direction"] == "responder_UP",
    "gene"
].tolist()

# FIND 10X MATRICES

samples = sorted(
    [
        x for x in os.listdir(args.input_dir)
        if (
            "Infusion" in x
            and os.path.isdir(
                os.path.join(args.input_dir, x)
            )
            and os.path.exists(
                os.path.join(
                    args.input_dir,
                    x,
                    "matrix.mtx.gz"
                )
            )
        )
    ]
)

print("\n10X samples:")

for x in samples:
    print(x)


patient_results = []

# PROCESS EACH PATIENT

for sample in samples:

    sample_dir = os.path.join(
        args.input_dir,
        sample
    )


    adata = sc.read_10x_mtx(
        sample_dir,
        var_names="gene_symbols",
        make_unique=True
    )

    print(
        "Cells:",
        adata.n_obs,
        "Genes:",
        adata.n_vars
    )


    # --------------------------------------------------------
    # MATCH SIGNATURE GENES
    # --------------------------------------------------------

    available = set(adata.var_names)

    resp = [
        g for g in responder_genes
        if g in available
    ]


    print(
        f"Responder genes available: "
        f"{len(resp)}/{len(responder_genes)}"
    )



    if len(resp) < 3:

        print(
            "WARNING: too few signature genes."
        )

        continue


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    sc.pp.normalize_total(
        adata,
        target_sum=1e4
    )

    sc.pp.log1p(adata)


    # --------------------------------------------------------
    # Z-SCORE EACH GENE ACROSS CELLS
    # --------------------------------------------------------

    genes = resp

    X = adata[:, genes].X

    if hasattr(X, "toarray"):
        X = X.toarray()

    X = np.asarray(X)


    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1

    Xz = (X - mean) / std


    n_resp = len(resp)

    responder_score = Xz[:, :n_resp].mean(axis=1)

    adata.obs["signature_score"] = (
        responder_score
    )


    # --------------------------------------------------------
    # PATIENT-LEVEL SCORE
    # --------------------------------------------------------

    patient_score = np.median(
        adata.obs["signature_score"]
    )


    patient = sample.replace(
        "-Infusion",
        ""
    )


    patient_results.append({

        "patient_id": patient,

        "sample_id": sample,

        "n_cells": adata.n_obs,

        "n_responder_genes": len(resp),

        "signature_score": patient_score

    })


# SAVE

results = pd.DataFrame(
    patient_results
)

results.to_csv(os.path.join(
    args.output_dir, "signature_score_scRNAseq.csv"),
    sep="\t",
    index=False
)


#Merge score and metada??

df = results.merge(
    metadata,
    on="patient_id",
    how="inner"
)


# GROUPS

responder = df.loc[
    df["response"] == "R",
    "signature_score"
]

nonresponder = df.loc[
    df["response"] == "NR",
    "signature_score"
]


print("\nMerged metadata:")
print(df)

print("\nResponse counts:")
print(df["response"].value_counts())

if df.empty:
    raise ValueError(
        "No samples matched between signature scores and metadata."
    )

if df["response"].nunique() < 2:
    raise ValueError(
        "Only one response group is present."
    )

# MANN-WHITNEY

stat, p = mannwhitneyu(
    responder,
    nonresponder,
    alternative="two-sided"
)


print("\nMann–Whitney U")
print("U =", stat)
print("p =", p)

# AUC

y = (
    df["response"] == "R"
).astype(int)

x = df["signature_score"]


auc = roc_auc_score(
    y,
    x
)


print("\nAUC =", auc)


# ROC

fpr, tpr, thresholds = roc_curve(
    y,
    x
)


plt.figure(figsize=(5, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title(
    "GSE197268 — signature validation"
)

plt.legend()

plt.tight_layout()

plt.savefig(os.path.join(
    args.figures_dir, 'ROC_GSE197268.png'
))

plt.close()


# ============================================================
# BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

sns.boxplot(
    data=df,
    x="response",
    y="signature_score"
)

sns.stripplot(
    data=df,
    x="response",
    y="signature_score",
    color="black",
    alpha=0.7,
    size=5
)

plt.xlabel("")
plt.ylabel("Signature score")

plt.title(
    f"GSE197268\n"
    f"Mann–Whitney p = {p:.3g}; "
    f"AUC = {auc:.3f}"
)

plt.tight_layout()


plt.savefig(os.path.join(
    args.figures_dir, 'signature_boxplot_GSE197268.png'
))

plt.close()


# ============================================================
# SAVE STATISTICS
# ============================================================

summary = pd.DataFrame({

    "dataset": ["GSE197268"],

    "n_total": [len(df)],

    "n_responder": [len(responder)],

    "n_nonresponder": [len(nonresponder)],

    "median_responder": [
        np.median(responder)
    ],

    "median_nonresponder": [
        np.median(nonresponder)
    ],

    "mann_whitney_U": [stat],

    "mann_whitney_p": [p],

    "AUC": [auc]

})


summary.to_csv(os.path.join(
    args.output_dir, "validation_statistics.tsv"),
    sep="\t",
    index=False
)
