#%%
# Classify groups 3 and 4 using ssGSEA scores
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
#%%
project_dir = "/Users/darakuzmina/Desktop/CAR_T"
ann_dir = "/Users/darakuzmina/Desktop/crc_metastasis_analysis/data"
results_dir = os.path.join(project_dir, "results") 
figures_dir = os.path.join(project_dir, "figures") 
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

dir_raw = os.path.join(project_dir, "data", "raw")
dir_proc = os.path.join(project_dir, "data", "processed")
#%%
# ============================================================
# 1. LOAD DATA
# ============================================================

scores_all = pd.read_csv(
    os.path.join(
        results_dir,
        "results_gsea",
        "scores_all.csv"
    ),
    index_col=0
)

meta_all = pd.read_csv(
    os.path.join(
        dir_proc,
        "meta_all.csv"
    ),
    index_col=0
)


# ============================================================
# 2. MATCH SAMPLES
# ============================================================

common_samples = scores_all.index.intersection(meta_all.index)

scores_all = scores_all.loc[common_samples]
meta_all = meta_all.loc[common_samples]

assert list(scores_all.index) == list(meta_all.index)


# ============================================================
# 3. SPLIT BY TIMEPOINT
# ============================================================

timepoints = [
    "product",
    "d7 after",
    "d28 after"
]

scores_known = {}
meta_known = {}

scores_unknown = {}
meta_unknown = {}


for tp in timepoints:

    # samples from current timepoint
    tp_mask = meta_all["timepoint"] == tp

    meta_tp = meta_all.loc[tp_mask].copy()
    scores_tp = scores_all.loc[tp_mask].copy()

    # Known:
    # responder / nonresponder
    known_mask = meta_tp["group_analysis_1"].isin(
        ["responder", "nonresponder"]
    )

    # Unknown:
    # groups 3 / 4
    unknown_mask = meta_tp["group_analysis_2"].isin(
        [3, 4]
    )

    scores_known[tp] = scores_tp.loc[known_mask].copy()
    meta_known[tp] = meta_tp.loc[known_mask].copy()

    scores_unknown[tp] = scores_tp.loc[unknown_mask].copy()
    meta_unknown[tp] = meta_tp.loc[unknown_mask].copy()


# ============================================================
# 4. LOGISTIC REGRESSION
#    separately for each timepoint
# ============================================================

classification_results = {}


for tp in timepoints:

    print("\n" + "=" * 70)
    print(f"TIMEPOINT: {tp}")
    print("=" * 70)

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    X_train = scores_known[tp].copy()

    y_train = (
        meta_known[tp]["group_analysis_1"]
        .map({
            "nonresponder": 0,
            "responder": 1
        })
    )

    # Unknown samples
    X_unknown = scores_unknown[tp].copy()
    meta_unknown_tp = meta_unknown[tp].copy()

    print("\nTraining:")
    print(y_train.value_counts())

    print("\nUnknown:")
    print(
        meta_unknown_tp["group_analysis_2"].value_counts()
    )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if len(X_train) < 6:
        print("Too few training samples. Skipping.")
        continue

    if y_train.nunique() < 2:
        print("Only one class in training data. Skipping.")
        continue

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "logistic",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    min_class_n = y_train.value_counts().min()

    if min_class_n >= 2:

        n_splits = min(5, min_class_n)

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy"
        )

        print(
            f"\nCV accuracy: "
            f"{cv_scores.mean():.3f} ± "
            f"{cv_scores.std():.3f}"
        )

        print(
            "CV scores:",
            np.round(cv_scores, 3)
        )

    else:

        print(
            "\nNot enough samples for cross-validation."
        )

    # --------------------------------------------------------
    # Fit on ALL known samples
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict groups 3/4
    # --------------------------------------------------------

    if len(X_unknown) == 0:

        print("\nNo groups 3/4 at this timepoint.")
        continue

    predictions = model.predict(X_unknown)

    probabilities = model.predict_proba(X_unknown)

    # --------------------------------------------------------
    # Result table
    # --------------------------------------------------------

    result = meta_unknown_tp.copy()

    result["predicted_group"] = np.where(
        predictions == 1,
        "responder",
        "nonresponder"
    )

    result["prob_responder"] = probabilities[:, 1]

    result["prob_nonresponder"] = probabilities[:, 0]

    result["confidence"] = np.maximum(
        probabilities[:, 0],
        probabilities[:, 1]
    )

    result["sample_id"] = result.index

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    result = result.sort_values(
        ["patient_id", "confidence"],
        ascending=[True, False]
    )

    classification_results[tp] = result

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nClassification:")

    print(
        result[
            [
                "sample_id",
                "patient_id",
                "group_analysis_2",
                "predicted_group",
                "prob_responder",
                "prob_nonresponder",
                "confidence"
            ]
        ]
    )


# ============================================================
# 5. KEEP ONLY NECESSARY COLUMNS
# ============================================================

dfs = []


for tp in timepoints:

    if tp not in classification_results:
        continue

    df = classification_results[tp].copy()

    # keep patient + prediction + probability
    df = df[
        [
            "patient_id",
            "predicted_group",
            "prob_responder"
        ]
    ].copy()

    # make unique column names
    tp_name = (
        tp.replace(" ", "_")
          .replace("-", "_")
    )

    df = df.rename(
        columns={
            "predicted_group":
                f"predicted_group_{tp_name}",

            "prob_responder":
                f"prob_responder_{tp_name}"
        }
    )

    dfs.append(df)


# ============================================================
# 6. MERGE BY PATIENT
# ============================================================

classification_combined = dfs[0].copy()

for df in dfs[1:]:

    classification_combined = classification_combined.merge(
        df,
        on="patient_id",
        how="outer"
    )


meta_fil = meta_all[["day_28", "month_2", "month_add","patient_id", "group_analysis_2"]]
merged_df = pd.merge(classification_combined, meta_fil, on='patient_id').drop_duplicates()
merged_df
merged_df.to_csv(os.path.join(results_dir, "predictions_3_4gr.csv"))
