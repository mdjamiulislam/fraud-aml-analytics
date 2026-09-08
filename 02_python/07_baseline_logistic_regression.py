import pandas as pd
import numpy as np
from pathlib import Path

import pyarrow.parquet as pq
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)


# ============================================================
# PROJECT 5 - STEP 8
# BASELINE LOGISTIC REGRESSION FRAUD MODEL
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "model_high_risk_population.parquet"
)

MASTER_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "master_analytical.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"

DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "step8_model"
)

CHART_DIR = (
    OUTPUT_DIR
    / "step8_charts"
)

MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


SAVED_MODEL_FILE = (
    MODEL_DIR
    / "baseline_logistic_regression.joblib"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "step8_test_predictions.parquet"
)

REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_8_Baseline_Logistic_Regression_Report.txt"
)


# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000

TRAIN_TARGET_PCT = 0.70

VALIDATION_TARGET_PCT = 0.10

RANDOM_STATE = 42


HIGH_RISK_TYPES = [
    "CASH_OUT",
    "TRANSFER"
]


# ------------------------------------------------------------
# 3. Columns required from modelling population
# ------------------------------------------------------------

LOAD_COLUMNS = [

    "transaction_id",

    "step",
    "hour_of_day",

    "type",

    "log_amount",

    "oldbalanceOrg",
    "oldbalanceDest",

    "high_value_200k_flag",
    "origin_zero_before_flag",
    "destination_zero_before_flag",

    "isFraud"
]


# ------------------------------------------------------------
# 4. Final baseline feature names
# ------------------------------------------------------------

CONTINUOUS_FEATURES = [

    "log_amount",

    "log_oldbalanceOrg",
    "log_oldbalanceDest",

    "hour_sin",
    "hour_cos"
]


BINARY_FEATURES = [

    "type_transfer_flag",

    "high_value_200k_flag",

    "origin_zero_before_flag",

    "destination_zero_before_flag"
]


MODEL_FEATURES = (
    CONTINUOUS_FEATURES
    + BINARY_FEATURES
)


print("=" * 80)
print("PROJECT 5 - STEP 8")
print("BASELINE LOGISTIC REGRESSION FRAUD MODEL")
print("=" * 80)


# ============================================================
# PART A
# DETERMINE ROW-BALANCED CHRONOLOGICAL SPLIT
# ============================================================


# ------------------------------------------------------------
# 5. Count high-risk rows by simulation step
# ------------------------------------------------------------

print("\nDetermining chronological split boundaries...")


parquet_file = pq.ParquetFile(
    MODEL_FILE
)


step_counts = pd.Series(
    dtype="int64"
)


for batch in parquet_file.iter_batches(

    batch_size=BATCH_SIZE,

    columns=["step"]

):

    batch_df = batch.to_pandas()

    batch_counts = (
        batch_df["step"]
        .value_counts()
    )

    step_counts = step_counts.add(
        batch_counts,
        fill_value=0
    )


step_counts = (
    step_counts
    .sort_index()
    .astype("int64")
)


total_rows = int(
    step_counts.sum()
)


step_profile = pd.DataFrame({

    "step":
        step_counts.index.astype(int),

    "row_count":
        step_counts.values
})


step_profile[
    "cumulative_rows"
] = (

    step_profile[
        "row_count"
    ].cumsum()

)


step_profile[
    "cumulative_pct"
] = (

    step_profile[
        "cumulative_rows"
    ]
    / total_rows

)


# ------------------------------------------------------------
# 6. Find boundary nearest 70%
# ------------------------------------------------------------

train_target_rows = (
    total_rows
    * TRAIN_TARGET_PCT
)


train_boundary_index = (

    (
        step_profile[
            "cumulative_rows"
        ]
        - train_target_rows
    )
    .abs()
    .idxmin()

)


TRAIN_END_STEP = int(

    step_profile.loc[
        train_boundary_index,
        "step"
    ]

)


# ------------------------------------------------------------
# 7. Find boundary nearest 80%
#
# 70% train + 10% validation
# ------------------------------------------------------------

validation_end_target_rows = (

    total_rows
    * (
        TRAIN_TARGET_PCT
        + VALIDATION_TARGET_PCT
    )

)


validation_boundary_candidates = (

    step_profile.loc[
        step_profile["step"]
        .gt(TRAIN_END_STEP)
    ]
)


validation_boundary_index = (

    (
        validation_boundary_candidates[
            "cumulative_rows"
        ]
        - validation_end_target_rows
    )
    .abs()
    .idxmin()

)


VALIDATION_END_STEP = int(

    step_profile.loc[
        validation_boundary_index,
        "step"
    ]

)


print("\nSelected chronological boundaries:")

print(
    f"TRAIN      : step <= "
    f"{TRAIN_END_STEP}"
)

print(
    f"VALIDATION : step "
    f"{TRAIN_END_STEP + 1}"
    f" to "
    f"{VALIDATION_END_STEP}"
)

print(
    f"TEST       : step > "
    f"{VALIDATION_END_STEP}"
)


# ============================================================
# PART B
# BUILD MEMORY-EFFICIENT MODEL DATA
# ============================================================


# ------------------------------------------------------------
# 8. Feature engineering function
# ------------------------------------------------------------

def prepare_model_batch(df):

    output = pd.DataFrame()


    # Traceability

    output[
        "transaction_id"
    ] = df[
        "transaction_id"
    ].astype("int64")


    output[
        "step"
    ] = df[
        "step"
    ].astype("int16")


    # --------------------------------------------------------
    # Transaction type
    # TRANSFER = 1
    # CASH_OUT = 0
    # --------------------------------------------------------

    output[
        "type_transfer_flag"
    ] = (

        df["type"]
        .eq("TRANSFER")
        .astype("int8")

    )


    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    output[
        "log_amount"
    ] = (

        df[
            "log_amount"
        ]
        .astype("float32")

    )


    # --------------------------------------------------------
    # Pre-transaction balances
    # --------------------------------------------------------

    output[
        "log_oldbalanceOrg"
    ] = (

        np.log1p(
            df[
                "oldbalanceOrg"
            ]
        )
        .astype("float32")

    )


    output[
        "log_oldbalanceDest"
    ] = (

        np.log1p(
            df[
                "oldbalanceDest"
            ]
        )
        .astype("float32")

    )


    # --------------------------------------------------------
    # Cyclical simulated-hour encoding
    # --------------------------------------------------------

    hour = (
        df[
            "hour_of_day"
        ]
        .astype("float32")
    )


    output[
        "hour_sin"
    ] = (

        np.sin(
            2
            * np.pi
            * hour
            / 24
        )
        .astype("float32")

    )


    output[
        "hour_cos"
    ] = (

        np.cos(
            2
            * np.pi
            * hour
            / 24
        )
        .astype("float32")

    )


    # --------------------------------------------------------
    # Binary risk indicators
    # --------------------------------------------------------

    for col in [

        "high_value_200k_flag",

        "origin_zero_before_flag",

        "destination_zero_before_flag"

    ]:

        output[col] = (

            df[col]
            .astype("int8")

        )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    output[
        "isFraud"
    ] = (

        df[
            "isFraud"
        ]
        .astype("int8")

    )


    return output


# ------------------------------------------------------------
# 9. Read high-risk Parquet in batches
# ------------------------------------------------------------

print("\nPreparing model datasets...")


train_parts = []
validation_parts = []
test_parts = []


rows_processed = 0


for batch_number, batch in enumerate(

    parquet_file.iter_batches(

        batch_size=BATCH_SIZE,

        columns=LOAD_COLUMNS

    ),

    start=1

):

    raw_df = batch.to_pandas()

    prepared_df = prepare_model_batch(
        raw_df
    )


    train_mask = (

        prepared_df[
            "step"
        ]
        .le(TRAIN_END_STEP)

    )


    validation_mask = (

        prepared_df[
            "step"
        ]
        .gt(TRAIN_END_STEP)

        &

        prepared_df[
            "step"
        ]
        .le(
            VALIDATION_END_STEP
        )

    )


    test_mask = (

        prepared_df[
            "step"
        ]
        .gt(
            VALIDATION_END_STEP
        )

    )


    if train_mask.any():

        train_parts.append(

            prepared_df.loc[
                train_mask
            ]

        )


    if validation_mask.any():

        validation_parts.append(

            prepared_df.loc[
                validation_mask
            ]

        )


    if test_mask.any():

        test_parts.append(

            prepared_df.loc[
                test_mask
            ]

        )


    rows_processed += len(
        prepared_df
    )


    print(

        f"Batch {batch_number} "
        f"| cumulative rows "
        f"{rows_processed:,}"

    )


# ------------------------------------------------------------
# 10. Combine partitions
# ------------------------------------------------------------

train_df = pd.concat(
    train_parts,
    ignore_index=True
)

validation_df = pd.concat(
    validation_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


# Release temporary lists

del train_parts
del validation_parts
del test_parts


# ------------------------------------------------------------
# 11. Split validation summary
# ------------------------------------------------------------

def split_summary_row(
    name,
    df
):

    fraud_count = int(
        df[
            "isFraud"
        ].sum()
    )

    rows = len(df)

    return {

        "Split":
            name,

        "Rows":
            rows,

        "Population_Pct":
            rows
            / total_rows
            * 100,

        "Fraud_Count":
            fraud_count,

        "Fraud_Rate_Pct":

            fraud_count
            / rows
            * 100

            if rows > 0
            else 0,

        "Minimum_Step":
            int(
                df["step"].min()
            ),

        "Maximum_Step":
            int(
                df["step"].max()
            )
    }


split_summary_df = pd.DataFrame([

    split_summary_row(
        "TRAIN",
        train_df
    ),

    split_summary_row(
        "VALIDATION",
        validation_df
    ),

    split_summary_row(
        "TEST",
        test_df
    )

])


split_summary_df.to_csv(

    OUTPUT_DIR
    / "step8_temporal_split_summary.csv",

    index=False

)


print("\n" + "=" * 80)
print("FINAL TEMPORAL SPLIT")
print("=" * 80)

print(
    split_summary_df.to_string(
        index=False
    )
)


# ============================================================
# PART C
# PREPARE MODEL MATRICES
# ============================================================


# ------------------------------------------------------------
# 12. Separate X and y
# ------------------------------------------------------------

X_train = train_df[
    MODEL_FEATURES
]

y_train = train_df[
    "isFraud"
]


X_validation = validation_df[
    MODEL_FEATURES
]

y_validation = validation_df[
    "isFraud"
]


X_test = test_df[
    MODEL_FEATURES
]

y_test = test_df[
    "isFraud"
]


# ------------------------------------------------------------
# 13. Build preprocessing pipeline
# ------------------------------------------------------------

preprocessor = ColumnTransformer(

    transformers=[

        (

            "continuous",

            StandardScaler(),

            CONTINUOUS_FEATURES

        ),

        (

            "binary",

            "passthrough",

            BINARY_FEATURES

        )

    ],

    remainder="drop"

)


# ------------------------------------------------------------
# 14. Logistic Regression
# ------------------------------------------------------------

classifier = LogisticRegression(

    class_weight="balanced",

    solver="lbfgs",

    max_iter=500,

    random_state=RANDOM_STATE
)


model_pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        classifier
    )

])


# ============================================================
# PART D
# TRAIN MODEL
# ============================================================


print("\n" + "=" * 80)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 80)


model_pipeline.fit(
    X_train,
    y_train
)


print("\nTraining completed.")


# ------------------------------------------------------------
# 15. Save trained pipeline
# ------------------------------------------------------------

joblib.dump(

    model_pipeline,

    SAVED_MODEL_FILE

)


# ============================================================
# PART E
# VALIDATION THRESHOLD SELECTION
# ============================================================


print("\nSelecting threshold using validation data...")


validation_scores = (

    model_pipeline
    .predict_proba(
        X_validation
    )[:, 1]

)


precision_curve, recall_curve, thresholds = (

    precision_recall_curve(

        y_validation,

        validation_scores

    )

)


# precision and recall have one extra element
precision_for_threshold = (
    precision_curve[:-1]
)

recall_for_threshold = (
    recall_curve[:-1]
)


# ------------------------------------------------------------
# 16. F2 score
#
# beta = 2
# Recall receives greater weight
# ------------------------------------------------------------

beta_squared = 4


f2_curve = (

    (
        1
        + beta_squared
    )
    * precision_for_threshold
    * recall_for_threshold

    /

    (
        beta_squared
        * precision_for_threshold

        + recall_for_threshold

        + 1e-12
    )

)


best_index = int(
    np.nanargmax(
        f2_curve
    )
)


SELECTED_THRESHOLD = float(
    thresholds[
        best_index
    ]
)


selected_validation_precision = float(
    precision_for_threshold[
        best_index
    ]
)


selected_validation_recall = float(
    recall_for_threshold[
        best_index
    ]
)


selected_validation_f2 = float(
    f2_curve[
        best_index
    ]
)


print(
    "\nSelected validation threshold:"
)

print(
    f"{SELECTED_THRESHOLD:.6f}"
)

print(
    "\nValidation precision:"
)

print(
    f"{selected_validation_precision:.4f}"
)

print(
    "\nValidation recall:"
)

print(
    f"{selected_validation_recall:.4f}"
)

print(
    "\nValidation F2:"
)

print(
    f"{selected_validation_f2:.4f}"
)


# ------------------------------------------------------------
# 17. Save top threshold candidates
# ------------------------------------------------------------

threshold_df = pd.DataFrame({

    "Threshold":
        thresholds,

    "Precision":
        precision_for_threshold,

    "Recall":
        recall_for_threshold,

    "F2":
        f2_curve

})


top_thresholds_df = (

    threshold_df
    .sort_values(
        "F2",
        ascending=False
    )
    .head(50)
)


top_thresholds_df.to_csv(

    OUTPUT_DIR
    / "step8_top_validation_thresholds.csv",

    index=False

)


# ============================================================
# PART F
# TEST EVALUATION
# ============================================================


# ------------------------------------------------------------
# 18. Test model scores
# ------------------------------------------------------------

print("\nEvaluating untouched test period...")


test_scores = (

    model_pipeline
    .predict_proba(
        X_test
    )[:, 1]

)


# ------------------------------------------------------------
# 19. Evaluation function
# ------------------------------------------------------------

def classification_metrics(

    y_true,

    scores,

    threshold,

    model_name

):

    predictions = (

        scores
        >= threshold

    ).astype("int8")


    tn, fp, fn, tp = (

        confusion_matrix(

            y_true,

            predictions,

            labels=[0, 1]

        ).ravel()

    )


    accuracy = accuracy_score(
        y_true,
        predictions
    )


    precision = precision_score(

        y_true,
        predictions,
        zero_division=0

    )


    recall = recall_score(

        y_true,
        predictions,
        zero_division=0

    )


    f1 = f1_score(

        y_true,
        predictions,
        zero_division=0

    )


    f2 = fbeta_score(

        y_true,
        predictions,

        beta=2,

        zero_division=0

    )


    roc_auc = roc_auc_score(

        y_true,
        scores

    )


    average_precision = (

        average_precision_score(

            y_true,
            scores

        )

    )


    alert_count = int(
        predictions.sum()
    )


    alert_rate = (

        alert_count
        / len(predictions)
        * 100

    )


    false_positive_rate = (

        fp
        / (fp + tn)
        * 100

        if (fp + tn) > 0
        else 0

    )


    specificity = (

        tn
        / (tn + fp)

        if (tn + fp) > 0
        else 0

    )


    return {

        "Model":
            model_name,

        "Threshold":
            threshold,

        "ROC_AUC":
            roc_auc,

        "Average_Precision":
            average_precision,

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "F2":
            f2,

        "Specificity":
            specificity,

        "True_Negative":
            int(tn),

        "False_Positive":
            int(fp),

        "False_Negative":
            int(fn),

        "True_Positive":
            int(tp),

        "Alerts":
            alert_count,

        "Alert_Rate_Pct":
            alert_rate,

        "False_Positive_Rate_Pct":
            false_positive_rate

    }


# ------------------------------------------------------------
# 20. Test at default threshold 0.50
# ------------------------------------------------------------

default_metrics = classification_metrics(

    y_test,

    test_scores,

    0.50,

    "Logistic Regression - 0.50 Threshold"

)


# ------------------------------------------------------------
# 21. Test at validation-selected threshold
# ------------------------------------------------------------

selected_metrics = classification_metrics(

    y_test,

    test_scores,

    SELECTED_THRESHOLD,

    "Logistic Regression - Validation F2 Threshold"

)


# ============================================================
# PART G
# EXISTING RULE BENCHMARK ON SAME TEST PERIOD
# ============================================================


print(
    "\nEvaluating existing isFlaggedFraud "
    "rule on the same test period..."
)


master_parquet = pq.ParquetFile(
    MASTER_FILE
)


rule_tn = 0
rule_fp = 0
rule_fn = 0
rule_tp = 0

rule_test_rows = 0


for batch in master_parquet.iter_batches(

    batch_size=BATCH_SIZE,

    columns=[

        "step",
        "type",
        "isFraud",
        "isFlaggedFraud"

    ]

):

    df = batch.to_pandas()


    rule_test_mask = (

        df["type"]
        .isin(
            HIGH_RISK_TYPES
        )

        &

        df["step"]
        .gt(
            VALIDATION_END_STEP
        )

    )


    rule_df = df.loc[
        rule_test_mask
    ]


    if len(rule_df) == 0:
        continue


    y_true = (

        rule_df[
            "isFraud"
        ].to_numpy()

    )


    y_pred = (

        rule_df[
            "isFlaggedFraud"
        ].to_numpy()

    )


    tn, fp, fn, tp = (

        confusion_matrix(

            y_true,

            y_pred,

            labels=[0, 1]

        ).ravel()

    )


    rule_tn += int(tn)
    rule_fp += int(fp)
    rule_fn += int(fn)
    rule_tp += int(tp)

    rule_test_rows += len(
        rule_df
    )


# ------------------------------------------------------------
# 22. Existing rule metrics
# ------------------------------------------------------------

rule_precision = (

    rule_tp
    / (
        rule_tp
        + rule_fp
    )

    if (
        rule_tp
        + rule_fp
    ) > 0

    else 0

)


rule_recall = (

    rule_tp
    / (
        rule_tp
        + rule_fn
    )

    if (
        rule_tp
        + rule_fn
    ) > 0

    else 0

)


rule_f1 = (

    2
    * rule_precision
    * rule_recall

    / (
        rule_precision
        + rule_recall
    )

    if (
        rule_precision
        + rule_recall
    ) > 0

    else 0

)


rule_f2 = (

    5
    * rule_precision
    * rule_recall

    / (
        4
        * rule_precision
        + rule_recall
    )

    if (
        4
        * rule_precision
        + rule_recall
    ) > 0

    else 0

)


rule_alerts = (
    rule_tp
    + rule_fp
)


rule_metrics = {

    "Model":
        "Existing isFlaggedFraud Rule",

    "Threshold":
        np.nan,

    "ROC_AUC":
        np.nan,

    "Average_Precision":
        np.nan,

    "Accuracy":

        (
            rule_tp
            + rule_tn
        )
        / rule_test_rows,

    "Precision":
        rule_precision,

    "Recall":
        rule_recall,

    "F1":
        rule_f1,

    "F2":
        rule_f2,

    "Specificity":

        rule_tn
        / (
            rule_tn
            + rule_fp
        )

        if (
            rule_tn
            + rule_fp
        ) > 0

        else 0,

    "True_Negative":
        rule_tn,

    "False_Positive":
        rule_fp,

    "False_Negative":
        rule_fn,

    "True_Positive":
        rule_tp,

    "Alerts":
        rule_alerts,

    "Alert_Rate_Pct":

        rule_alerts
        / rule_test_rows
        * 100,

    "False_Positive_Rate_Pct":

        rule_fp
        / (
            rule_fp
            + rule_tn
        )
        * 100

        if (
            rule_fp
            + rule_tn
        ) > 0

        else 0

}


# ------------------------------------------------------------
# 23. Combine model comparison
# ------------------------------------------------------------

comparison_df = pd.DataFrame([

    default_metrics,

    selected_metrics,

    rule_metrics

])


comparison_df.to_csv(

    OUTPUT_DIR
    / "step8_model_comparison.csv",

    index=False

)


# ============================================================
# PART H
# COEFFICIENT INTERPRETATION
# ============================================================


# ------------------------------------------------------------
# 24. Get feature names after preprocessing
# ------------------------------------------------------------

preprocessor_fitted = (

    model_pipeline
    .named_steps[
        "preprocessor"
    ]

)


classifier_fitted = (

    model_pipeline
    .named_steps[
        "classifier"
    ]

)


feature_names = (

    preprocessor_fitted
    .get_feature_names_out()

)


coefficients = (

    classifier_fitted
    .coef_[0]

)


coefficient_df = pd.DataFrame({

    "Feature":
        feature_names,

    "Coefficient":
        coefficients,

    "Odds_Ratio":
        np.exp(
            np.clip(
                coefficients,
                -20,
                20
            )
        )

})


coefficient_df[

    "Absolute_Coefficient"

] = (

    coefficient_df[
        "Coefficient"
    ].abs()

)


coefficient_df = (

    coefficient_df
    .sort_values(

        "Absolute_Coefficient",

        ascending=False

    )

)


coefficient_df.to_csv(

    OUTPUT_DIR
    / "step8_logistic_coefficients.csv",

    index=False

)


# ============================================================
# PART I
# SAVE TEST PREDICTIONS
# ============================================================


test_predictions_df = pd.DataFrame({

    "transaction_id":

        test_df[
            "transaction_id"
        ].to_numpy(),

    "step":

        test_df[
            "step"
        ].to_numpy(),

    "isFraud":

        y_test.to_numpy(),

    "fraud_score":

        test_scores,

    "predicted_fraud_050":

        (
            test_scores
            >= 0.50
        ).astype("int8"),

    "predicted_fraud_selected":

        (
            test_scores
            >= SELECTED_THRESHOLD
        ).astype("int8")

})


test_predictions_df.to_parquet(

    PREDICTION_FILE,

    index=False

)


# ============================================================
# PART J
# CHARTS
# ============================================================


# ------------------------------------------------------------
# 25. Validation precision-recall curve
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.plot(

    recall_curve,

    precision_curve

)


ax.scatter(

    selected_validation_recall,

    selected_validation_precision,

    s=60

)


ax.set_title(
    "Validation Precision-Recall Curve"
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "01_validation_precision_recall_curve.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 26. Test ROC curve
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(

    y_test,

    test_scores

)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.plot(
    fpr,
    tpr
)


ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


ax.set_title(
    "Baseline Logistic Regression - Test ROC Curve"
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "02_test_roc_curve.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 27. Test threshold trade-off
# ------------------------------------------------------------

sample_indices = np.linspace(

    0,

    len(thresholds) - 1,

    num=min(
        300,
        len(thresholds)
    ),

    dtype=int

)


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.plot(

    thresholds[
        sample_indices
    ],

    precision_for_threshold[
        sample_indices
    ],

    label="Precision"

)


ax.plot(

    thresholds[
        sample_indices
    ],

    recall_for_threshold[
        sample_indices
    ],

    label="Recall"

)


ax.axvline(

    SELECTED_THRESHOLD,

    linestyle="--"

)


ax.set_title(
    "Validation Threshold Trade-off"
)

ax.set_xlabel(
    "Classification Threshold"
)

ax.set_ylabel(
    "Metric"
)

ax.legend()


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "03_validation_threshold_tradeoff.png",

    dpi=160

)


plt.close(fig)


# ============================================================
# PART K
# REPORT
# ============================================================


with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as report:


    report.write(
        "PROJECT 5 - STEP 8\n"
    )

    report.write(
        "BASELINE LOGISTIC REGRESSION FRAUD MODEL\n"
    )

    report.write(
        "=" * 80
    )


    report.write(
        "\n\nTEMPORAL SPLIT\n\n"
    )

    report.write(

        split_summary_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nSELECTED VALIDATION THRESHOLD\n\n"
    )

    report.write(

        f"Threshold: "
        f"{SELECTED_THRESHOLD:.8f}\n"

    )

    report.write(

        f"Validation Precision: "
        f"{selected_validation_precision:.6f}\n"

    )

    report.write(

        f"Validation Recall: "
        f"{selected_validation_recall:.6f}\n"

    )

    report.write(

        f"Validation F2: "
        f"{selected_validation_f2:.6f}\n"

    )


    report.write(
        "\n\nTEST MODEL COMPARISON\n\n"
    )

    report.write(

        comparison_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nLOGISTIC REGRESSION COEFFICIENTS\n\n"
    )

    report.write(

        coefficient_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nMODEL GOVERNANCE NOTES\n\n"
    )


    report.write(

        "1. The model uses a chronological "
        "train/validation/test design.\n"

        "2. Split boundaries were selected "
        "using transaction counts while keeping "
        "complete simulation steps together.\n"

        "3. Threshold selection used validation "
        "data only. The final test set remained "
        "untouched until evaluation.\n"

        "4. Logistic Regression used "
        "class_weight='balanced' to address "
        "severe class imbalance.\n"

        "5. The selected operating threshold "
        "maximises validation F2, placing greater "
        "weight on fraud recall.\n"

        "6. Post-transaction balances, "
        "isFlaggedFraud, raw identifiers, "
        "zero_amount_flag and the highly "
        "simulation-specific "
        "amount_exceeds_origin_balance_flag "
        "were excluded from the baseline model.\n"

        "7. Because class weighting changes the "
        "probability scale, fraud_score should be "
        "treated as a model ranking score rather "
        "than a fully calibrated real-world "
        "fraud probability.\n"

    )


# ============================================================
# PART L
# DISPLAY FINAL RESULTS
# ============================================================


print("\n" + "=" * 80)
print("TEST MODEL COMPARISON")
print("=" * 80)


print(

    comparison_df[
        [

            "Model",

            "Threshold",

            "ROC_AUC",

            "Average_Precision",

            "Precision",

            "Recall",

            "F1",

            "F2",

            "True_Positive",

            "False_Positive",

            "False_Negative",

            "Alerts"

        ]
    ].to_string(
        index=False
    )

)


print("\n" + "=" * 80)
print("TOP LOGISTIC REGRESSION COEFFICIENTS")
print("=" * 80)


print(

    coefficient_df.head(
        15
    ).to_string(
        index=False
    )

)


print("\nSelected threshold:")

print(
    f"{SELECTED_THRESHOLD:.6f}"
)


print("\nSaved model:")

print(
    SAVED_MODEL_FILE
)


print("\nTest predictions:")

print(
    PREDICTION_FILE
)


print("\nReport:")

print(
    REPORT_FILE
)


print(
    "\nStep 8 completed successfully."
)
