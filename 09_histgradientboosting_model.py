import pandas as pd
import numpy as np
from pathlib import Path

import pyarrow.parquet as pq
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    precision_recall_curve,
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
# PROJECT 5 - STEP 10
# HISTGRADIENTBOOSTING FRAUD MODEL
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_POPULATION_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "model_high_risk_population.parquet"
)

SPLIT_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step8_temporal_split_summary.csv"
)

STEP9_COMPARISON_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step9_model_stress_test_comparison.csv"
)

STEP8_COMPARISON_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step8_model_comparison.csv"
)

MODEL_B_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step9_models"
    / "model_b_log_balances_only.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_outputs"
)

DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "step10_model"
)

CHART_DIR = (
    OUTPUT_DIR
    / "step10_charts"
)

MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


HGB_MODEL_FILE = (
    MODEL_DIR
    / "histgradientboosting_defensible_features.joblib"
)

TEST_PREDICTION_FILE = (
    OUTPUT_DIR
    / "step10_hgb_test_predictions.parquet"
)

REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_10_HistGradientBoosting_Model_Report.txt"
)


# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000

RANDOM_STATE = 42


print("=" * 82)
print("PROJECT 5 - STEP 10")
print("HISTGRADIENTBOOSTING FRAUD MODEL")
print("=" * 82)


# ============================================================
# PART A
# READ OFFICIAL TEMPORAL SPLIT
# ============================================================


# ------------------------------------------------------------
# 3. Read Step 8 split
# ------------------------------------------------------------

split_df = pd.read_csv(
    SPLIT_FILE
)


train_row = (
    split_df.loc[
        split_df["Split"].eq("TRAIN")
    ]
    .iloc[0]
)


validation_row = (
    split_df.loc[
        split_df["Split"].eq("VALIDATION")
    ]
    .iloc[0]
)


test_row = (
    split_df.loc[
        split_df["Split"].eq("TEST")
    ]
    .iloc[0]
)


TRAIN_END_STEP = int(
    train_row["Maximum_Step"]
)

VALIDATION_END_STEP = int(
    validation_row["Maximum_Step"]
)


EXPECTED_TRAIN_ROWS = int(
    train_row["Rows"]
)

EXPECTED_VALIDATION_ROWS = int(
    validation_row["Rows"]
)

EXPECTED_TEST_ROWS = int(
    test_row["Rows"]
)


print("\nOfficial Step 8 split:")

print(
    f"TRAIN      : step 1-{TRAIN_END_STEP}"
)

print(
    f"VALIDATION : step "
    f"{TRAIN_END_STEP + 1}-{VALIDATION_END_STEP}"
)

print(
    f"TEST       : step "
    f"{VALIDATION_END_STEP + 1}-743"
)


# ============================================================
# PART B
# PREPARE SAME SEVEN DEFENSIBLE FEATURES
# ============================================================


# ------------------------------------------------------------
# 4. Required source columns
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

    "isFraud"
]


# ------------------------------------------------------------
# 5. Official seven-feature set
# ------------------------------------------------------------

MODEL_FEATURES = [

    "type_transfer_flag",

    "log_amount",

    "log_oldbalanceOrg",
    "log_oldbalanceDest",

    "hour_sin",
    "hour_cos",

    "high_value_200k_flag"
]


# ------------------------------------------------------------
# 6. Feature engineering
# ------------------------------------------------------------

def prepare_batch(df):

    output = pd.DataFrame()


    # Traceability

    output["transaction_id"] = (
        df["transaction_id"]
        .astype("int64")
    )


    # Temporal split control

    output["step"] = (
        df["step"]
        .astype("int16")
    )


    # --------------------------------------------------------
    # Transaction type
    #
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
    # Transaction amount
    # --------------------------------------------------------

    output["log_amount"] = (

        df["log_amount"]
        .astype("float32")

    )


    # --------------------------------------------------------
    # Pre-transaction balances
    # --------------------------------------------------------

    output[
        "log_oldbalanceOrg"
    ] = (

        np.log1p(
            df["oldbalanceOrg"]
        )
        .astype("float32")

    )


    output[
        "log_oldbalanceDest"
    ] = (

        np.log1p(
            df["oldbalanceDest"]
        )
        .astype("float32")

    )


    # --------------------------------------------------------
    # Cyclical simulated hour
    # --------------------------------------------------------

    hour = (

        df["hour_of_day"]
        .astype("float32")

    )


    output["hour_sin"] = (

        np.sin(
            2
            * np.pi
            * hour
            / 24
        )
        .astype("float32")

    )


    output["hour_cos"] = (

        np.cos(
            2
            * np.pi
            * hour
            / 24
        )
        .astype("float32")

    )


    # --------------------------------------------------------
    # High-value indicator
    # --------------------------------------------------------

    output[
        "high_value_200k_flag"
    ] = (

        df[
            "high_value_200k_flag"
        ]
        .astype("int8")

    )


    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    output["isFraud"] = (

        df["isFraud"]
        .astype("int8")

    )


    return output


# ============================================================
# PART C
# LOAD MODEL POPULATION
# ============================================================


parquet_file = pq.ParquetFile(
    MODEL_POPULATION_FILE
)


train_parts = []
validation_parts = []
test_parts = []


rows_processed = 0


print("\nPreparing Step 10 datasets...\n")


for batch_number, batch in enumerate(

    parquet_file.iter_batches(

        batch_size=BATCH_SIZE,

        columns=LOAD_COLUMNS

    ),

    start=1

):

    raw_df = batch.to_pandas()

    df = prepare_batch(
        raw_df
    )


    # --------------------------------------------------------
    # Apply same chronological split
    # --------------------------------------------------------

    train_mask = (
        df["step"]
        .le(TRAIN_END_STEP)
    )


    validation_mask = (

        df["step"]
        .gt(TRAIN_END_STEP)

        &

        df["step"]
        .le(
            VALIDATION_END_STEP
        )

    )


    test_mask = (

        df["step"]
        .gt(
            VALIDATION_END_STEP
        )

    )


    if train_mask.any():

        train_parts.append(

            df.loc[
                train_mask
            ]

        )


    if validation_mask.any():

        validation_parts.append(

            df.loc[
                validation_mask
            ]

        )


    if test_mask.any():

        test_parts.append(

            df.loc[
                test_mask
            ]

        )


    rows_processed += len(df)


    print(

        f"Batch {batch_number} "
        f"| cumulative "
        f"{rows_processed:,}"

    )


# ------------------------------------------------------------
# 7. Combine partitions
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


del train_parts
del validation_parts
del test_parts


# ------------------------------------------------------------
# 8. Validate split
# ------------------------------------------------------------

assert len(train_df) == EXPECTED_TRAIN_ROWS

assert (
    len(validation_df)
    == EXPECTED_VALIDATION_ROWS
)

assert len(test_df) == EXPECTED_TEST_ROWS


print("\nSplit validation passed.")

print(
    f"TRAIN      : {len(train_df):,}"
)

print(
    f"VALIDATION : {len(validation_df):,}"
)

print(
    f"TEST       : {len(test_df):,}"
)


# ============================================================
# PART D
# X / y MATRICES
# ============================================================


X_train = (
    train_df[
        MODEL_FEATURES
    ]
)

y_train = (
    train_df[
        "isFraud"
    ]
)


X_validation = (
    validation_df[
        MODEL_FEATURES
    ]
)

y_validation = (
    validation_df[
        "isFraud"
    ]
)


X_test = (
    test_df[
        MODEL_FEATURES
    ]
)

y_test = (
    test_df[
        "isFraud"
    ]
)


# ============================================================
# PART E
# TRAIN HISTGRADIENTBOOSTING
# ============================================================


print("\n" + "=" * 82)

print(
    "TRAINING HISTGRADIENTBOOSTING"
)

print("=" * 82)


# ------------------------------------------------------------
# 9. Conservative model specification
# ------------------------------------------------------------

hgb_model = HistGradientBoostingClassifier(

    learning_rate=0.08,

    max_iter=200,

    max_leaf_nodes=31,

    min_samples_leaf=100,

    l2_regularization=1.0,

    class_weight="balanced",

    early_stopping=False,

    random_state=RANDOM_STATE
)


hgb_model.fit(
    X_train,
    y_train
)


print("\nHGB training completed.")


# ------------------------------------------------------------
# 10. Save trained model
# ------------------------------------------------------------

joblib.dump(

    hgb_model,

    HGB_MODEL_FILE

)


# ============================================================
# PART F
# VALIDATION THRESHOLD SELECTION
# ============================================================


# ------------------------------------------------------------
# 11. HGB validation scores
# ------------------------------------------------------------

validation_scores = (

    hgb_model
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


precision_values = (
    precision_curve[:-1]
)

recall_values = (
    recall_curve[:-1]
)


# ------------------------------------------------------------
# 12. Validation F2
# ------------------------------------------------------------

f2_values = (

    5
    * precision_values
    * recall_values

    /

    (
        4
        * precision_values
        + recall_values
        + 1e-12
    )

)


best_index = int(
    np.nanargmax(
        f2_values
    )
)


SELECTED_THRESHOLD = float(
    thresholds[
        best_index
    ]
)


VALIDATION_PRECISION = float(
    precision_values[
        best_index
    ]
)


VALIDATION_RECALL = float(
    recall_values[
        best_index
    ]
)


VALIDATION_F2 = float(
    f2_values[
        best_index
    ]
)


print("\nSelected HGB threshold:")

print(
    f"{SELECTED_THRESHOLD:.6f}"
)


print("\nValidation Precision:")

print(
    f"{VALIDATION_PRECISION:.4f}"
)


print("\nValidation Recall:")

print(
    f"{VALIDATION_RECALL:.4f}"
)


print("\nValidation F2:")

print(
    f"{VALIDATION_F2:.4f}"
)


# ------------------------------------------------------------
# 13. Save best threshold candidates
# ------------------------------------------------------------

threshold_df = pd.DataFrame({

    "Threshold":
        thresholds,

    "Precision":
        precision_values,

    "Recall":
        recall_values,

    "F2":
        f2_values

})


(
    threshold_df
    .sort_values(
        "F2",
        ascending=False
    )
    .head(50)
    .to_csv(

        OUTPUT_DIR
        / "step10_top_validation_thresholds.csv",

        index=False

    )
)


# ============================================================
# PART G
# TEST EVALUATION
# ============================================================


# ------------------------------------------------------------
# 14. Generic evaluation function
# ------------------------------------------------------------

def evaluate_model(
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


    alerts = int(
        predictions.sum()
    )


    return {

        "Model":
            model_name,

        "Threshold":
            threshold,

        "ROC_AUC":

            roc_auc_score(
                y_true,
                scores
            ),

        "Average_Precision":

            average_precision_score(
                y_true,
                scores
            ),

        "Accuracy":

            accuracy_score(
                y_true,
                predictions
            ),

        "Precision":
            precision,

        "Recall":
            recall,

        "F1":
            f1,

        "F2":
            f2,

        "True_Negative":
            int(tn),

        "False_Positive":
            int(fp),

        "False_Negative":
            int(fn),

        "True_Positive":
            int(tp),

        "Alerts":
            alerts,

        "Alert_Rate_Pct":

            alerts
            / len(y_true)
            * 100,

        "False_Positive_Rate_Pct":

            fp
            / (
                fp
                + tn
            )
            * 100

            if (
                fp
                + tn
            ) > 0

            else 0

    }


# ------------------------------------------------------------
# 15. HGB test scores
# ------------------------------------------------------------

print("\nEvaluating untouched test data...")


hgb_test_scores = (

    hgb_model
    .predict_proba(
        X_test
    )[:, 1]

)


hgb_metrics = evaluate_model(

    y_test,

    hgb_test_scores,

    SELECTED_THRESHOLD,

    "HistGradientBoosting - Defensible Features"

)


# ============================================================
# PART H
# REPRODUCE MODEL B ON EXACT SAME TEST DATA
# ============================================================


print(
    "\nLoading Step 9 Model B..."
)


model_b = joblib.load(
    MODEL_B_FILE
)


# ------------------------------------------------------------
# 16. Model B threshold from Step 9
# ------------------------------------------------------------

step9_comparison = pd.read_csv(
    STEP9_COMPARISON_FILE
)


model_b_row = (

    step9_comparison.loc[

        step9_comparison[
            "Model"
        ].eq(
            "Model_B_Log_Balances_Only"
        )

    ]
    .iloc[0]

)


MODEL_B_THRESHOLD = float(
    model_b_row[
        "Validation_Threshold"
    ]
)


# ------------------------------------------------------------
# 17. Score same test observations
# ------------------------------------------------------------

model_b_test_scores = (

    model_b
    .predict_proba(
        X_test
    )[:, 1]

)


model_b_metrics = evaluate_model(

    y_test,

    model_b_test_scores,

    MODEL_B_THRESHOLD,

    "Logistic Regression - Model B"

)


# ============================================================
# PART I
# ADD EXISTING RULE BENCHMARK
# ============================================================


step8_comparison = pd.read_csv(
    STEP8_COMPARISON_FILE
)


existing_rule_row = (

    step8_comparison.loc[

        step8_comparison[
            "Model"
        ].eq(
            "Existing isFlaggedFraud Rule"
        )

    ]
    .iloc[0]

)


existing_rule_metrics = {

    "Model":
        "Existing isFlaggedFraud Rule",

    "Threshold":
        np.nan,

    "ROC_AUC":
        np.nan,

    "Average_Precision":
        np.nan,

    "Accuracy":
        existing_rule_row[
            "Accuracy"
        ],

    "Precision":
        existing_rule_row[
            "Precision"
        ],

    "Recall":
        existing_rule_row[
            "Recall"
        ],

    "F1":
        existing_rule_row[
            "F1"
        ],

    "F2":
        existing_rule_row[
            "F2"
        ],

    "True_Negative":
        int(
            existing_rule_row[
                "True_Negative"
            ]
        ),

    "False_Positive":
        int(
            existing_rule_row[
                "False_Positive"
            ]
        ),

    "False_Negative":
        int(
            existing_rule_row[
                "False_Negative"
            ]
        ),

    "True_Positive":
        int(
            existing_rule_row[
                "True_Positive"
            ]
        ),

    "Alerts":
        int(
            existing_rule_row[
                "Alerts"
            ]
        ),

    "Alert_Rate_Pct":
        existing_rule_row[
            "Alert_Rate_Pct"
        ],

    "False_Positive_Rate_Pct":
        existing_rule_row[
            "False_Positive_Rate_Pct"
        ]

}


# ============================================================
# PART J
# MODEL COMPARISON
# ============================================================


comparison_df = pd.DataFrame([

    hgb_metrics,

    model_b_metrics,

    existing_rule_metrics

])


comparison_df.to_csv(

    OUTPUT_DIR
    / "step10_model_comparison.csv",

    index=False

)


# ------------------------------------------------------------
# 18. HGB vs Model B delta
# ------------------------------------------------------------

delta_df = pd.DataFrame({

    "Metric": [

        "ROC_AUC",

        "Average_Precision",

        "Precision",

        "Recall",

        "F1",

        "F2",

        "True_Positive",

        "False_Positive",

        "False_Negative",

        "Alerts",

        "Alert_Rate_Pct"

    ],

    "HGB_Value": [

        hgb_metrics["ROC_AUC"],

        hgb_metrics[
            "Average_Precision"
        ],

        hgb_metrics["Precision"],

        hgb_metrics["Recall"],

        hgb_metrics["F1"],

        hgb_metrics["F2"],

        hgb_metrics[
            "True_Positive"
        ],

        hgb_metrics[
            "False_Positive"
        ],

        hgb_metrics[
            "False_Negative"
        ],

        hgb_metrics["Alerts"],

        hgb_metrics[
            "Alert_Rate_Pct"
        ]

    ],

    "Model_B_Value": [

        model_b_metrics["ROC_AUC"],

        model_b_metrics[
            "Average_Precision"
        ],

        model_b_metrics["Precision"],

        model_b_metrics["Recall"],

        model_b_metrics["F1"],

        model_b_metrics["F2"],

        model_b_metrics[
            "True_Positive"
        ],

        model_b_metrics[
            "False_Positive"
        ],

        model_b_metrics[
            "False_Negative"
        ],

        model_b_metrics["Alerts"],

        model_b_metrics[
            "Alert_Rate_Pct"
        ]

    ]

})


delta_df[
    "HGB_Minus_Model_B"
] = (

    delta_df[
        "HGB_Value"
    ]

    -

    delta_df[
        "Model_B_Value"
    ]

)


delta_df.to_csv(

    OUTPUT_DIR
    / "step10_hgb_delta_vs_logistic.csv",

    index=False

)


# ============================================================
# PART K
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

    "hgb_fraud_score":

        hgb_test_scores,

    "hgb_predicted_fraud":

        (
            hgb_test_scores
            >= SELECTED_THRESHOLD
        ).astype("int8"),

    "logistic_model_b_score":

        model_b_test_scores,

    "logistic_model_b_predicted":

        (
            model_b_test_scores
            >= MODEL_B_THRESHOLD
        ).astype("int8")

})


test_predictions_df.to_parquet(

    TEST_PREDICTION_FILE,

    index=False

)


# ============================================================
# PART L
# PERMUTATION FEATURE IMPORTANCE
# ============================================================


print(
    "\nCalculating HGB permutation importance..."
)


# ------------------------------------------------------------
# 19. Use validation sample for diagnostic importance
# ------------------------------------------------------------

IMPORTANCE_SAMPLE_SIZE = min(
    100_000,
    len(validation_df)
)


importance_sample = (

    validation_df
    .sample(

        n=IMPORTANCE_SAMPLE_SIZE,

        random_state=RANDOM_STATE

    )

)


X_importance = (
    importance_sample[
        MODEL_FEATURES
    ]
)

y_importance = (
    importance_sample[
        "isFraud"
    ]
)


# ------------------------------------------------------------
# Ensure sample contains both classes
# ------------------------------------------------------------

if y_importance.nunique() == 2:

    importance_result = permutation_importance(

        hgb_model,

        X_importance,

        y_importance,

        scoring="average_precision",

        n_repeats=3,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )


    importance_df = pd.DataFrame({

        "Feature":
            MODEL_FEATURES,

        "Importance_Mean":

            importance_result[
                "importances_mean"
            ],

        "Importance_STD":

            importance_result[
                "importances_std"
            ]

    })


    importance_df = (

        importance_df
        .sort_values(

            "Importance_Mean",

            ascending=False

        )

    )

else:

    importance_df = pd.DataFrame({

        "Feature":
            MODEL_FEATURES,

        "Importance_Mean":
            np.nan,

        "Importance_STD":
            np.nan

    })


importance_df.to_csv(

    OUTPUT_DIR
    / "step10_hgb_permutation_importance.csv",

    index=False

)


# ============================================================
# PART M
# CHARTS
# ============================================================


# ------------------------------------------------------------
# 20. F2 comparison
# ------------------------------------------------------------

plot_df = comparison_df.loc[
    comparison_df["Model"]
    .ne(
        "Existing isFlaggedFraud Rule"
    )
].copy()


fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.bar(

    plot_df["Model"],

    plot_df["F2"]

)


ax.set_title(
    "Test F2: Logistic Regression vs HistGradientBoosting"
)

ax.set_ylabel(
    "F2 Score"
)


ax.tick_params(
    axis="x",
    rotation=20
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "01_f2_model_comparison.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 21. Precision / Recall comparison
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for _, row in plot_df.iterrows():

    ax.scatter(

        row["Recall"],

        row["Precision"],

        s=80

    )


    ax.annotate(

        row["Model"],

        (
            row["Recall"],
            row["Precision"]
        )

    )


ax.set_title(
    "Precision vs Recall: Defensible Models"
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
    / "02_precision_recall_comparison.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 22. Feature importance chart
# ------------------------------------------------------------

if importance_df[
    "Importance_Mean"
].notna().any():

    importance_plot_df = (

        importance_df
        .sort_values(
            "Importance_Mean"
        )

    )


    fig, ax = plt.subplots(
        figsize=(9, 6)
    )


    ax.barh(

        importance_plot_df[
            "Feature"
        ],

        importance_plot_df[
            "Importance_Mean"
        ]

    )


    ax.set_title(
        "HistGradientBoosting Permutation Importance"
    )

    ax.set_xlabel(
        "Decrease in Average Precision"
    )


    fig.tight_layout()


    fig.savefig(

        CHART_DIR
        / "03_hgb_permutation_importance.png",

        dpi=160

    )


    plt.close(fig)


# ------------------------------------------------------------
# 23. Alert volume
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.bar(

    plot_df["Model"],

    plot_df["Alerts"]

)


ax.set_title(
    "Investigation Alert Volume"
)

ax.set_ylabel(
    "Test Alerts"
)


ax.tick_params(
    axis="x",
    rotation=20
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "04_alert_volume_comparison.png",

    dpi=160

)


plt.close(fig)


# ============================================================
# PART N
# REPORT
# ============================================================


with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as report:


    report.write(
        "PROJECT 5 - STEP 10\n"
    )


    report.write(
        "HISTGRADIENTBOOSTING FRAUD MODEL\n"
    )


    report.write(
        "=" * 82
    )


    report.write(
        "\n\nTEMPORAL SPLIT\n\n"
    )


    report.write(

        split_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nFEATURE SET\n\n"
    )


    for feature in MODEL_FEATURES:

        report.write(
            f"- {feature}\n"
        )


    report.write(
        "\n\nHGB VALIDATION THRESHOLD\n\n"
    )


    report.write(

        f"Selected threshold: "
        f"{SELECTED_THRESHOLD:.8f}\n"

    )


    report.write(

        f"Validation precision: "
        f"{VALIDATION_PRECISION:.6f}\n"

    )


    report.write(

        f"Validation recall: "
        f"{VALIDATION_RECALL:.6f}\n"

    )


    report.write(

        f"Validation F2: "
        f"{VALIDATION_F2:.6f}\n"

    )


    report.write(
        "\n\nFINAL MODEL COMPARISON\n\n"
    )


    report.write(

        comparison_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nHGB VS LOGISTIC MODEL B\n\n"
    )


    report.write(

        delta_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nPERMUTATION IMPORTANCE\n\n"
    )


    report.write(

        importance_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nMODEL GOVERNANCE NOTES\n\n"
    )


    report.write(

        "1. HistGradientBoosting and Logistic "
        "Regression Model B use exactly the same "
        "seven defensible features.\n\n"

        "2. Both models use the same chronological "
        "Train / Validation / Test populations.\n\n"

        "3. HistGradientBoosting threshold selection "
        "uses validation F2 only. Test data remains "
        "untouched until final evaluation.\n\n"

        "4. The tree model deliberately excludes "
        "zero_amount_flag and "
        "amount_exceeds_origin_balance_flag because "
        "Step 9 showed these may exploit "
        "PaySim-specific synthetic behaviour.\n\n"

        "5. HistGradientBoosting does not require "
        "standardisation because tree splits are "
        "not scale-dependent.\n\n"

        "6. Permutation importance is calculated "
        "on a validation sample and should be "
        "interpreted as diagnostic model importance, "
        "not causal importance.\n"

    )


# ============================================================
# PART O
# DISPLAY RESULTS
# ============================================================


print("\n" + "=" * 82)

print(
    "STEP 10 FINAL MODEL COMPARISON"
)

print("=" * 82)


display_columns = [

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

    "Alerts",

    "Alert_Rate_Pct"
]


print(

    comparison_df[
        display_columns
    ].to_string(
        index=False
    )

)


print("\n" + "=" * 82)

print(
    "HGB FEATURE IMPORTANCE"
)

print("=" * 82)


print(

    importance_df.to_string(
        index=False
    )

)


print("\nSaved HGB model:")

print(
    HGB_MODEL_FILE
)


print("\nTest predictions:")

print(
    TEST_PREDICTION_FILE
)


print("\nReport:")

print(
    REPORT_FILE
)


print(
    "\nStep 10 completed successfully."
)