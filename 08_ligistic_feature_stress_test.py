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
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# PROJECT 5 - STEP 9
# LOGISTIC REGRESSION FEATURE STRESS TEST
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

SPLIT_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step8_temporal_split_summary.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"

DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)

MODEL_DIR = (
    OUTPUT_DIR
    / "step9_models"
)

CHART_DIR = (
    OUTPUT_DIR
    / "step9_charts"
)

MODEL_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_9_Logistic_Feature_Stress_Test_Report.txt"
)


# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000

RANDOM_STATE = 42


print("=" * 82)
print("PROJECT 5 - STEP 9")
print("LOGISTIC REGRESSION FEATURE STRESS TEST")
print("=" * 82)


# ============================================================
# PART A
# READ AND VALIDATE STEP 8 SPLIT
# ============================================================


# ------------------------------------------------------------
# 3. Read Step 8 temporal split
# ------------------------------------------------------------

split_df = pd.read_csv(
    SPLIT_FILE
)


train_row = split_df.loc[
    split_df["Split"].eq("TRAIN")
].iloc[0]


validation_row = split_df.loc[
    split_df["Split"].eq("VALIDATION")
].iloc[0]


test_row = split_df.loc[
    split_df["Split"].eq("TEST")
].iloc[0]


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


print("\nUsing Step 8 split boundaries:")

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
# LOAD AND ENGINEER MODEL FEATURES
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

    "zero_amount_flag",

    "high_value_200k_flag",

    "origin_zero_before_flag",

    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "isFraud"
]


# ------------------------------------------------------------
# 5. Feature-engineering function
# ------------------------------------------------------------

def prepare_batch(df):

    output = pd.DataFrame()


    # Traceability / split control

    output["transaction_id"] = (
        df["transaction_id"]
        .astype("int64")
    )


    output["step"] = (
        df["step"]
        .astype("int16")
    )


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
    # Cyclical hour features
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
    # Binary features
    # --------------------------------------------------------

    binary_columns = [

        "zero_amount_flag",

        "high_value_200k_flag",

        "origin_zero_before_flag",

        "destination_zero_before_flag",

        "amount_exceeds_origin_balance_flag"
    ]


    for col in binary_columns:

        output[col] = (

            df[col]
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


# ------------------------------------------------------------
# 6. Read model population
# ------------------------------------------------------------

parquet_file = pq.ParquetFile(
    MODEL_FILE
)


train_parts = []
validation_parts = []
test_parts = []


rows_processed = 0


print("\nPreparing Step 9 model data...\n")


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
# 7. Combine split datasets
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
# 8. Verify row counts
# ------------------------------------------------------------

assert len(train_df) == EXPECTED_TRAIN_ROWS, (
    "TRAIN row count does not match Step 8."
)

assert len(validation_df) == EXPECTED_VALIDATION_ROWS, (
    "VALIDATION row count does not match Step 8."
)

assert len(test_df) == EXPECTED_TEST_ROWS, (
    "TEST row count does not match Step 8."
)


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
# PART C
# DEFINE FOUR FEATURE SETS
# ============================================================


# ------------------------------------------------------------
# 9. Common feature groups
# ------------------------------------------------------------

TIME_FEATURES = [

    "hour_sin",
    "hour_cos"
]


TRANSACTION_FEATURES = [

    "log_amount",
    "type_transfer_flag",
    "high_value_200k_flag"
]


LOG_BALANCE_FEATURES = [

    "log_oldbalanceOrg",
    "log_oldbalanceDest"
]


ZERO_BALANCE_FLAGS = [

    "origin_zero_before_flag",
    "destination_zero_before_flag"
]


SIMULATION_SPECIFIC_FEATURES = [

    "zero_amount_flag",
    "amount_exceeds_origin_balance_flag"
]


# ------------------------------------------------------------
# 10. Model feature definitions
# ------------------------------------------------------------

MODEL_CONFIGS = {


    # --------------------------------------------------------
    # Model A
    # Reproduce Step 8 baseline
    # --------------------------------------------------------

    "Model_A_Current_Baseline": {

        "continuous": [

            "log_amount",

            "log_oldbalanceOrg",
            "log_oldbalanceDest",

            "hour_sin",
            "hour_cos"

        ],

        "binary": [

            "type_transfer_flag",

            "high_value_200k_flag",

            "origin_zero_before_flag",
            "destination_zero_before_flag"

        ],

        "purpose":

            "Step 8 baseline with both "
            "log balances and zero-balance flags."
    },


    # --------------------------------------------------------
    # Model B
    # Keep log balances, remove zero-balance flags
    # --------------------------------------------------------

    "Model_B_Log_Balances_Only": {

        "continuous": [

            "log_amount",

            "log_oldbalanceOrg",
            "log_oldbalanceDest",

            "hour_sin",
            "hour_cos"

        ],

        "binary": [

            "type_transfer_flag",

            "high_value_200k_flag"

        ],

        "purpose":

            "Tests whether strong performance "
            "remains after removing redundant "
            "zero-balance indicators."
    },


    # --------------------------------------------------------
    # Model C
    # Keep zero-balance flags, remove log balances
    # --------------------------------------------------------

    "Model_C_Balance_Flags_Only": {

        "continuous": [

            "log_amount",

            "hour_sin",
            "hour_cos"

        ],

        "binary": [

            "type_transfer_flag",

            "high_value_200k_flag",

            "origin_zero_before_flag",
            "destination_zero_before_flag"

        ],

        "purpose":

            "Tests whether simple balance-state "
            "flags provide sufficient signal "
            "without continuous balance amounts."
    },


    # --------------------------------------------------------
    # Model D
    # Enhanced synthetic-data model
    # --------------------------------------------------------

    "Model_D_Enhanced_PaySim": {

        "continuous": [

            "log_amount",

            "log_oldbalanceOrg",
            "log_oldbalanceDest",

            "hour_sin",
            "hour_cos"

        ],

        "binary": [

            "type_transfer_flag",

            "high_value_200k_flag",

            "origin_zero_before_flag",
            "destination_zero_before_flag",

            "zero_amount_flag",

            "amount_exceeds_origin_balance_flag"

        ],

        "purpose":

            "Stress-test model including "
            "highly predictive PaySim-specific "
            "transaction-balance behaviour."
    }

}


# ============================================================
# PART D
# MODEL HELPER FUNCTIONS
# ============================================================


# ------------------------------------------------------------
# 11. Find F2-maximising validation threshold
# ------------------------------------------------------------

def select_f2_threshold(
    y_true,
    scores
):

    precision_curve, recall_curve, thresholds = (

        precision_recall_curve(
            y_true,
            scores
        )

    )


    precision_values = (
        precision_curve[:-1]
    )

    recall_values = (
        recall_curve[:-1]
    )


    beta_squared = 4


    f2_values = (

        5
        * precision_values
        * recall_values

        /

        (
            beta_squared
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


    return {

        "threshold":

            float(
                thresholds[
                    best_index
                ]
            ),

        "precision":

            float(
                precision_values[
                    best_index
                ]
            ),

        "recall":

            float(
                recall_values[
                    best_index
                ]
            ),

        "f2":

            float(
                f2_values[
                    best_index
                ]
            )

    }


# ------------------------------------------------------------
# 12. Calculate test metrics
# ------------------------------------------------------------

def evaluate_scores(
    y_true,
    scores,
    threshold
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


# ============================================================
# PART E
# TRAIN AND TEST ALL FOUR MODELS
# ============================================================


results = []

coefficient_outputs = []

feature_set_rows = []


for model_name, config in (
    MODEL_CONFIGS.items()
):


    print("\n" + "=" * 82)

    print(
        f"TRAINING {model_name}"
    )

    print("=" * 82)


    continuous_features = (
        config["continuous"]
    )


    binary_features = (
        config["binary"]
    )


    model_features = (

        continuous_features
        + binary_features

    )


    print("\nFeatures:")

    for feature in model_features:

        print(
            f" - {feature}"
        )


    # --------------------------------------------------------
    # Record feature set
    # --------------------------------------------------------

    for feature in model_features:

        feature_set_rows.append({

            "Model":
                model_name,

            "Feature":
                feature,

            "Feature_Type":

                "Continuous"

                if feature
                in continuous_features

                else "Binary",

            "Purpose":
                config["purpose"]

        })


    # --------------------------------------------------------
    # X / y data
    # --------------------------------------------------------

    X_train = train_df[
        model_features
    ]

    y_train = train_df[
        "isFraud"
    ]


    X_validation = validation_df[
        model_features
    ]

    y_validation = validation_df[
        "isFraud"
    ]


    X_test = test_df[
        model_features
    ]

    y_test = test_df[
        "isFraud"
    ]


    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "continuous",

                StandardScaler(),

                continuous_features

            ),

            (

                "binary",

                "passthrough",

                binary_features

            )

        ],

        remainder="drop"

    )


    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    classifier = LogisticRegression(

        class_weight="balanced",

        solver="lbfgs",

        max_iter=700,

        random_state=RANDOM_STATE

    )


    pipeline = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )

    ])


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Validation scores
    # --------------------------------------------------------

    validation_scores = (

        pipeline
        .predict_proba(
            X_validation
        )[:, 1]

    )


    threshold_result = (

        select_f2_threshold(

            y_validation,

            validation_scores

        )

    )


    selected_threshold = (
        threshold_result[
            "threshold"
        ]
    )


    # --------------------------------------------------------
    # Test scores
    # --------------------------------------------------------

    test_scores = (

        pipeline
        .predict_proba(
            X_test
        )[:, 1]

    )


    test_metrics = (

        evaluate_scores(

            y_test,

            test_scores,

            selected_threshold

        )

    )


    # --------------------------------------------------------
    # Store model result
    # --------------------------------------------------------

    result_row = {

        "Model":
            model_name,

        "Feature_Count":
            len(
                model_features
            ),

        "Validation_Threshold":
            selected_threshold,

        "Validation_Precision":
            threshold_result[
                "precision"
            ],

        "Validation_Recall":
            threshold_result[
                "recall"
            ],

        "Validation_F2":
            threshold_result[
                "f2"
            ],

        **test_metrics

    }


    results.append(
        result_row
    )


    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    fitted_preprocessor = (

        pipeline
        .named_steps[
            "preprocessor"
        ]

    )


    fitted_classifier = (

        pipeline
        .named_steps[
            "classifier"
        ]

    )


    feature_names = (

        fitted_preprocessor
        .get_feature_names_out()

    )


    coefficients = (

        fitted_classifier
        .coef_[0]

    )


    for feature_name, coefficient in zip(

        feature_names,

        coefficients

    ):

        coefficient_outputs.append({

            "Model":
                model_name,

            "Feature":
                feature_name,

            "Coefficient":
                coefficient,

            "Odds_Ratio":

                np.exp(

                    np.clip(
                        coefficient,
                        -20,
                        20
                    )

                ),

            "Absolute_Coefficient":

                abs(
                    coefficient
                )

        })


    # --------------------------------------------------------
    # Save each model
    # --------------------------------------------------------

    safe_model_name = (

        model_name
        .lower()
        .replace(" ", "_")

    )


    joblib.dump(

        pipeline,

        MODEL_DIR
        / f"{safe_model_name}.joblib"

    )


    print("\nSelected threshold:")

    print(
        f"{selected_threshold:.6f}"
    )


    print("\nTest Precision:")

    print(
        f"{test_metrics['Precision']:.4f}"
    )


    print("\nTest Recall:")

    print(
        f"{test_metrics['Recall']:.4f}"
    )


    print("\nTest F2:")

    print(
        f"{test_metrics['F2']:.4f}"
    )


    print("\nTest Average Precision:")

    print(
        f"{test_metrics['Average_Precision']:.4f}"
    )


# ============================================================
# PART F
# BUILD COMPARISON TABLES
# ============================================================


# ------------------------------------------------------------
# 13. Main stress-test comparison
# ------------------------------------------------------------

comparison_df = pd.DataFrame(
    results
)


comparison_df = (

    comparison_df
    .sort_values(

        "F2",

        ascending=False

    )

)


comparison_df.to_csv(

    OUTPUT_DIR
    / "step9_model_stress_test_comparison.csv",

    index=False

)


# ------------------------------------------------------------
# 14. Coefficient table
# ------------------------------------------------------------

coefficient_df = pd.DataFrame(
    coefficient_outputs
)


coefficient_df.to_csv(

    OUTPUT_DIR
    / "step9_all_model_coefficients.csv",

    index=False

)


# ------------------------------------------------------------
# 15. Feature-set register
# ------------------------------------------------------------

feature_sets_df = pd.DataFrame(
    feature_set_rows
)


feature_sets_df.to_csv(

    DOCUMENTATION_DIR
    / "Step_9_Model_Feature_Sets.csv",

    index=False

)


# ============================================================
# PART G
# COEFFICIENT STABILITY TABLE
# ============================================================


# ------------------------------------------------------------
# 16. Pivot coefficients for comparison
# ------------------------------------------------------------

coefficient_pivot = (

    coefficient_df
    .pivot_table(

        index="Feature",

        columns="Model",

        values="Coefficient",

        aggfunc="first"

    )
    .reset_index()

)


coefficient_pivot.to_csv(

    OUTPUT_DIR
    / "step9_coefficient_stability.csv",

    index=False

)


# ============================================================
# PART H
# MODEL DELTA TABLE
# ============================================================


# ------------------------------------------------------------
# 17. Compare all models with Model A
# ------------------------------------------------------------

baseline_row = (

    comparison_df.loc[

        comparison_df[
            "Model"
        ].eq(
            "Model_A_Current_Baseline"
        )

    ]
    .iloc[0]

)


delta_rows = []


for _, row in (
    comparison_df.iterrows()
):

    delta_rows.append({

        "Model":
            row["Model"],

        "ROC_AUC_Delta_vs_A":

            row["ROC_AUC"]
            - baseline_row[
                "ROC_AUC"
            ],

        "Average_Precision_Delta_vs_A":

            row[
                "Average_Precision"
            ]
            - baseline_row[
                "Average_Precision"
            ],

        "Precision_Delta_vs_A":

            row["Precision"]
            - baseline_row[
                "Precision"
            ],

        "Recall_Delta_vs_A":

            row["Recall"]
            - baseline_row[
                "Recall"
            ],

        "F2_Delta_vs_A":

            row["F2"]
            - baseline_row[
                "F2"
            ],

        "Alert_Rate_Delta_vs_A":

            row[
                "Alert_Rate_Pct"
            ]
            - baseline_row[
                "Alert_Rate_Pct"
            ],

        "True_Positive_Delta_vs_A":

            int(
                row[
                    "True_Positive"
                ]
                - baseline_row[
                    "True_Positive"
                ]
            ),

        "False_Positive_Delta_vs_A":

            int(
                row[
                    "False_Positive"
                ]
                - baseline_row[
                    "False_Positive"
                ]
            )
    })


delta_df = pd.DataFrame(
    delta_rows
)


delta_df.to_csv(

    OUTPUT_DIR
    / "step9_model_delta_vs_baseline.csv",

    index=False

)


# ============================================================
# PART I
# CHARTS
# ============================================================


# ------------------------------------------------------------
# 18. F2 comparison
# ------------------------------------------------------------

plot_df = (

    comparison_df
    .sort_values(
        "F2"
    )

)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.barh(

    plot_df["Model"],

    plot_df["F2"]

)


ax.set_title(
    "Logistic Regression Feature Stress Test - Test F2"
)

ax.set_xlabel(
    "Test F2 Score"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "01_test_f2_model_comparison.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 19. Average Precision comparison
# ------------------------------------------------------------

plot_df = (

    comparison_df
    .sort_values(
        "Average_Precision"
    )

)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.barh(

    plot_df["Model"],

    plot_df[
        "Average_Precision"
    ]

)


ax.set_title(
    "Logistic Regression Feature Stress Test - Average Precision"
)

ax.set_xlabel(
    "Test Average Precision"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "02_average_precision_model_comparison.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 20. Precision vs Recall
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 6)
)


for _, row in (
    comparison_df.iterrows()
):

    ax.scatter(

        row["Recall"],

        row["Precision"],

        s=80

    )


    ax.annotate(

        row["Model"]
        .replace(
            "Model_",
            ""
        ),

        (
            row["Recall"],
            row["Precision"]
        )

    )


ax.set_title(
    "Precision vs Recall - Feature Stress Test"
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
    / "03_precision_recall_model_comparison.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 21. Alert volume comparison
# ------------------------------------------------------------

plot_df = (

    comparison_df
    .sort_values(
        "Alerts"
    )

)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.barh(

    plot_df["Model"],

    plot_df["Alerts"]

)


ax.set_title(
    "Investigation Alert Volume by Model"
)

ax.set_xlabel(
    "Test Alerts"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "04_alert_volume_model_comparison.png",

    dpi=160

)


plt.close(fig)


# ============================================================
# PART J
# REPORT
# ============================================================


with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as report:


    report.write(
        "PROJECT 5 - STEP 9\n"
    )


    report.write(
        "LOGISTIC REGRESSION FEATURE STRESS TEST\n"
    )


    report.write(
        "=" * 82
    )


    report.write(
        "\n\nSTEP 8 SPLIT USED\n\n"
    )


    report.write(

        split_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nMODEL COMPARISON\n\n"
    )


    report.write(

        comparison_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nDELTA VS MODEL A\n\n"
    )


    report.write(

        delta_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nCOEFFICIENT STABILITY\n\n"
    )


    report.write(

        coefficient_pivot.to_string(
            index=False
        )

    )


    report.write(
        "\n\nINTERPRETATION FRAMEWORK\n\n"
    )


    report.write(

        "Model A reproduces the Step 8 baseline.\n\n"

        "Model B removes zero-balance indicators "
        "while retaining continuous log balances. "
        "If performance remains close to Model A, "
        "the simpler specification is preferred "
        "for coefficient stability.\n\n"

        "Model C removes continuous balance amounts "
        "and retains only simple zero-balance flags. "
        "This measures how much discrimination comes "
        "from balance-state indicators alone.\n\n"

        "Model D deliberately introduces the "
        "zero-amount flag and "
        "amount-exceeds-origin-balance flag. "
        "These variables showed extremely strong "
        "PaySim-specific relationships in prior EDA. "
        "A large performance jump should therefore "
        "be interpreted as evidence of synthetic-data "
        "mechanism exploitation, not automatically "
        "as superior real-world generalisability.\n\n"

        "All models use exactly the same chronological "
        "Train / Validation / Test periods.\n\n"

        "Each classification threshold is selected "
        "independently on validation data using F2 "
        "and is then applied once to the untouched "
        "test period.\n"

    )


# ============================================================
# PART K
# DISPLAY FINAL RESULTS
# ============================================================


print("\n" + "=" * 82)
print("STEP 9 MODEL STRESS TEST COMPARISON")
print("=" * 82)


display_columns = [

    "Model",

    "Feature_Count",

    "Validation_Threshold",

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
print("DELTA VS MODEL A")
print("=" * 82)


print(

    delta_df.to_string(
        index=False
    )

)


print("\nReport:")
print(REPORT_FILE)


print(
    "\nStep 9 completed successfully."
)