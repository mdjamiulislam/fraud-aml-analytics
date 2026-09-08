import pandas as pd
import numpy as np
from pathlib import Path

import pyarrow.parquet as pq
import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix
)


# ============================================================
# PROJECT 5 - STEP 11
# FRAUD RISK TIERS & INVESTIGATOR ALERT QUEUE
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


MASTER_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "master_analytical.parquet"
)


SPLIT_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step8_temporal_split_summary.csv"
)


STEP10_COMPARISON_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step10_model_comparison.csv"
)


HGB_MODEL_FILE = (
    PROJECT_ROOT
    / "05_outputs"
    / "step10_model"
    / "histgradientboosting_defensible_features.joblib"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_outputs"
)


DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)


CHART_DIR = (
    OUTPUT_DIR
    / "step11_charts"
)


CHART_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_11_Fraud_Risk_Tiers_and_Alert_Queue_Report.txt"
)


SCORED_TEST_FILE = (
    OUTPUT_DIR
    / "step11_scored_high_risk_test.parquet"
)


INVESTIGATOR_QUEUE_FILE = (
    OUTPUT_DIR
    / "step11_investigator_alert_queue.csv"
)


QUEUE_EVALUATION_FILE = (
    OUTPUT_DIR
    / "step11_alert_queue_evaluation.csv"
)


TOP_100_FILE = (
    OUTPUT_DIR
    / "step11_top_100_alerts.csv"
)


# ------------------------------------------------------------
# 2. Processing settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000


HIGH_RISK_TYPES = [
    "CASH_OUT",
    "TRANSFER"
]


MODEL_FEATURES = [

    "type_transfer_flag",

    "log_amount",

    "log_oldbalanceOrg",
    "log_oldbalanceDest",

    "hour_sin",
    "hour_cos",

    "high_value_200k_flag"
]


print("=" * 84)

print(
    "PROJECT 5 - STEP 11"
)

print(
    "FRAUD RISK TIERS & INVESTIGATOR ALERT QUEUE"
)

print("=" * 84)


# ============================================================
# PART A
# READ OFFICIAL SPLIT & OPERATING THRESHOLD
# ============================================================


# ------------------------------------------------------------
# 3. Read Step 8 temporal split
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


EXPECTED_VALIDATION_ROWS = int(
    validation_row["Rows"]
)


EXPECTED_TEST_ROWS = int(
    test_row["Rows"]
)


print("\nOfficial temporal split:")

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


# ------------------------------------------------------------
# 4. Read champion-model operating threshold
# ------------------------------------------------------------

step10_comparison_df = pd.read_csv(
    STEP10_COMPARISON_FILE
)


hgb_row = (

    step10_comparison_df.loc[

        step10_comparison_df[
            "Model"
        ].str.contains(
            "HistGradientBoosting",
            na=False
        )

    ]
    .iloc[0]

)


OPERATIONAL_THRESHOLD = float(
    hgb_row["Threshold"]
)


print(
    "\nChampion HGB operational threshold:"
)

print(
    f"{OPERATIONAL_THRESHOLD:.8f}"
)


# ------------------------------------------------------------
# 5. Load champion model
# ------------------------------------------------------------

hgb_model = joblib.load(
    HGB_MODEL_FILE
)


# ============================================================
# PART B
# FEATURE ENGINEERING FUNCTION
# ============================================================


# ------------------------------------------------------------
# 6. Columns required from master data
# ------------------------------------------------------------

LOAD_COLUMNS = [

    "transaction_id",

    "step",
    "transaction_day",
    "hour_of_day",

    "type",

    "amount",
    "log_amount",

    "nameOrig",
    "nameDest",

    "oldbalanceOrg",
    "oldbalanceDest",

    "high_value_200k_flag",

    "isFraud",
    "isFlaggedFraud"
]


# ------------------------------------------------------------
# 7. Prepare seven model features
# ------------------------------------------------------------

def prepare_model_features(df):

    output = pd.DataFrame(
        index=df.index
    )


    # TRANSFER = 1
    # CASH_OUT = 0

    output[
        "type_transfer_flag"
    ] = (

        df["type"]
        .eq("TRANSFER")
        .astype("int8")

    )


    output[
        "log_amount"
    ] = (

        df["log_amount"]
        .astype("float32")

    )


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


    hour = (
        df["hour_of_day"]
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


    output[
        "high_value_200k_flag"
    ] = (

        df[
            "high_value_200k_flag"
        ]
        .astype("int8")

    )


    return output[
        MODEL_FEATURES
    ]


# ============================================================
# PART C
# SCORE VALIDATION PERIOD
# AND CREATE RISK-TIER BOUNDARIES
# ============================================================


parquet_file = pq.ParquetFile(
    MASTER_FILE
)


validation_score_parts = []


validation_rows_processed = 0


print(
    "\nScoring validation population "
    "for risk-tier boundaries..."
)


for batch in parquet_file.iter_batches(

    batch_size=BATCH_SIZE,

    columns=LOAD_COLUMNS

):

    df = batch.to_pandas()


    validation_mask = (

        df["type"]
        .isin(
            HIGH_RISK_TYPES
        )

        &

        df["step"]
        .gt(
            TRAIN_END_STEP
        )

        &

        df["step"]
        .le(
            VALIDATION_END_STEP
        )

    )


    validation_df = df.loc[
        validation_mask
    ].copy()


    if len(validation_df) == 0:

        continue


    X_validation = prepare_model_features(
        validation_df
    )


    validation_scores = (

        hgb_model
        .predict_proba(
            X_validation
        )[:, 1]

    )


    validation_score_parts.append(
        validation_scores
    )


    validation_rows_processed += len(
        validation_df
    )


validation_scores = np.concatenate(
    validation_score_parts
)


del validation_score_parts


# ------------------------------------------------------------
# 8. Validate row count
# ------------------------------------------------------------

assert (
    validation_rows_processed
    == EXPECTED_VALIDATION_ROWS
), (
    "Validation row count does not match Step 8."
)


print(
    "\nValidation rows scored:"
)

print(
    f"{validation_rows_processed:,}"
)


# ------------------------------------------------------------
# 9. Risk-tier percentile thresholds
# ------------------------------------------------------------

MEDIUM_THRESHOLD = float(
    np.quantile(
        validation_scores,
        0.95
    )
)


HIGH_THRESHOLD = float(
    np.quantile(
        validation_scores,
        0.99
    )
)


CRITICAL_THRESHOLD = float(
    np.quantile(
        validation_scores,
        0.995
    )
)


# ------------------------------------------------------------
# 10. Tier threshold table
# ------------------------------------------------------------

threshold_df = pd.DataFrame({

    "Threshold_Type": [

        "Medium Risk Threshold",

        "High Risk Threshold",

        "Critical Risk Threshold",

        "Operational Investigation Threshold"
    ],

    "Source": [

        "Validation 95th percentile",

        "Validation 99th percentile",

        "Validation 99.5th percentile",

        "Step 10 Validation F2 optimisation"
    ],

    "Score_Threshold": [

        MEDIUM_THRESHOLD,

        HIGH_THRESHOLD,

        CRITICAL_THRESHOLD,

        OPERATIONAL_THRESHOLD
    ]
})


threshold_df.to_csv(

    OUTPUT_DIR
    / "step11_risk_tier_thresholds.csv",

    index=False
)


print("\nRisk tier thresholds:")

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# PART D
# SCORE OUT-OF-TIME TEST POPULATION
# ============================================================


print(
    "\nScoring out-of-time test population..."
)


test_parts = []


test_rows_processed = 0


# Re-open Parquet iterator

parquet_file = pq.ParquetFile(
    MASTER_FILE
)


for batch_number, batch in enumerate(

    parquet_file.iter_batches(

        batch_size=BATCH_SIZE,

        columns=LOAD_COLUMNS

    ),

    start=1

):

    df = batch.to_pandas()


    test_mask = (

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


    test_df = df.loc[
        test_mask
    ].copy()


    if len(test_df) == 0:

        continue


    X_test = prepare_model_features(
        test_df
    )


    scores = (

        hgb_model
        .predict_proba(
            X_test
        )[:, 1]

    )


    test_df[
        "hgb_fraud_score"
    ] = scores


    # --------------------------------------------------------
    # Relative risk tiers
    # --------------------------------------------------------

    test_df[
        "risk_tier"
    ] = np.select(

        [

            test_df[
                "hgb_fraud_score"
            ].ge(
                CRITICAL_THRESHOLD
            ),

            test_df[
                "hgb_fraud_score"
            ].ge(
                HIGH_THRESHOLD
            ),

            test_df[
                "hgb_fraud_score"
            ].ge(
                MEDIUM_THRESHOLD
            )

        ],

        [

            "Critical",

            "High",

            "Medium"

        ],

        default="Low"

    )


    # --------------------------------------------------------
    # Operational alert
    # --------------------------------------------------------

    test_df[
        "operational_alert"
    ] = (

        test_df[
            "hgb_fraud_score"
        ]
        .ge(
            OPERATIONAL_THRESHOLD
        )
        .astype("int8")

    )


    test_parts.append(
        test_df
    )


    test_rows_processed += len(
        test_df
    )


    print(

        f"Processed master batch "
        f"{batch_number} "
        f"| test rows cumulative "
        f"{test_rows_processed:,}"

    )


# ------------------------------------------------------------
# 11. Combine test population
# ------------------------------------------------------------

scored_test_df = pd.concat(

    test_parts,

    ignore_index=True

)


del test_parts


# ------------------------------------------------------------
# 12. Validate test row count
# ------------------------------------------------------------

assert (
    len(scored_test_df)
    == EXPECTED_TEST_ROWS
), (
    "Test row count does not match Step 8."
)


print(
    "\nTest rows scored:"
)

print(
    f"{len(scored_test_df):,}"
)


# ============================================================
# PART E
# RISK-TIER SUMMARY
# ============================================================


# ------------------------------------------------------------
# 13. Overall fraud totals
# ------------------------------------------------------------

TOTAL_TEST_FRAUD = int(

    scored_test_df[
        "isFraud"
    ].sum()

)


TOTAL_TEST_FRAUD_VALUE = float(

    scored_test_df.loc[

        scored_test_df[
            "isFraud"
        ].eq(1),

        "amount"

    ].sum()

)


# ------------------------------------------------------------
# 14. Risk-tier summary
# ------------------------------------------------------------

tier_order = [

    "Critical",
    "High",
    "Medium",
    "Low"
]


tier_rows = []


for tier in tier_order:


    tier_df = scored_test_df.loc[

        scored_test_df[
            "risk_tier"
        ].eq(tier)

    ]


    transaction_count = len(
        tier_df
    )


    fraud_count = int(

        tier_df[
            "isFraud"
        ].sum()

    )


    transaction_value = float(

        tier_df[
            "amount"
        ].sum()

    )


    fraud_value = float(

        tier_df.loc[

            tier_df[
                "isFraud"
            ].eq(1),

            "amount"

        ].sum()

    )


    operational_alert_count = int(

        tier_df[
            "operational_alert"
        ].sum()

    )


    tier_rows.append({

        "Risk_Tier":
            tier,

        "Transaction_Count":
            transaction_count,

        "Population_Pct":

            transaction_count
            / len(scored_test_df)
            * 100,

        "Average_Model_Score":

            tier_df[
                "hgb_fraud_score"
            ].mean(),

        "Transaction_Value":
            transaction_value,

        "Fraud_Count":
            fraud_count,

        "Fraud_Rate_Pct":

            fraud_count
            / transaction_count
            * 100

            if transaction_count > 0
            else 0,

        "Fraud_Capture_Pct":

            fraud_count
            / TOTAL_TEST_FRAUD
            * 100

            if TOTAL_TEST_FRAUD > 0
            else 0,

        "Fraud_Value":
            fraud_value,

        "Fraud_Value_Capture_Pct":

            fraud_value
            / TOTAL_TEST_FRAUD_VALUE
            * 100

            if TOTAL_TEST_FRAUD_VALUE > 0
            else 0,

        "Operational_Alerts":
            operational_alert_count

    })


tier_summary_df = pd.DataFrame(
    tier_rows
)


tier_summary_df.to_csv(

    OUTPUT_DIR
    / "step11_risk_tier_summary.csv",

    index=False
)


# ============================================================
# PART F
# OPERATIONAL ALERT PERFORMANCE
# ============================================================


# ------------------------------------------------------------
# 15. Champion-model alert confusion matrix
# ------------------------------------------------------------

y_true = (

    scored_test_df[
        "isFraud"
    ].to_numpy()

)


y_pred = (

    scored_test_df[
        "operational_alert"
    ].to_numpy()

)


tn, fp, fn, tp = (

    confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]

    ).ravel()

)


precision = precision_score(

    y_true,
    y_pred,
    zero_division=0

)


recall = recall_score(

    y_true,
    y_pred,
    zero_division=0

)


f1 = f1_score(

    y_true,
    y_pred,
    zero_division=0

)


f2 = fbeta_score(

    y_true,
    y_pred,

    beta=2,

    zero_division=0

)


alerts = int(
    y_pred.sum()
)


# ------------------------------------------------------------
# 16. Existing-rule performance on same test rows
# ------------------------------------------------------------

legacy_pred = (

    scored_test_df[
        "isFlaggedFraud"
    ]
    .astype("int8")
    .to_numpy()

)


legacy_tn, legacy_fp, legacy_fn, legacy_tp = (

    confusion_matrix(

        y_true,

        legacy_pred,

        labels=[0, 1]

    ).ravel()

)


legacy_precision = precision_score(

    y_true,
    legacy_pred,
    zero_division=0

)


legacy_recall = recall_score(

    y_true,
    legacy_pred,
    zero_division=0

)


# ------------------------------------------------------------
# 17. Operational comparison summary
# ------------------------------------------------------------

alert_summary_df = pd.DataFrame({

    "Metric": [

        "Test transactions",

        "Actual fraud transactions",

        "Champion model alerts",

        "Champion model alert rate (%)",

        "Champion model true positives",

        "Champion model false positives",

        "Champion model false negatives",

        "Champion model precision (%)",

        "Champion model recall (%)",

        "Champion model F1",

        "Champion model F2",

        "Legacy rule alerts",

        "Legacy rule true positives",

        "Legacy rule false positives",

        "Legacy rule false negatives",

        "Legacy rule precision (%)",

        "Legacy rule recall (%)",

        "Additional fraud detected vs legacy rule",

        "Detection multiple vs legacy rule"
    ],

    "Value": [

        len(scored_test_df),

        TOTAL_TEST_FRAUD,

        alerts,

        alerts
        / len(scored_test_df)
        * 100,

        int(tp),

        int(fp),

        int(fn),

        precision
        * 100,

        recall
        * 100,

        f1,

        f2,

        int(
            legacy_pred.sum()
        ),

        int(
            legacy_tp
        ),

        int(
            legacy_fp
        ),

        int(
            legacy_fn
        ),

        legacy_precision
        * 100,

        legacy_recall
        * 100,

        int(
            tp
            - legacy_tp
        ),

        (
            tp
            / legacy_tp

            if legacy_tp > 0

            else np.nan
        )

    ]
})


alert_summary_df.to_csv(

    OUTPUT_DIR
    / "step11_operational_alert_summary.csv",

    index=False
)


# ============================================================
# PART G
# CREATE INVESTIGATOR ALERT QUEUE
# ============================================================


# ------------------------------------------------------------
# 18. Filter operational alerts
# ------------------------------------------------------------

alert_evaluation_df = (

    scored_test_df.loc[

        scored_test_df[
            "operational_alert"
        ].eq(1)

    ]
    .copy()

)


# ------------------------------------------------------------
# 19. Sort by model score, then amount
# ------------------------------------------------------------

alert_evaluation_df = (

    alert_evaluation_df
    .sort_values(

        [

            "hgb_fraud_score",

            "amount"

        ],

        ascending=[

            False,

            False

        ]

    )
    .reset_index(
        drop=True
    )

)


# ------------------------------------------------------------
# 20. Add investigation rank
# ------------------------------------------------------------

alert_evaluation_df.insert(

    0,

    "investigation_rank",

    np.arange(

        1,

        len(alert_evaluation_df)
        + 1

    )

)


# ------------------------------------------------------------
# 21. Human-readable transaction type
# ------------------------------------------------------------

alert_evaluation_df[
    "priority_band"
] = (

    alert_evaluation_df[
        "risk_tier"
    ]

)


# ------------------------------------------------------------
# 22. Evaluation queue
#
# Includes actual fraud label.
# This is for project evaluation only.
# ------------------------------------------------------------

evaluation_columns = [

    "investigation_rank",

    "transaction_id",

    "step",
    "transaction_day",
    "hour_of_day",

    "type",

    "amount",

    "nameOrig",
    "nameDest",

    "oldbalanceOrg",
    "oldbalanceDest",

    "hgb_fraud_score",

    "risk_tier",
    "priority_band",

    "high_value_200k_flag",

    "isFlaggedFraud",

    "isFraud"
]


alert_evaluation_df[
    evaluation_columns
].to_csv(

    QUEUE_EVALUATION_FILE,

    index=False
)


# ------------------------------------------------------------
# 23. Investigator-facing queue
#
# Excludes actual isFraud because that would not
# be known during real investigation.
# ------------------------------------------------------------

investigator_columns = [

    "investigation_rank",

    "transaction_id",

    "step",
    "transaction_day",
    "hour_of_day",

    "type",

    "amount",

    "nameOrig",
    "nameDest",

    "oldbalanceOrg",
    "oldbalanceDest",

    "hgb_fraud_score",

    "risk_tier",
    "priority_band",

    "high_value_200k_flag",

    "isFlaggedFraud"
]


investigator_queue_df = (

    alert_evaluation_df[
        investigator_columns
    ]
    .copy()

)


investigator_queue_df.to_csv(

    INVESTIGATOR_QUEUE_FILE,

    index=False
)


# ------------------------------------------------------------
# 24. Top 100 alerts
# ------------------------------------------------------------

investigator_queue_df.head(
    100
).to_csv(

    TOP_100_FILE,

    index=False
)


# ============================================================
# PART H
# INVESTIGATION CAPACITY ANALYSIS
# ============================================================


# ------------------------------------------------------------
# 25. Capacity levels
# ------------------------------------------------------------

capacity_levels = [

    100,
    250,
    500,
    1_000,
    2_000,
    len(
        alert_evaluation_df
    )
]


capacity_levels = sorted(
    set(
        capacity_levels
    )
)


capacity_rows = []


for capacity in capacity_levels:


    reviewed_df = (

        alert_evaluation_df
        .head(
            capacity
        )

    )


    fraud_found = int(

        reviewed_df[
            "isFraud"
        ].sum()

    )


    precision_at_capacity = (

        fraud_found
        / len(reviewed_df)

        if len(reviewed_df) > 0

        else 0

    )


    fraud_capture = (

        fraud_found
        / TOTAL_TEST_FRAUD

        if TOTAL_TEST_FRAUD > 0

        else 0

    )


    capacity_rows.append({

        "Investigation_Capacity":
            capacity,

        "Alerts_Reviewed":
            len(
                reviewed_df
            ),

        "Fraud_Found":
            fraud_found,

        "Precision_Pct":

            precision_at_capacity
            * 100,

        "Total_Test_Fraud_Captured_Pct":

            fraud_capture
            * 100

    })


capacity_df = pd.DataFrame(
    capacity_rows
)


capacity_df.to_csv(

    OUTPUT_DIR
    / "step11_investigation_capacity_summary.csv",

    index=False
)


# ============================================================
# PART I
# SAVE FULL SCORED TEST POPULATION
# ============================================================


# ------------------------------------------------------------
# 26. Save model scoring output
# ------------------------------------------------------------

scored_output_columns = [

    "transaction_id",

    "step",
    "transaction_day",
    "hour_of_day",

    "type",

    "amount",

    "nameOrig",
    "nameDest",

    "oldbalanceOrg",
    "oldbalanceDest",

    "high_value_200k_flag",

    "hgb_fraud_score",

    "risk_tier",

    "operational_alert",

    "isFlaggedFraud",

    "isFraud"
]


scored_test_df[
    scored_output_columns
].to_parquet(

    SCORED_TEST_FILE,

    index=False

)


# ============================================================
# PART J
# CHARTS
# ============================================================


# ------------------------------------------------------------
# 27. Fraud rate by risk tier
# ------------------------------------------------------------

chart_df = (

    tier_summary_df
    .set_index(
        "Risk_Tier"
    )
    .loc[
        tier_order
    ]
    .reset_index()

)


fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(

    chart_df[
        "Risk_Tier"
    ],

    chart_df[
        "Fraud_Rate_Pct"
    ]

)


ax.set_title(
    "Observed Fraud Rate by Model Risk Tier"
)

ax.set_xlabel(
    "Risk Tier"
)

ax.set_ylabel(
    "Fraud Rate (%)"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "01_fraud_rate_by_risk_tier.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 28. Fraud capture by risk tier
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(

    chart_df[
        "Risk_Tier"
    ],

    chart_df[
        "Fraud_Capture_Pct"
    ]

)


ax.set_title(
    "Fraud Capture by Model Risk Tier"
)

ax.set_xlabel(
    "Risk Tier"
)

ax.set_ylabel(
    "Share of Test Fraud (%)"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "02_fraud_capture_by_risk_tier.png",

    dpi=160

)


plt.close(fig)


# ------------------------------------------------------------
# 29. Investigation capacity curve
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.plot(

    capacity_df[
        "Alerts_Reviewed"
    ],

    capacity_df[
        "Total_Test_Fraud_Captured_Pct"
    ],

    marker="o"

)


ax.set_title(
    "Fraud Capture vs Investigation Capacity"
)

ax.set_xlabel(
    "Alerts Reviewed"
)

ax.set_ylabel(
    "Total Test Fraud Captured (%)"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "03_fraud_capture_vs_investigation_capacity.png",

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
        "PROJECT 5 - STEP 11\n"
    )


    report.write(
        "FRAUD RISK TIERS & INVESTIGATOR ALERT QUEUE\n"
    )


    report.write(
        "=" * 84
    )


    report.write(
        "\n\nRISK TIER METHODOLOGY\n\n"
    )


    report.write(

        "Risk tiers are based on model-score "
        "percentiles from the validation period.\n\n"

        "Critical = top 0.5% of validation scores.\n"

        "High = 99th to 99.5th percentile.\n"

        "Medium = 95th to 99th percentile.\n"

        "Low = below the 95th percentile.\n\n"

        "These are relative model-risk tiers, "
        "not calibrated probabilities.\n\n"

        "The operational investigation threshold "
        "remains the validation-F2 threshold "
        "selected in Step 10.\n"

    )


    report.write(
        "\n\nRISK TIER THRESHOLDS\n\n"
    )


    report.write(

        threshold_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nRISK TIER SUMMARY\n\n"
    )


    report.write(

        tier_summary_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nOPERATIONAL ALERT PERFORMANCE\n\n"
    )


    report.write(

        alert_summary_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nINVESTIGATION CAPACITY ANALYSIS\n\n"
    )


    report.write(

        capacity_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nMODEL GOVERNANCE NOTES\n\n"
    )


    report.write(

        "1. Risk-tier thresholds were developed "
        "using validation scores only.\n\n"

        "2. The final test period was not used to "
        "define tier thresholds or the operational "
        "alert threshold.\n\n"

        "3. HGB model scores are ranking scores, "
        "not calibrated real-world fraud "
        "probabilities.\n\n"

        "4. The investigator-facing queue excludes "
        "the actual fraud label because that value "
        "would not be known in live operations.\n\n"

        "5. The evaluation queue retains isFraud "
        "only to measure model performance in this "
        "synthetic portfolio project.\n\n"

        "6. Risk tiers help prioritisation; the "
        "separate operational threshold determines "
        "which transactions are formally alerted.\n"

    )


# ============================================================
# PART L
# DISPLAY FINAL RESULTS
# ============================================================


print("\n" + "=" * 84)

print(
    "RISK TIER SUMMARY"
)

print("=" * 84)


print(

    tier_summary_df[
        [

            "Risk_Tier",

            "Transaction_Count",

            "Population_Pct",

            "Fraud_Count",

            "Fraud_Rate_Pct",

            "Fraud_Capture_Pct",

            "Operational_Alerts"

        ]
    ].to_string(
        index=False
    )

)


print("\n" + "=" * 84)

print(
    "OPERATIONAL ALERT SUMMARY"
)

print("=" * 84)


print(

    alert_summary_df.to_string(
        index=False
    )

)


print("\n" + "=" * 84)

print(
    "INVESTIGATION CAPACITY SUMMARY"
)

print("=" * 84)


print(

    capacity_df.to_string(
        index=False
    )

)


print("\nInvestigator queue:")

print(
    INVESTIGATOR_QUEUE_FILE
)


print("\nScored test dataset:")

print(
    SCORED_TEST_FILE
)


print("\nReport:")

print(
    REPORT_FILE
)


print(
    "\nStep 11 completed successfully."
)