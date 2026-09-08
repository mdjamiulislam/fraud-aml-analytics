import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq


# ============================================================
# PROJECT 5 - STEP 7
# HIGH-RISK POPULATION ANALYSIS & MODEL PREPARATION
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

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"

DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)

OUTPUT_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_7_Model_Preparation_Report.txt"
)


# ------------------------------------------------------------
# 2. Processing settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000


# Chronological split boundaries

TRAIN_END_STEP = 520

VALIDATION_END_STEP = 594


# ------------------------------------------------------------
# 3. Columns to analyse
# ------------------------------------------------------------

COLUMNS = [

    "transaction_id",

    "step",
    "transaction_day",
    "hour_of_day",

    "type",

    "amount",
    "log_amount",

    "oldbalanceOrg",
    "oldbalanceDest",

    "zero_amount_flag",
    "high_value_200k_flag",

    "origin_zero_before_flag",
    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "destination_is_merchant",

    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",

    "isFraud"
]


BINARY_FLAGS = [

    "zero_amount_flag",

    "high_value_200k_flag",

    "origin_zero_before_flag",

    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "destination_is_merchant"
]


NUMERIC_FEATURES = [

    "transaction_day",
    "hour_of_day",

    "amount",
    "log_amount",

    "oldbalanceOrg",
    "oldbalanceDest",

    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio"
]


# ------------------------------------------------------------
# 4. Initialisation
# ------------------------------------------------------------

print("=" * 78)

print(
    "PROJECT 5 - STEP 7: "
    "HIGH-RISK MODEL PREPARATION"
)

print("=" * 78)

print("\nModel population file:")
print(MODEL_FILE)

print("\nBatch size:")
print(f"{BATCH_SIZE:,}")


total_rows = 0
total_fraud = 0


# Missing-value counters

missing_counts = {
    col: 0
    for col in COLUMNS
}


# Infinite-value counters

infinite_counts = {
    col: 0
    for col in NUMERIC_FEATURES
}


# Type analysis

type_stats = defaultdict(
    lambda: {
        "count": 0,
        "fraud": 0,
        "value": 0.0,
        "fraud_value": 0.0
    }
)


# Binary flag analysis

flag_stats = {

    flag: {

        "true_count": 0,
        "true_fraud": 0,

        "false_count": 0,
        "false_fraud": 0

    }

    for flag in BINARY_FLAGS
}


# Numeric statistics by fraud class

numeric_stats = {

    feature: {

        0: {
            "count": 0,
            "sum": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "missing": 0
        },

        1: {
            "count": 0,
            "sum": 0.0,
            "min": np.inf,
            "max": -np.inf,
            "missing": 0
        }

    }

    for feature in NUMERIC_FEATURES
}


# Value distributions for low-cardinality fields

category_counts = {

    feature: defaultdict(int)

    for feature in (
        ["type"]
        + BINARY_FLAGS
    )
}


# Chronological split statistics

split_stats = {

    "TRAIN": {
        "rows": 0,
        "fraud": 0
    },

    "VALIDATION": {
        "rows": 0,
        "fraud": 0
    },

    "TEST": {
        "rows": 0,
        "fraud": 0
    }
}


# ------------------------------------------------------------
# 5. Open Parquet file
# ------------------------------------------------------------

parquet_file = pq.ParquetFile(
    MODEL_FILE
)

parquet_rows = (
    parquet_file.metadata.num_rows
)


print("\nRows recorded in Parquet:")
print(f"{parquet_rows:,}")

print("\nBeginning Step 7 scan...\n")


# ------------------------------------------------------------
# 6. Process batches
# ------------------------------------------------------------

for batch_number, batch in enumerate(

    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=COLUMNS
    ),

    start=1

):

    df = batch.to_pandas()

    rows_in_batch = len(df)

    total_rows += rows_in_batch

    total_fraud += int(
        df["isFraud"].sum()
    )


    print(

        f"Processing batch {batch_number} "
        f"| {rows_in_batch:,} rows "
        f"| cumulative {total_rows:,}"

    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    batch_missing = (
        df.isna().sum()
    )

    for col in COLUMNS:

        missing_counts[col] += int(
            batch_missing[col]
        )


    # --------------------------------------------------------
    # Infinite-value checks
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        series = df[feature]

        non_null = (
            series.dropna()
        )

        if len(non_null) > 0:

            infinite_counts[
                feature
            ] += int(

                np.isinf(
                    non_null.to_numpy(
                        dtype="float64"
                    )
                ).sum()

            )


    # --------------------------------------------------------
    # Transaction type analysis
    # --------------------------------------------------------

    for transaction_type, group in df.groupby(

        "type",
        sort=False

    ):

        fraud_group = group.loc[
            group["isFraud"].eq(1)
        ]

        stats = type_stats[
            transaction_type
        ]

        stats["count"] += len(group)

        stats["fraud"] += len(
            fraud_group
        )

        stats["value"] += (
            group["amount"].sum()
        )

        stats["fraud_value"] += (
            fraud_group["amount"].sum()
        )


    # --------------------------------------------------------
    # Binary flag analysis
    # --------------------------------------------------------

    fraud_mask = (
        df["isFraud"].eq(1)
    )


    for flag in BINARY_FLAGS:

        true_mask = (
            df[flag].eq(1)
        )


        true_count = int(
            true_mask.sum()
        )


        true_fraud = int(

            (
                true_mask
                & fraud_mask
            ).sum()

        )


        false_count = (
            rows_in_batch
            - true_count
        )


        false_fraud = int(

            (
                ~true_mask
                & fraud_mask
            ).sum()

        )


        flag_stats[
            flag
        ]["true_count"] += (
            true_count
        )


        flag_stats[
            flag
        ]["true_fraud"] += (
            true_fraud
        )


        flag_stats[
            flag
        ]["false_count"] += (
            false_count
        )


        flag_stats[
            flag
        ]["false_fraud"] += (
            false_fraud
        )


    # --------------------------------------------------------
    # Low-cardinality value distributions
    # --------------------------------------------------------

    for feature in (
        ["type"]
        + BINARY_FLAGS
    ):

        value_counts = (
            df[feature]
            .value_counts(
                dropna=False
            )
        )

        for value, count in (
            value_counts.items()
        ):

            category_counts[
                feature
            ][str(value)] += int(
                count
            )


    # --------------------------------------------------------
    # Numeric statistics by fraud class
    # --------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        for fraud_class in [0, 1]:

            class_series = df.loc[

                df["isFraud"]
                .eq(fraud_class),

                feature

            ]


            missing = int(
                class_series
                .isna()
                .sum()
            )


            valid_series = (
                class_series
                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )
                .dropna()
            )


            stats = (
                numeric_stats[
                    feature
                ][fraud_class]
            )


            stats[
                "missing"
            ] += missing


            if len(valid_series) == 0:
                continue


            stats[
                "count"
            ] += len(
                valid_series
            )


            stats[
                "sum"
            ] += (
                valid_series.sum()
            )


            stats[
                "min"
            ] = min(

                stats["min"],

                valid_series.min()

            )


            stats[
                "max"
            ] = max(

                stats["max"],

                valid_series.max()

            )


    # --------------------------------------------------------
    # Chronological train / validation / test split
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


    for split_name, split_mask in [

        ("TRAIN", train_mask),

        (
            "VALIDATION",
            validation_mask
        ),

        ("TEST", test_mask)

    ]:

        split_stats[
            split_name
        ]["rows"] += int(
            split_mask.sum()
        )


        split_stats[
            split_name
        ]["fraud"] += int(

            df.loc[
                split_mask,
                "isFraud"
            ].sum()

        )


print("\nFull Step 7 scan completed.")


# ------------------------------------------------------------
# 7. Population summary
# ------------------------------------------------------------

population_df = pd.DataFrame({

    "Metric": [

        "Parquet rows",

        "Rows processed",

        "Total fraud",

        "Legitimate transactions",

        "Fraud rate (%)",

        "Minimum step",

        "Maximum step"
    ],

    "Value": [

        parquet_rows,

        total_rows,

        total_fraud,

        total_rows
        - total_fraud,

        total_fraud
        / total_rows
        * 100,

        1,

        743
    ]
})


# ------------------------------------------------------------
# 8. Missing-value summary
# ------------------------------------------------------------

missing_rows = []


for col in COLUMNS:

    missing_rows.append({

        "Feature":
            col,

        "Missing_Count":
            missing_counts[col],

        "Missing_Pct":

            missing_counts[col]
            / total_rows
            * 100

    })


missing_df = pd.DataFrame(
    missing_rows
)


# ------------------------------------------------------------
# 9. Infinite-value summary
# ------------------------------------------------------------

infinite_df = pd.DataFrame({

    "Feature":
        list(
            infinite_counts.keys()
        ),

    "Infinite_Count":
        list(
            infinite_counts.values()
        )
})


# ------------------------------------------------------------
# 10. Transaction type summary
# ------------------------------------------------------------

type_rows = []


for transaction_type in sorted(
    type_stats.keys()
):

    stats = type_stats[
        transaction_type
    ]


    type_rows.append({

        "Transaction_Type":
            transaction_type,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Fraud_Count":
            stats["fraud"],

        "Fraud_Value":
            stats["fraud_value"],

        "Fraud_Rate_Pct":

            stats["fraud"]
            / stats["count"]
            * 100

    })


type_df = pd.DataFrame(
    type_rows
)


# ------------------------------------------------------------
# 11. Binary flag summary
# ------------------------------------------------------------

flag_rows = []


for flag in BINARY_FLAGS:

    stats = flag_stats[
        flag
    ]


    true_rate = (

        stats["true_fraud"]
        / stats["true_count"]
        * 100

        if stats["true_count"] > 0

        else 0
    )


    false_rate = (

        stats["false_fraud"]
        / stats["false_count"]
        * 100

        if stats["false_count"] > 0

        else 0
    )


    lift = (

        true_rate
        / false_rate

        if false_rate > 0

        else np.nan
    )


    flag_rows.append({

        "Feature":
            flag,

        "Flag_True_Count":
            stats["true_count"],

        "Fraud_When_True":
            stats["true_fraud"],

        "Fraud_Rate_True_Pct":
            true_rate,

        "Flag_False_Count":
            stats["false_count"],

        "Fraud_When_False":
            stats["false_fraud"],

        "Fraud_Rate_False_Pct":
            false_rate,

        "Fraud_Rate_Lift":
            lift
    })


flag_df = pd.DataFrame(
    flag_rows
)


# ------------------------------------------------------------
# 12. Numeric class comparison
# ------------------------------------------------------------

numeric_rows = []


for feature in NUMERIC_FEATURES:

    for fraud_class, class_name in [

        (0, "Legitimate"),

        (1, "Fraud")

    ]:

        stats = (
            numeric_stats[
                feature
            ][fraud_class]
        )


        mean_value = (

            stats["sum"]
            / stats["count"]

            if stats["count"] > 0

            else np.nan
        )


        minimum = (

            stats["min"]

            if stats["count"] > 0

            else np.nan
        )


        maximum = (

            stats["max"]

            if stats["count"] > 0

            else np.nan
        )


        numeric_rows.append({

            "Feature":
                feature,

            "Class":
                class_name,

            "Valid_Count":
                stats["count"],

            "Missing_Count":
                stats["missing"],

            "Mean":
                mean_value,

            "Minimum":
                minimum,

            "Maximum":
                maximum
        })


numeric_df = pd.DataFrame(
    numeric_rows
)


# ------------------------------------------------------------
# 13. Temporal split summary
# ------------------------------------------------------------

split_rows = []


for split_name in [

    "TRAIN",
    "VALIDATION",
    "TEST"

]:

    stats = split_stats[
        split_name
    ]


    if split_name == "TRAIN":

        step_range = (
            "1-520"
        )


    elif split_name == "VALIDATION":

        step_range = (
            "521-594"
        )


    else:

        step_range = (
            "595-743"
        )


    fraud_rate = (

        stats["fraud"]
        / stats["rows"]
        * 100

        if stats["rows"] > 0

        else 0
    )


    split_rows.append({

        "Split":
            split_name,

        "Step_Range":
            step_range,

        "Rows":
            stats["rows"],

        "Population_Pct":

            stats["rows"]
            / total_rows
            * 100,

        "Fraud_Count":
            stats["fraud"],

        "Fraud_Rate_Pct":
            fraud_rate
    })


split_df = pd.DataFrame(
    split_rows
)


# ------------------------------------------------------------
# 14. Feature variation summary
# ------------------------------------------------------------

variation_rows = []


for feature in (

    ["type"]
    + BINARY_FLAGS

):

    counts = category_counts[
        feature
    ]


    variation_rows.append({

        "Feature":
            feature,

        "Unique_Values":
            len(counts),

        "Value_Distribution":
            "; ".join(

                f"{key}: {value:,}"

                for key, value
                in sorted(
                    counts.items()
                )

            )
    })


variation_df = pd.DataFrame(
    variation_rows
)


# ------------------------------------------------------------
# 15. Baseline feature decision register
# ------------------------------------------------------------

feature_decision_df = pd.DataFrame({

    "Feature": [

        "transaction_id",

        "step",

        "transaction_day",

        "hour_of_day",

        "type",

        "amount",

        "log_amount",

        "oldbalanceOrg",

        "oldbalanceDest",

        "zero_amount_flag",

        "high_value_200k_flag",

        "origin_zero_before_flag",

        "destination_zero_before_flag",

        "amount_exceeds_origin_balance_flag",

        "destination_is_merchant",

        "amount_to_origin_balance_ratio",

        "amount_to_destination_balance_ratio",

        "isFraud"
    ],


    "Baseline_Logistic_Decision": [

        "EXCLUDE",

        "SPLIT CONTROL ONLY",

        "KEEP",

        "KEEP",

        "KEEP",

        "EXCLUDE FROM BASELINE",

        "KEEP",

        "TRANSFORM LOG1P",

        "TRANSFORM LOG1P",

        "KEEP",

        "KEEP",

        "KEEP",

        "KEEP",

        "KEEP",

        "DROP IF REDUNDANT",

        "HOLD OUT FROM BASELINE",

        "HOLD OUT FROM BASELINE",

        "TARGET"
    ],


    "Reason": [

        "Identifier only; no predictive meaning",

        "Used to enforce chronological split; overlaps with transaction_day",

        "Captures simulation-period effects",

        "Captures simulation-hour effects",

        "TRANSFER versus CASH_OUT risk difference",

        "Highly skewed; log_amount is preferred for interpretable baseline",

        "Reduces amount skew while preserving transaction-value signal",

        "Pre-transaction feature; log-transform before scaling",

        "Pre-transaction feature; log-transform before scaling",

        "Retained because all 16 zero-amount cases are fraud, but extremely rare",

        "Strong interpretable high-value risk indicator",

        "Observed inverse association may carry useful signal",

        "Observed positive fraud-rate lift",

        "Observed strong inverse association; evaluate coefficient carefully",

        "Expected to have little/no variation in CASH_OUT + TRANSFER population",

        "High missingness/skew possible; evaluate after baseline",

        "High missingness/skew possible; evaluate after baseline",

        "Fraud prediction target only"
    ]
})


# ------------------------------------------------------------
# 16. Save outputs
# ------------------------------------------------------------

population_df.to_csv(

    OUTPUT_DIR
    / "step7_high_risk_population_summary.csv",

    index=False
)


missing_df.to_csv(

    OUTPUT_DIR
    / "step7_feature_missingness.csv",

    index=False
)


infinite_df.to_csv(

    OUTPUT_DIR
    / "step7_infinite_value_check.csv",

    index=False
)


type_df.to_csv(

    OUTPUT_DIR
    / "step7_high_risk_type_summary.csv",

    index=False
)


flag_df.to_csv(

    OUTPUT_DIR
    / "step7_binary_flag_summary.csv",

    index=False
)


numeric_df.to_csv(

    OUTPUT_DIR
    / "step7_numeric_class_summary.csv",

    index=False
)


split_df.to_csv(

    OUTPUT_DIR
    / "step7_temporal_split_summary.csv",

    index=False
)


variation_df.to_csv(

    OUTPUT_DIR
    / "step7_feature_variation.csv",

    index=False
)


feature_decision_df.to_csv(

    DOCUMENTATION_DIR
    / "Step_7_Model_Feature_Decision_Register.csv",

    index=False
)


# ------------------------------------------------------------
# 17. Create readable report
# ------------------------------------------------------------

with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as report:


    report.write(
        "PROJECT 5 - STEP 7\n"
    )


    report.write(
        "HIGH-RISK POPULATION ANALYSIS "
        "AND MODEL PREPARATION\n"
    )


    report.write(
        "=" * 78
    )


    report.write(
        "\n\nPOPULATION SUMMARY\n\n"
    )


    report.write(
        population_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nHIGH-RISK TRANSACTION TYPES\n\n"
    )


    report.write(
        type_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nFEATURE MISSINGNESS\n\n"
    )


    report.write(
        missing_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nBINARY FLAG ANALYSIS\n\n"
    )


    report.write(
        flag_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nNUMERIC CLASS COMPARISON\n\n"
    )


    report.write(
        numeric_df.to_string(
            index=False
        )
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
        "\n\nFEATURE VARIATION\n\n"
    )


    report.write(
        variation_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nBASELINE FEATURE DECISIONS\n\n"
    )


    report.write(
        feature_decision_df.to_string(
            index=False
        )
    )


    report.write(
        "\n\nMODEL GOVERNANCE NOTES\n\n"
    )


    report.write(

        "1. The final test period must remain "
        "untouched during model development.\n"

        "2. Any imputation, scaling, encoding, "
        "or class balancing must be fitted only "
        "on training data.\n"

        "3. Post-transaction balances and "
        "isFlaggedFraud are excluded from the "
        "primary model to reduce leakage risk.\n"

        "4. Accuracy is not an appropriate primary "
        "metric because fraud is highly imbalanced.\n"

        "5. Threshold selection must use validation "
        "data, not the final test data.\n"

    )


# ------------------------------------------------------------
# 18. Display important outputs
# ------------------------------------------------------------

print("\n" + "=" * 78)
print("HIGH-RISK POPULATION SUMMARY")
print("=" * 78)

print(
    population_df.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("TRANSACTION TYPE SUMMARY")
print("=" * 78)

print(
    type_df.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("FEATURE MISSINGNESS")
print("=" * 78)

print(
    missing_df.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("BINARY FLAG SUMMARY")
print("=" * 78)

print(
    flag_df.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("=" * 78)

print(
    split_df.to_string(
        index=False
    )
)


print("\n" + "=" * 78)
print("FEATURE VARIATION")
print("=" * 78)

print(
    variation_df.to_string(
        index=False
    )
)


print("\nReport:")
print(REPORT_FILE)

print("\nStep 7 completed successfully.")
