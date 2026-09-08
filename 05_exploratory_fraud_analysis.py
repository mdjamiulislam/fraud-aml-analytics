import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pyarrow.parquet as pq


# ============================================================
# PROJECT 5 - STEP 6
# EXPLORATORY FRAUD ANALYSIS
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

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"

CHART_DIR = (
    OUTPUT_DIR
    / "step6_charts"
)

DOCUMENTATION_DIR = (
    PROJECT_ROOT
    / "06_documentation"
)

CHART_DIR.mkdir(
    exist_ok=True
)

DOCUMENTATION_DIR.mkdir(
    exist_ok=True
)


REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_6_Exploratory_Fraud_Analysis_Report.txt"
)


# ------------------------------------------------------------
# 2. Processing settings
# ------------------------------------------------------------

BATCH_SIZE = 250_000


COLUMNS = [

    "type",
    "amount",
    "isFraud",

    "transaction_day",
    "hour_of_day",

    "oldbalanceOrg",
    "oldbalanceDest",

    "zero_amount_flag",
    "high_value_200k_flag",

    "origin_zero_before_flag",
    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "destination_is_merchant"
]


RISK_FLAGS = [

    "zero_amount_flag",

    "high_value_200k_flag",

    "origin_zero_before_flag",

    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "destination_is_merchant"
]


AMOUNT_BINS = [

    -0.001,
    0,
    10_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    5_000_000,
    np.inf
]


AMOUNT_LABELS = [

    "0",

    "0-10K",

    "10K-50K",

    "50K-100K",

    "100K-200K",

    "200K-500K",

    "500K-1M",

    "1M-5M",

    ">5M"
]


print("=" * 75)
print("PROJECT 5 - STEP 6")
print("EXPLORATORY FRAUD ANALYSIS")
print("=" * 75)

print("\nDataset:")
print(MASTER_FILE)

print("\nBatch size:")
print(f"{BATCH_SIZE:,}")


# ------------------------------------------------------------
# 3. Helper function
# ------------------------------------------------------------

def empty_stats():

    return {
        "count": 0,
        "value": 0.0,
        "fraud_count": 0,
        "fraud_value": 0.0
    }


# ------------------------------------------------------------
# 4. Initialise accumulators
# ------------------------------------------------------------

overall = {

    0: {
        "count": 0,
        "value": 0.0,
        "min_amount": np.inf,
        "max_amount": -np.inf,
        "origin_balance_sum": 0.0,
        "destination_balance_sum": 0.0
    },

    1: {
        "count": 0,
        "value": 0.0,
        "min_amount": np.inf,
        "max_amount": -np.inf,
        "origin_balance_sum": 0.0,
        "destination_balance_sum": 0.0
    }
}


type_stats = defaultdict(
    empty_stats
)

hour_stats = defaultdict(
    empty_stats
)

day_stats = defaultdict(
    empty_stats
)

amount_bucket_stats = defaultdict(
    empty_stats
)


flag_stats = {

    flag: {

        "true_count": 0,
        "true_fraud": 0,

        "false_count": 0,
        "false_fraud": 0

    }

    for flag in RISK_FLAGS
}


# ------------------------------------------------------------
# 5. Open Parquet file
# ------------------------------------------------------------

parquet_file = pq.ParquetFile(
    MASTER_FILE
)

total_parquet_rows = (
    parquet_file.metadata.num_rows
)


print("\nRows in master Parquet:")
print(f"{total_parquet_rows:,}")

print("\nBeginning EDA scan...\n")


# ------------------------------------------------------------
# 6. Process Parquet in batches
# ------------------------------------------------------------

rows_processed = 0


for batch_number, batch in enumerate(

    parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=COLUMNS
    ),

    start=1

):

    df = batch.to_pandas()

    rows_processed += len(df)


    print(
        f"Processing batch {batch_number} "
        f"| {len(df):,} rows "
        f"| cumulative {rows_processed:,}"
    )


    fraud_mask = (
        df["isFraud"].eq(1)
    )


    # --------------------------------------------------------
    # Overall fraud / legitimate statistics
    # --------------------------------------------------------

    for fraud_class in [0, 1]:

        class_mask = (
            df["isFraud"]
            .eq(fraud_class)
        )

        class_df = df.loc[
            class_mask
        ]

        if len(class_df) == 0:
            continue


        overall[
            fraud_class
        ]["count"] += len(
            class_df
        )


        overall[
            fraud_class
        ]["value"] += (
            class_df["amount"].sum()
        )


        overall[
            fraud_class
        ]["min_amount"] = min(

            overall[
                fraud_class
            ]["min_amount"],

            class_df[
                "amount"
            ].min()

        )


        overall[
            fraud_class
        ]["max_amount"] = max(

            overall[
                fraud_class
            ]["max_amount"],

            class_df[
                "amount"
            ].max()

        )


        overall[
            fraud_class
        ]["origin_balance_sum"] += (
            class_df[
                "oldbalanceOrg"
            ].sum()
        )


        overall[
            fraud_class
        ]["destination_balance_sum"] += (
            class_df[
                "oldbalanceDest"
            ].sum()
        )


    # --------------------------------------------------------
    # Transaction-type analysis
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

        stats["count"] += len(
            group
        )

        stats["value"] += (
            group["amount"].sum()
        )

        stats["fraud_count"] += len(
            fraud_group
        )

        stats["fraud_value"] += (
            fraud_group["amount"].sum()
        )


    # --------------------------------------------------------
    # Hourly fraud analysis
    # --------------------------------------------------------

    for hour, group in df.groupby(
        "hour_of_day",
        sort=False
    ):

        fraud_group = group.loc[
            group["isFraud"].eq(1)
        ]

        stats = hour_stats[
            int(hour)
        ]

        stats["count"] += len(
            group
        )

        stats["value"] += (
            group["amount"].sum()
        )

        stats["fraud_count"] += len(
            fraud_group
        )

        stats["fraud_value"] += (
            fraud_group["amount"].sum()
        )


    # --------------------------------------------------------
    # Daily fraud analysis
    # --------------------------------------------------------

    for day, group in df.groupby(
        "transaction_day",
        sort=False
    ):

        fraud_group = group.loc[
            group["isFraud"].eq(1)
        ]

        stats = day_stats[
            int(day)
        ]

        stats["count"] += len(
            group
        )

        stats["value"] += (
            group["amount"].sum()
        )

        stats["fraud_count"] += len(
            fraud_group
        )

        stats["fraud_value"] += (
            fraud_group["amount"].sum()
        )


    # --------------------------------------------------------
    # Amount bucket analysis
    # --------------------------------------------------------

    amount_bucket = pd.cut(

        df["amount"],

        bins=AMOUNT_BINS,

        labels=AMOUNT_LABELS,

        include_lowest=True,

        right=True
    )


    temporary_bucket_df = pd.DataFrame({

        "amount_bucket":
            amount_bucket,

        "amount":
            df["amount"],

        "isFraud":
            df["isFraud"]
    })


    for bucket_name, group in (

        temporary_bucket_df
        .groupby(
            "amount_bucket",
            observed=False
        )

    ):

        fraud_group = group.loc[
            group["isFraud"].eq(1)
        ]

        stats = amount_bucket_stats[
            str(bucket_name)
        ]

        stats["count"] += len(
            group
        )

        stats["value"] += (
            group["amount"].sum()
        )

        stats["fraud_count"] += len(
            fraud_group
        )

        stats["fraud_value"] += (
            fraud_group[
                "amount"
            ].sum()
        )


    # --------------------------------------------------------
    # Risk-flag analysis
    # --------------------------------------------------------

    for flag in RISK_FLAGS:

        flag_true = (
            df[flag].eq(1)
        )

        true_count = int(
            flag_true.sum()
        )

        true_fraud = int(
            (
                flag_true
                & fraud_mask
            ).sum()
        )


        false_count = (
            len(df)
            - true_count
        )


        false_fraud = int(
            (
                ~flag_true
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


print("\nFull EDA scan completed.")


# ------------------------------------------------------------
# 7. Create overall fraud summary
# ------------------------------------------------------------

overall_rows = []


for fraud_class, label in [

    (0, "Legitimate"),
    (1, "Fraud")

]:

    stats = overall[
        fraud_class
    ]


    average_amount = (

        stats["value"]
        / stats["count"]

        if stats["count"] > 0

        else np.nan
    )


    average_origin_balance = (

        stats[
            "origin_balance_sum"
        ]
        / stats["count"]

        if stats["count"] > 0

        else np.nan
    )


    average_destination_balance = (

        stats[
            "destination_balance_sum"
        ]
        / stats["count"]

        if stats["count"] > 0

        else np.nan
    )


    overall_rows.append({

        "Class":
            label,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Average_Amount":
            average_amount,

        "Minimum_Amount":
            stats["min_amount"],

        "Maximum_Amount":
            stats["max_amount"],

        "Average_Origin_Balance_Before":
            average_origin_balance,

        "Average_Destination_Balance_Before":
            average_destination_balance
    })


overall_df = pd.DataFrame(
    overall_rows
)


# ------------------------------------------------------------
# 8. Transaction type summary
# ------------------------------------------------------------

type_rows = []


for transaction_type in sorted(
    type_stats.keys()
):

    stats = type_stats[
        transaction_type
    ]


    fraud_rate = (

        stats["fraud_count"]
        / stats["count"]
        * 100

        if stats["count"] > 0

        else 0
    )


    fraud_value_share = (

        stats["fraud_value"]
        / stats["value"]
        * 100

        if stats["value"] > 0

        else 0
    )


    average_amount = (

        stats["value"]
        / stats["count"]

        if stats["count"] > 0

        else 0
    )


    average_fraud_amount = (

        stats["fraud_value"]
        / stats["fraud_count"]

        if stats["fraud_count"] > 0

        else 0
    )


    type_rows.append({

        "Transaction_Type":
            transaction_type,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Average_Transaction_Amount":
            average_amount,

        "Fraud_Count":
            stats["fraud_count"],

        "Fraud_Value":
            stats["fraud_value"],

        "Average_Fraud_Amount":
            average_fraud_amount,

        "Fraud_Rate_Pct":
            fraud_rate,

        "Fraud_Value_Share_Pct":
            fraud_value_share
    })


type_df = pd.DataFrame(
    type_rows
)


# ------------------------------------------------------------
# 9. Hourly summary
# ------------------------------------------------------------

hour_rows = []


for hour in sorted(
    hour_stats.keys()
):

    stats = hour_stats[
        hour
    ]


    hour_rows.append({

        "Hour_of_Day":
            hour,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Fraud_Count":
            stats["fraud_count"],

        "Fraud_Value":
            stats["fraud_value"],

        "Fraud_Rate_Pct":

            stats["fraud_count"]
            / stats["count"]
            * 100

            if stats["count"] > 0

            else 0
    })


hour_df = pd.DataFrame(
    hour_rows
)


# ------------------------------------------------------------
# 10. Daily summary
# ------------------------------------------------------------

day_rows = []


for day in sorted(
    day_stats.keys()
):

    stats = day_stats[
        day
    ]


    day_rows.append({

        "Transaction_Day":
            day,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Fraud_Count":
            stats["fraud_count"],

        "Fraud_Value":
            stats["fraud_value"],

        "Fraud_Rate_Pct":

            stats["fraud_count"]
            / stats["count"]
            * 100

            if stats["count"] > 0

            else 0
    })


day_df = pd.DataFrame(
    day_rows
)


# ------------------------------------------------------------
# 11. Amount bucket summary
# ------------------------------------------------------------

amount_bucket_rows = []


for bucket in AMOUNT_LABELS:

    stats = amount_bucket_stats[
        bucket
    ]


    amount_bucket_rows.append({

        "Amount_Bucket":
            bucket,

        "Transaction_Count":
            stats["count"],

        "Transaction_Value":
            stats["value"],

        "Fraud_Count":
            stats["fraud_count"],

        "Fraud_Value":
            stats["fraud_value"],

        "Fraud_Rate_Pct":

            stats["fraud_count"]
            / stats["count"]
            * 100

            if stats["count"] > 0

            else 0
    })


amount_bucket_df = pd.DataFrame(
    amount_bucket_rows
)


# ------------------------------------------------------------
# 12. Risk flag summary
# ------------------------------------------------------------

flag_rows = []


for flag in RISK_FLAGS:

    stats = flag_stats[
        flag
    ]


    fraud_rate_true = (

        stats["true_fraud"]
        / stats["true_count"]
        * 100

        if stats["true_count"] > 0

        else 0
    )


    fraud_rate_false = (

        stats["false_fraud"]
        / stats["false_count"]
        * 100

        if stats["false_count"] > 0

        else 0
    )


    if fraud_rate_false > 0:

        lift = (
            fraud_rate_true
            / fraud_rate_false
        )

    else:

        lift = np.nan


    flag_rows.append({

        "Risk_Flag":
            flag,

        "Flag_True_Count":
            stats["true_count"],

        "Fraud_When_True":
            stats["true_fraud"],

        "Fraud_Rate_When_True_Pct":
            fraud_rate_true,

        "Flag_False_Count":
            stats["false_count"],

        "Fraud_When_False":
            stats["false_fraud"],

        "Fraud_Rate_When_False_Pct":
            fraud_rate_false,

        "Fraud_Rate_Lift":
            lift
    })


flag_df = pd.DataFrame(
    flag_rows
)


# ------------------------------------------------------------
# 13. Save summary tables
# ------------------------------------------------------------

overall_df.to_csv(

    OUTPUT_DIR
    / "step6_overall_fraud_summary.csv",

    index=False
)


type_df.to_csv(

    OUTPUT_DIR
    / "step6_transaction_type_summary.csv",

    index=False
)


hour_df.to_csv(

    OUTPUT_DIR
    / "step6_hourly_fraud_summary.csv",

    index=False
)


day_df.to_csv(

    OUTPUT_DIR
    / "step6_daily_fraud_summary.csv",

    index=False
)


amount_bucket_df.to_csv(

    OUTPUT_DIR
    / "step6_amount_bucket_summary.csv",

    index=False
)


flag_df.to_csv(

    OUTPUT_DIR
    / "step6_risk_flag_summary.csv",

    index=False
)


# ------------------------------------------------------------
# 14. Chart 1
# Average transaction amount:
# Fraud vs legitimate
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(8, 5)
)


ax.bar(

    overall_df["Class"],

    overall_df[
        "Average_Amount"
    ]
)


ax.set_title(
    "Average Transaction Amount: Fraud vs Legitimate"
)

ax.set_ylabel(
    "Average Transaction Amount"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "01_average_amount_fraud_vs_legitimate.png",

    dpi=160
)


plt.close(fig)


# ------------------------------------------------------------
# 15. Chart 2
# Fraud rate by transaction type
# ------------------------------------------------------------

plot_type_df = (
    type_df
    .sort_values(
        "Fraud_Rate_Pct",
        ascending=False
    )
)


fig, ax = plt.subplots(
    figsize=(9, 5)
)


ax.bar(

    plot_type_df[
        "Transaction_Type"
    ],

    plot_type_df[
        "Fraud_Rate_Pct"
    ]
)


ax.set_title(
    "Fraud Rate by Transaction Type"
)

ax.set_ylabel(
    "Fraud Rate (%)"
)

ax.set_xlabel(
    "Transaction Type"
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "02_fraud_rate_by_transaction_type.png",

    dpi=160
)


plt.close(fig)


# ------------------------------------------------------------
# 16. Chart 3
# Fraud count by simulated hour
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 5)
)


ax.plot(

    hour_df[
        "Hour_of_Day"
    ],

    hour_df[
        "Fraud_Count"
    ],

    marker="o"
)


ax.set_title(
    "Fraud Transactions by Simulated Hour"
)

ax.set_xlabel(
    "Simulated Hour of Day"
)

ax.set_ylabel(
    "Fraud Transactions"
)

ax.set_xticks(
    range(0, 24)
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "03_fraud_count_by_hour.png",

    dpi=160
)


plt.close(fig)


# ------------------------------------------------------------
# 17. Chart 4
# Fraud rate by transaction amount bucket
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(11, 5)
)


ax.bar(

    amount_bucket_df[
        "Amount_Bucket"
    ],

    amount_bucket_df[
        "Fraud_Rate_Pct"
    ]
)


ax.set_title(
    "Fraud Rate by Transaction Amount Band"
)

ax.set_xlabel(
    "Transaction Amount Band"
)

ax.set_ylabel(
    "Fraud Rate (%)"
)


ax.tick_params(
    axis="x",
    rotation=45
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "04_fraud_rate_by_amount_bucket.png",

    dpi=160
)


plt.close(fig)


# ------------------------------------------------------------
# 18. Chart 5
# Fraud-rate lift of engineered flags
# ------------------------------------------------------------

lift_plot_df = flag_df.loc[

    flag_df[
        "Fraud_Rate_Lift"
    ].notna()

].copy()


lift_plot_df = lift_plot_df.sort_values(

    "Fraud_Rate_Lift",

    ascending=False
)


fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.bar(

    lift_plot_df[
        "Risk_Flag"
    ],

    lift_plot_df[
        "Fraud_Rate_Lift"
    ]
)


ax.set_title(
    "Fraud-Rate Lift of Engineered Risk Flags"
)

ax.set_ylabel(
    "Fraud Rate Lift"
)


ax.tick_params(
    axis="x",
    rotation=45
)


fig.tight_layout()


fig.savefig(

    CHART_DIR
    / "05_risk_flag_fraud_lift.png",

    dpi=160
)


plt.close(fig)


# ------------------------------------------------------------
# 19. Create readable report
# ------------------------------------------------------------

with open(

    REPORT_FILE,

    "w",

    encoding="utf-8"

) as report:


    report.write(
        "PROJECT 5 - STEP 6\n"
    )


    report.write(
        "EXPLORATORY FRAUD ANALYSIS\n"
    )


    report.write(
        "=" * 75
    )


    report.write(
        "\n\nOVERALL FRAUD SUMMARY\n\n"
    )


    report.write(

        overall_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nTRANSACTION TYPE ANALYSIS\n\n"
    )


    report.write(

        type_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nHOURLY FRAUD ANALYSIS\n\n"
    )


    report.write(

        hour_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nDAILY FRAUD ANALYSIS\n\n"
    )


    report.write(

        day_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nAMOUNT BAND ANALYSIS\n\n"
    )


    report.write(

        amount_bucket_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nRISK FLAG ANALYSIS\n\n"
    )


    report.write(

        flag_df.to_string(
            index=False
        )

    )


    report.write(
        "\n\nIMPORTANT INTERPRETATION NOTE\n\n"
    )


    report.write(

        "PaySim uses simulated time steps. "
        "The hour-of-day and day variables therefore "
        "represent simulation time rather than actual "
        "calendar timestamps or real customer behaviour.\n"

    )


# ------------------------------------------------------------
# 20. Display results
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("OVERALL FRAUD SUMMARY")
print("=" * 75)

print(
    overall_df.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("TRANSACTION TYPE FRAUD SUMMARY")
print("=" * 75)

print(
    type_df.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("AMOUNT BAND FRAUD SUMMARY")
print("=" * 75)

print(
    amount_bucket_df.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("RISK FLAG SUMMARY")
print("=" * 75)

print(
    flag_df.to_string(
        index=False
    )
)


print("\nRows processed:")
print(f"{rows_processed:,}")

print("\nCharts saved to:")
print(CHART_DIR)

print("\nReport:")
print(REPORT_FILE)

print("\nStep 6 completed successfully.")