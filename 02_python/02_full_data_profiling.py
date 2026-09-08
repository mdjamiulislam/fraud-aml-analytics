import pandas as pd
import numpy as np
from pathlib import Path
import shutil


# ============================================================
# PROJECT 5 - STEP 3
# FULL DATA PROFILING AUDIT
# ============================================================


# ------------------------------------------------------------
# 1. Define project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"
DOCUMENTATION_DIR = PROJECT_ROOT / "06_documentation"

TEMP_DIR = OUTPUT_DIR / "_step3_temp"

OUTPUT_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)

# Remove any old temporary profiling files
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

TEMP_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Profiling settings
# ------------------------------------------------------------

CHUNK_SIZE = 500_000

print("=" * 70)
print("PROJECT 5 - STEP 3: FULL DATA PROFILING AUDIT")
print("=" * 70)

print("\nDataset:")
print(DATA_FILE)

print("\nChunk size:")
print(f"{CHUNK_SIZE:,} rows")


# ------------------------------------------------------------
# 3. Initialise profiling variables
# ------------------------------------------------------------

total_rows = 0
chunk_number = 0

columns_seen = None
data_types = None

missing_counts = None

estimated_memory_mb = 0.0


# Transaction-type summaries
type_counts = pd.Series(dtype="float64")
type_amount_sum = pd.Series(dtype="float64")

fraud_by_type = pd.Series(dtype="float64")
fraud_amount_by_type = pd.Series(dtype="float64")


# Fraud summaries
total_fraud = 0
total_flagged = 0

fraud_amount_total = 0.0
legitimate_amount_total = 0.0


# Existing flagging system performance
true_positive = 0
false_positive = 0
true_negative = 0
false_negative = 0


# Numeric columns
numeric_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

numeric_min = {col: np.inf for col in numeric_columns}
numeric_max = {col: -np.inf for col in numeric_columns}
numeric_sum = {col: 0.0 for col in numeric_columns}
numeric_count = {col: 0 for col in numeric_columns}

zero_counts = {col: 0 for col in numeric_columns}
negative_counts = {col: 0 for col in numeric_columns}


# Temporary hash files
row_hash_files = []
origin_hash_files = []
destination_hash_files = []


# ------------------------------------------------------------
# 4. Read full dataset in chunks
# ------------------------------------------------------------

print("\nBeginning full dataset scan...\n")

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    print(
        f"Processing chunk {chunk_number} "
        f"| Rows in chunk: {len(chunk):,}"
    )

    # --------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------

    if columns_seen is None:
        columns_seen = chunk.columns.tolist()
        data_types = chunk.dtypes.astype(str)

        missing_counts = pd.Series(
            0,
            index=chunk.columns,
            dtype="int64"
        )

    total_rows += len(chunk)

    # Estimate memory if the whole dataset were loaded
    estimated_memory_mb += (
        chunk.memory_usage(deep=True).sum() / 1024**2
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_counts = (
        missing_counts
        .add(chunk.isna().sum(), fill_value=0)
        .astype("int64")
    )


    # --------------------------------------------------------
    # Transaction types
    # --------------------------------------------------------

    type_counts = type_counts.add(
        chunk["type"].value_counts(),
        fill_value=0
    )

    type_amount_sum = type_amount_sum.add(
        chunk.groupby("type")["amount"].sum(),
        fill_value=0
    )


    # --------------------------------------------------------
    # Fraud analysis
    # --------------------------------------------------------

    fraud_mask = chunk["isFraud"].eq(1)
    flagged_mask = chunk["isFlaggedFraud"].eq(1)

    fraud_count_chunk = int(fraud_mask.sum())
    flagged_count_chunk = int(flagged_mask.sum())

    total_fraud += fraud_count_chunk
    total_flagged += flagged_count_chunk

    fraud_amount_total += chunk.loc[
        fraud_mask,
        "amount"
    ].sum()

    legitimate_amount_total += chunk.loc[
        ~fraud_mask,
        "amount"
    ].sum()


    # --------------------------------------------------------
    # Fraud by transaction type
    # --------------------------------------------------------

    fraud_by_type = fraud_by_type.add(
        chunk.loc[
            fraud_mask,
            "type"
        ].value_counts(),
        fill_value=0
    )

    fraud_amount_by_type = fraud_amount_by_type.add(
        chunk.loc[
            fraud_mask
        ].groupby("type")["amount"].sum(),
        fill_value=0
    )


    # --------------------------------------------------------
    # Existing flagging rule performance
    # --------------------------------------------------------

    true_positive += int(
        (fraud_mask & flagged_mask).sum()
    )

    false_positive += int(
        (~fraud_mask & flagged_mask).sum()
    )

    false_negative += int(
        (fraud_mask & ~flagged_mask).sum()
    )

    true_negative += int(
        (~fraud_mask & ~flagged_mask).sum()
    )


    # --------------------------------------------------------
    # Numeric profiling
    # --------------------------------------------------------

    for col in numeric_columns:

        series = chunk[col].dropna()

        if len(series) > 0:

            numeric_min[col] = min(
                numeric_min[col],
                series.min()
            )

            numeric_max[col] = max(
                numeric_max[col],
                series.max()
            )

            numeric_sum[col] += series.sum()
            numeric_count[col] += series.count()

            zero_counts[col] += int(
                series.eq(0).sum()
            )

            negative_counts[col] += int(
                series.lt(0).sum()
            )


    # --------------------------------------------------------
    # Create hashes for duplicate and unique-ID analysis
    #
    # This avoids storing millions of long customer IDs
    # directly in Python memory.
    # --------------------------------------------------------

    row_hashes = pd.util.hash_pandas_object(
        chunk,
        index=False
    ).to_numpy(dtype=np.uint64)

    origin_hashes = pd.util.hash_pandas_object(
        chunk["nameOrig"],
        index=False
    ).to_numpy(dtype=np.uint64)

    destination_hashes = pd.util.hash_pandas_object(
        chunk["nameDest"],
        index=False
    ).to_numpy(dtype=np.uint64)


    row_file = TEMP_DIR / f"row_hash_{chunk_number}.npy"
    origin_file = TEMP_DIR / f"origin_hash_{chunk_number}.npy"
    destination_file = (
        TEMP_DIR
        / f"destination_hash_{chunk_number}.npy"
    )


    np.save(row_file, row_hashes)
    np.save(origin_file, origin_hashes)
    np.save(destination_file, destination_hashes)


    row_hash_files.append(row_file)
    origin_hash_files.append(origin_file)
    destination_hash_files.append(destination_file)


print("\nFull CSV scan completed.")


# ------------------------------------------------------------
# 5. Functions for hash-based distinct/duplicate analysis
# ------------------------------------------------------------

def combine_hash_files(file_list):

    arrays = [
        np.load(file, mmap_mode="r")
        for file in file_list
    ]

    return np.concatenate(arrays)


def count_unique_hashes(file_list):

    hashes = combine_hash_files(file_list)

    hashes.sort()

    if len(hashes) == 0:
        return 0

    unique_count = (
        1
        + np.count_nonzero(
            hashes[1:] != hashes[:-1]
        )
    )

    return int(unique_count)


def count_duplicate_hashes(file_list):

    hashes = combine_hash_files(file_list)

    hashes.sort()

    if len(hashes) == 0:
        return 0

    duplicate_count = np.count_nonzero(
        hashes[1:] == hashes[:-1]
    )

    return int(duplicate_count)


# ------------------------------------------------------------
# 6. Calculate unique accounts and duplicate rows
# ------------------------------------------------------------

print("\nCalculating unique origin accounts...")

unique_origin_accounts = count_unique_hashes(
    origin_hash_files
)

print("Calculating unique destination accounts...")

unique_destination_accounts = count_unique_hashes(
    destination_hash_files
)

print("Checking duplicate transactions...")

duplicate_rows = count_duplicate_hashes(
    row_hash_files
)


# ------------------------------------------------------------
# 7. Overall calculations
# ------------------------------------------------------------

total_transaction_value = (
    fraud_amount_total
    + legitimate_amount_total
)

average_transaction_amount = (
    total_transaction_value / total_rows
)

fraud_rate_pct = (
    total_fraud / total_rows * 100
)

fraud_value_share_pct = (
    fraud_amount_total
    / total_transaction_value
    * 100
)


# Existing flagging system metrics

if (true_positive + false_positive) > 0:

    flag_precision = (
        true_positive
        / (true_positive + false_positive)
        * 100
    )

else:
    flag_precision = 0


if (true_positive + false_negative) > 0:

    flag_recall = (
        true_positive
        / (true_positive + false_negative)
        * 100
    )

else:
    flag_recall = 0


# ------------------------------------------------------------
# 8. Create overall profile summary
# ------------------------------------------------------------

summary_df = pd.DataFrame({

    "Metric": [

        "Total rows",
        "Total columns",

        "Minimum step",
        "Maximum step",

        "Total transaction value",
        "Average transaction amount",
        "Minimum transaction amount",
        "Maximum transaction amount",

        "Fraud transactions",
        "Fraud rate (%)",
        "Fraud transaction value",
        "Fraud value share (%)",

        "Flagged transactions",
        "Flag precision (%)",
        "Flag recall (%)",

        "Unique origin accounts",
        "Unique destination accounts",

        "Duplicate rows",

        "Estimated full DataFrame memory (MB)"
    ],

    "Value": [

        total_rows,
        len(columns_seen),

        numeric_min["step"],
        numeric_max["step"],

        total_transaction_value,
        average_transaction_amount,
        numeric_min["amount"],
        numeric_max["amount"],

        total_fraud,
        fraud_rate_pct,
        fraud_amount_total,
        fraud_value_share_pct,

        total_flagged,
        flag_precision,
        flag_recall,

        unique_origin_accounts,
        unique_destination_accounts,

        duplicate_rows,

        estimated_memory_mb
    ]
})


# ------------------------------------------------------------
# 9. Missing-value summary
# ------------------------------------------------------------

missing_df = pd.DataFrame({

    "Column": missing_counts.index,

    "Missing_Count": missing_counts.values,

    "Missing_Percentage": (
        missing_counts.values
        / total_rows
        * 100
    )
})


# ------------------------------------------------------------
# 10. Data-type summary
# ------------------------------------------------------------

dtype_df = pd.DataFrame({

    "Column": data_types.index,

    "Data_Type": data_types.values
})


# ------------------------------------------------------------
# 11. Numeric summary
# ------------------------------------------------------------

numeric_summary_rows = []

for col in numeric_columns:

    mean_value = (
        numeric_sum[col]
        / numeric_count[col]
        if numeric_count[col] > 0
        else np.nan
    )

    numeric_summary_rows.append({

        "Column": col,

        "Minimum": numeric_min[col],

        "Maximum": numeric_max[col],

        "Mean": mean_value,

        "Zero_Count": zero_counts[col],

        "Negative_Count": negative_counts[col]
    })


numeric_summary_df = pd.DataFrame(
    numeric_summary_rows
)


# ------------------------------------------------------------
# 12. Transaction-type summary
# ------------------------------------------------------------

type_summary_df = pd.DataFrame({

    "Transaction_Count": type_counts,

    "Transaction_Value": type_amount_sum,

    "Fraud_Count": fraud_by_type,

    "Fraud_Value": fraud_amount_by_type

}).fillna(0)


type_summary_df["Fraud_Rate_Pct"] = (

    type_summary_df["Fraud_Count"]

    / type_summary_df["Transaction_Count"]

    * 100
)


type_summary_df["Fraud_Value_Rate_Pct"] = (

    type_summary_df["Fraud_Value"]

    / type_summary_df["Transaction_Value"]

    * 100
)


type_summary_df.index.name = "Transaction_Type"

type_summary_df.reset_index(
    inplace=True
)


# Convert count fields to integer

type_summary_df["Transaction_Count"] = (
    type_summary_df["Transaction_Count"]
    .astype(int)
)

type_summary_df["Fraud_Count"] = (
    type_summary_df["Fraud_Count"]
    .astype(int)
)


# ------------------------------------------------------------
# 13. Existing flagging-system performance
# ------------------------------------------------------------

flagging_df = pd.DataFrame({

    "Metric": [

        "True Positive",
        "False Positive",
        "True Negative",
        "False Negative",

        "Precision (%)",
        "Recall / Detection Rate (%)"
    ],

    "Value": [

        true_positive,
        false_positive,
        true_negative,
        false_negative,

        flag_precision,
        flag_recall
    ]
})


# ------------------------------------------------------------
# 14. Save profiling outputs
# ------------------------------------------------------------

summary_df.to_csv(
    OUTPUT_DIR / "step3_profile_summary.csv",
    index=False
)

missing_df.to_csv(
    OUTPUT_DIR / "step3_missing_values.csv",
    index=False
)

dtype_df.to_csv(
    OUTPUT_DIR / "step3_data_types.csv",
    index=False
)

numeric_summary_df.to_csv(
    OUTPUT_DIR / "step3_numeric_summary.csv",
    index=False
)

type_summary_df.to_csv(
    OUTPUT_DIR
    / "step3_transaction_type_summary.csv",
    index=False
)

flagging_df.to_csv(
    OUTPUT_DIR
    / "step3_flagging_performance.csv",
    index=False
)


# ------------------------------------------------------------
# 15. Create a readable profiling report
# ------------------------------------------------------------

REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_3_Full_Data_Profiling_Report.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "PROJECT 5 - FULL DATA PROFILING REPORT\n"
    )

    report.write("=" * 70 + "\n\n")

    report.write("OVERALL PROFILE\n\n")
    report.write(
        summary_df.to_string(index=False)
    )

    report.write("\n\n")

    report.write("MISSING VALUES\n\n")
    report.write(
        missing_df.to_string(index=False)
    )

    report.write("\n\n")

    report.write("TRANSACTION TYPE ANALYSIS\n\n")
    report.write(
        type_summary_df.to_string(index=False)
    )

    report.write("\n\n")

    report.write("FLAGGING SYSTEM PERFORMANCE\n\n")
    report.write(
        flagging_df.to_string(index=False)
    )

    report.write("\n")


# ------------------------------------------------------------
# 16. Remove temporary hash files
# ------------------------------------------------------------

shutil.rmtree(
    TEMP_DIR,
    ignore_errors=True
)


# ------------------------------------------------------------
# 17. Display final results
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL PROFILE")
print("=" * 70)

print(
    summary_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    missing_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("TRANSACTION TYPE / FRAUD PROFILE")
print("=" * 70)

print(
    type_summary_df.to_string(index=False)
)


print("\n" + "=" * 70)
print("EXISTING FRAUD-FLAGGING PERFORMANCE")
print("=" * 70)

print(
    flagging_df.to_string(index=False)
)


print("\nOutput files created successfully.")

print("\nReport:")
print(REPORT_FILE)

print("\nStep 3 completed successfully.")
