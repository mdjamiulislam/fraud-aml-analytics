import pandas as pd
import numpy as np
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PROJECT 5 - STEP 5
# CREATE MASTER ANALYTICAL DATASET
# AND FIRST FRAUD-RISK FEATURES
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"
DOCUMENTATION_DIR = PROJECT_ROOT / "06_documentation"

OUTPUT_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


MASTER_FILE = (
    OUTPUT_DIR
    / "master_analytical.parquet"
)

MODEL_FILE = (
    OUTPUT_DIR
    / "model_high_risk_population.parquet"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "step5_validation_summary.csv"
)

FEATURE_REGISTER_FILE = (
    DOCUMENTATION_DIR
    / "Step_5_Feature_Register.csv"
)

REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_5_Analytical_Dataset_Report.txt"
)


# ------------------------------------------------------------
# 2. Remove old Step 5 outputs if they exist
# ------------------------------------------------------------

for file in [
    MASTER_FILE,
    MODEL_FILE,
    VALIDATION_FILE,
    FEATURE_REGISTER_FILE,
    REPORT_FILE
]:
    if file.exists():
        file.unlink()


# ------------------------------------------------------------
# 3. Processing settings
# ------------------------------------------------------------

CHUNK_SIZE = 500_000

HIGH_RISK_TYPES = [
    "TRANSFER",
    "CASH_OUT"
]

BALANCE_TOLERANCE = 0.01


print("=" * 75)
print("PROJECT 5 - STEP 5")
print("CREATE MASTER ANALYTICAL DATASET")
print("=" * 75)

print("\nRaw data:")
print(RAW_FILE)

print("\nChunk size:")
print(f"{CHUNK_SIZE:,}")


# ------------------------------------------------------------
# 4. Initialise counters
# ------------------------------------------------------------

source_rows = 0
master_rows = 0
model_rows = 0

source_fraud = 0
master_fraud = 0
model_fraud = 0

zero_amount_transactions = 0
zero_amount_fraud = 0

high_value_transactions = 0

transaction_counter = 0


# Parquet writers
master_writer = None
model_writer = None


# ------------------------------------------------------------
# 5. Fields allowed in the primary modelling dataset
#
# IMPORTANT:
# newbalanceOrig
# newbalanceDest
# isFlaggedFraud
# raw account IDs
#
# are deliberately excluded.
# ------------------------------------------------------------

MODEL_COLUMNS = [

    # Traceability only
    "transaction_id",

    # Time
    "step",
    "transaction_day",
    "hour_of_day",

    # Transaction characteristics
    "type",
    "amount",
    "log_amount",

    # Pre-transaction balances
    "oldbalanceOrg",
    "oldbalanceDest",

    # Model-safe engineered features
    "zero_amount_flag",
    "high_value_200k_flag",

    "origin_zero_before_flag",
    "destination_zero_before_flag",

    "amount_exceeds_origin_balance_flag",

    "destination_is_merchant",

    "amount_to_origin_balance_ratio",
    "amount_to_destination_balance_ratio",

    # Target only
    "isFraud"
]


# ------------------------------------------------------------
# 6. Process full dataset in chunks
# ------------------------------------------------------------

print("\nBeginning Step 5 transformation...\n")


try:

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            RAW_FILE,
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        rows_in_chunk = len(chunk)

        print(
            f"Processing chunk {chunk_number} "
            f"| {rows_in_chunk:,} rows"
        )


        # ----------------------------------------------------
        # Source validation counters
        # ----------------------------------------------------

        source_rows += rows_in_chunk

        source_fraud += int(
            chunk["isFraud"].sum()
        )


        # ----------------------------------------------------
        # Unique analytical transaction ID
        # ----------------------------------------------------

        start_id = transaction_counter + 1

        end_id = (
            transaction_counter
            + rows_in_chunk
            + 1
        )

        chunk.insert(
            0,
            "transaction_id",
            np.arange(
                start_id,
                end_id,
                dtype=np.int64
            )
        )

        transaction_counter += rows_in_chunk


        # ----------------------------------------------------
        # Time features
        # ----------------------------------------------------

        chunk["transaction_day"] = (
            ((chunk["step"] - 1) // 24) + 1
        ).astype("int16")


        chunk["hour_of_day"] = (
            (chunk["step"] - 1) % 24
        ).astype("int8")


        # ----------------------------------------------------
        # Amount features
        # ----------------------------------------------------

        chunk["log_amount"] = np.log1p(
            chunk["amount"]
        )


        chunk["zero_amount_flag"] = (
            chunk["amount"].eq(0)
        ).astype("int8")


        chunk["high_value_200k_flag"] = (
            chunk["amount"].gt(200_000)
        ).astype("int8")


        # ----------------------------------------------------
        # Pre-transaction balance flags
        # ----------------------------------------------------

        chunk["origin_zero_before_flag"] = (
            chunk["oldbalanceOrg"].eq(0)
        ).astype("int8")


        chunk[
            "destination_zero_before_flag"
        ] = (
            chunk["oldbalanceDest"].eq(0)
        ).astype("int8")


        chunk[
            "amount_exceeds_origin_balance_flag"
        ] = (
            chunk["amount"]
            .gt(chunk["oldbalanceOrg"])
        ).astype("int8")


        # ----------------------------------------------------
        # Account-type feature
        #
        # M = merchant destination
        # ----------------------------------------------------

        chunk["destination_is_merchant"] = (
            chunk["nameDest"]
            .astype(str)
            .str.startswith("M")
        ).astype("int8")


        # ----------------------------------------------------
        # Amount-to-balance ratios
        #
        # If balance is zero, ratio is left as NaN.
        # The zero-balance flag preserves the business signal.
        # ----------------------------------------------------

        chunk[
            "amount_to_origin_balance_ratio"
        ] = (
            chunk["amount"]
            / chunk["oldbalanceOrg"].replace(
                0,
                np.nan
            )
        )


        chunk[
            "amount_to_destination_balance_ratio"
        ] = (
            chunk["amount"]
            / chunk["oldbalanceDest"].replace(
                0,
                np.nan
            )
        )


        # ----------------------------------------------------
        # Retrospective balance diagnostic
        #
        # THIS USES POST-TRANSACTION INFORMATION.
        # KEEP FOR EDA ONLY.
        # DO NOT USE IN PRIMARY MODEL.
        # ----------------------------------------------------

        cash_in_mask = chunk["type"].eq(
            "CASH_IN"
        )


        expected_new_origin_balance = (
            chunk["oldbalanceOrg"]
            - chunk["amount"]
        )


        expected_new_origin_balance.loc[
            cash_in_mask
        ] = (
            chunk.loc[
                cash_in_mask,
                "oldbalanceOrg"
            ]
            +
            chunk.loc[
                cash_in_mask,
                "amount"
            ]
        )


        chunk[
            "origin_balance_error_abs"
        ] = (
            expected_new_origin_balance
            - chunk["newbalanceOrig"]
        ).abs()


        chunk[
            "origin_balance_mismatch_flag"
        ] = (
            chunk[
                "origin_balance_error_abs"
            ]
            .gt(BALANCE_TOLERANCE)
        ).astype("int8")


        # ----------------------------------------------------
        # Data type optimisation for binary fields
        # ----------------------------------------------------

        chunk["isFraud"] = (
            chunk["isFraud"]
            .astype("int8")
        )


        chunk["isFlaggedFraud"] = (
            chunk["isFlaggedFraud"]
            .astype("int8")
        )


        # ----------------------------------------------------
        # Validation counters
        # ----------------------------------------------------

        master_rows += len(chunk)

        master_fraud += int(
            chunk["isFraud"].sum()
        )


        zero_amount_transactions += int(
            chunk["zero_amount_flag"].sum()
        )


        zero_amount_fraud += int(
            chunk.loc[
                chunk["zero_amount_flag"].eq(1),
                "isFraud"
            ].sum()
        )


        high_value_transactions += int(
            chunk[
                "high_value_200k_flag"
            ].sum()
        )


        # ----------------------------------------------------
        # Write master analytical dataset
        # ----------------------------------------------------

        master_table = pa.Table.from_pandas(
            chunk,
            preserve_index=False
        )


        if master_writer is None:

            master_writer = pq.ParquetWriter(
                str(MASTER_FILE),
                master_table.schema,
                compression="snappy",
                use_dictionary=True
            )


        master_writer.write_table(
            master_table
        )


        # ----------------------------------------------------
        # Create focused fraud modelling population
        #
        # Only CASH_OUT and TRANSFER
        # ----------------------------------------------------

        model_chunk = chunk.loc[
            chunk["type"].isin(
                HIGH_RISK_TYPES
            ),
            MODEL_COLUMNS
        ].copy()


        model_rows += len(
            model_chunk
        )


        model_fraud += int(
            model_chunk["isFraud"].sum()
        )


        # ----------------------------------------------------
        # Write modelling population
        # ----------------------------------------------------

        if len(model_chunk) > 0:

            model_table = pa.Table.from_pandas(
                model_chunk,
                preserve_index=False
            )


            if model_writer is None:

                model_writer = (
                    pq.ParquetWriter(
                        str(MODEL_FILE),
                        model_table.schema,
                        compression="snappy",
                        use_dictionary=True
                    )
                )


            model_writer.write_table(
                model_table
            )


finally:

    if master_writer is not None:
        master_writer.close()

    if model_writer is not None:
        model_writer.close()


print("\nTransformation completed.")


# ------------------------------------------------------------
# 7. Validate Parquet output row counts
# ------------------------------------------------------------

master_parquet = pq.ParquetFile(
    MASTER_FILE
)

model_parquet = pq.ParquetFile(
    MODEL_FILE
)


master_file_rows = (
    master_parquet.metadata.num_rows
)

model_file_rows = (
    model_parquet.metadata.num_rows
)


# ------------------------------------------------------------
# 8. Calculate fraud rates
# ------------------------------------------------------------

full_fraud_rate = (
    master_fraud
    / master_rows
    * 100
)


model_fraud_rate = (
    model_fraud
    / model_rows
    * 100
)


# ------------------------------------------------------------
# 9. File sizes
# ------------------------------------------------------------

master_size_mb = (
    MASTER_FILE.stat().st_size
    / 1024**2
)

model_size_mb = (
    MODEL_FILE.stat().st_size
    / 1024**2
)


# ------------------------------------------------------------
# 10. Validation summary
# ------------------------------------------------------------

validation_df = pd.DataFrame({

    "Metric": [

        "Source rows processed",

        "Master analytical rows",

        "Master Parquet rows",

        "Rows lost during processing",

        "Source fraud transactions",

        "Master fraud transactions",

        "Full dataset fraud rate (%)",

        "High-risk modelling rows",

        "High-risk Parquet rows",

        "High-risk modelling fraud",

        "High-risk fraud rate (%)",

        "Zero-amount transactions",

        "Zero-amount fraud transactions",

        "Transactions above 200k",

        "Master file size (MB)",

        "High-risk model file size (MB)"
    ],

    "Value": [

        source_rows,

        master_rows,

        master_file_rows,

        source_rows - master_file_rows,

        source_fraud,

        master_fraud,

        full_fraud_rate,

        model_rows,

        model_file_rows,

        model_fraud,

        model_fraud_rate,

        zero_amount_transactions,

        zero_amount_fraud,

        high_value_transactions,

        master_size_mb,

        model_size_mb
    ]
})


validation_df.to_csv(
    VALIDATION_FILE,
    index=False
)


# ------------------------------------------------------------
# 11. Feature register
# ------------------------------------------------------------

feature_register_df = pd.DataFrame({

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

        "origin_balance_error_abs",

        "origin_balance_mismatch_flag",

        "newbalanceOrig",
        "newbalanceDest",

        "nameOrig",
        "nameDest",

        "isFlaggedFraud",

        "isFraud"
    ],

    "Purpose": [

        "Unique transaction key",

        "Simulation time step",
        "Derived simulation day",
        "Derived simulated hour",

        "Transaction category",

        "Original transaction value",
        "Log-transformed transaction value",

        "Pre-transaction origin balance",
        "Pre-transaction destination balance",

        "Flags amount equal to zero",

        "Flags amount above 200k",

        "Flags zero origin balance before transaction",

        "Flags zero destination balance before transaction",

        "Flags transaction amount above origin balance",

        "Identifies merchant destination",

        "Transaction amount relative to origin balance",

        "Transaction amount relative to destination balance",

        "Retrospective origin balance diagnostic",

        "Retrospective balance mismatch indicator",

        "Post-transaction origin balance",
        "Post-transaction destination balance",

        "Raw origin identifier",
        "Raw destination identifier",

        "Existing rule-based fraud flag",

        "Actual fraud target"
    ],

    "Primary_Model_Use": [

        "NO - traceability only",

        "CANDIDATE",
        "YES",
        "YES",

        "YES",

        "YES",
        "YES",

        "YES",
        "YES",

        "YES",

        "YES",

        "YES",

        "YES",

        "YES",

        "YES",

        "CANDIDATE",

        "CANDIDATE",

        "NO - post-transaction diagnostic",

        "NO - post-transaction diagnostic",

        "NO - post-transaction",
        "NO - post-transaction",

        "NO - raw high-cardinality ID",
        "NO - raw high-cardinality ID",

        "NO - benchmark only",

        "TARGET ONLY"
    ]
})


feature_register_df.to_csv(
    FEATURE_REGISTER_FILE,
    index=False
)


# ------------------------------------------------------------
# 12. Create report
# ------------------------------------------------------------

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "PROJECT 5 - STEP 5\n"
    )

    report.write(
        "MASTER ANALYTICAL DATASET "
        "AND FEATURE ENGINEERING\n"
    )

    report.write("=" * 75)
    report.write("\n\n")

    report.write(
        "VALIDATION SUMMARY\n\n"
    )

    report.write(
        validation_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "FEATURE REGISTER\n\n"
    )

    report.write(
        feature_register_df.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# 13. Display sample from the master Parquet file
# ------------------------------------------------------------

first_row_group = (
    master_parquet
    .read_row_group(0)
    .slice(0, 5)
    .to_pandas()
)


print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print(
    validation_df.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("FIRST FIVE PROCESSED TRANSACTIONS")
print("=" * 75)

print(
    first_row_group[
        [
            "transaction_id",
            "step",
            "transaction_day",
            "hour_of_day",
            "type",
            "amount",
            "log_amount",
            "zero_amount_flag",
            "high_value_200k_flag",
            "destination_is_merchant",
            "isFraud"
        ]
    ].to_string(index=False)
)


print("\nMaster analytical dataset:")
print(MASTER_FILE)

print("\nHigh-risk modelling dataset:")
print(MODEL_FILE)

print("\nStep 5 completed successfully.")
