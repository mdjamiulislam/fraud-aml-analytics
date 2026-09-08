import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT 5 - STEP 4
# DATA QUALITY ASSESSMENT & CLEANING RULES
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "05_outputs"
DOCUMENTATION_DIR = PROJECT_ROOT / "06_documentation"

OUTPUT_DIR.mkdir(exist_ok=True)
DOCUMENTATION_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# 2. Settings
# ------------------------------------------------------------

CHUNK_SIZE = 500_000

ALLOWED_TYPES = {
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER"
}

BALANCE_COLUMNS = [
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

NUMERIC_COLUMNS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud"
]

BALANCE_TOLERANCE = 0.01


# ------------------------------------------------------------
# 3. Initialise counters
# ------------------------------------------------------------

total_rows = 0

zero_amount_count = 0
negative_amount_count = 0

zero_amount_fraud = 0
zero_amount_legitimate = 0

invalid_step_count = 0
invalid_type_count = 0

invalid_isfraud_count = 0
invalid_isflagged_count = 0

blank_origin_count = 0
blank_destination_count = 0

nonfinite_numeric_count = 0

negative_balance_counts = {
    col: 0 for col in BALANCE_COLUMNS
}

zero_balance_counts = {
    col: 0 for col in BALANCE_COLUMNS
}

origin_prefix_counts = pd.Series(dtype="float64")
destination_prefix_counts = pd.Series(dtype="float64")

type_counts = pd.Series(dtype="float64")

high_value_type_counts = pd.Series(dtype="float64")
high_value_fraud_counts = pd.Series(dtype="float64")

outgoing_balance_mismatch = 0
cash_in_balance_mismatch = 0

outgoing_balance_checked = 0
cash_in_balance_checked = 0


# ------------------------------------------------------------
# 4. Start scan
# ------------------------------------------------------------

print("=" * 72)
print("PROJECT 5 - STEP 4: DATA QUALITY ASSESSMENT")
print("=" * 72)

print("\nScanning full dataset...\n")


for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE),
    start=1
):

    print(
        f"Processing chunk {chunk_number} "
        f"| {len(chunk):,} rows"
    )

    total_rows += len(chunk)


    # --------------------------------------------------------
    # Transaction amount checks
    # --------------------------------------------------------

    zero_amount_mask = chunk["amount"].eq(0)
    negative_amount_mask = chunk["amount"].lt(0)

    zero_amount_count += int(
        zero_amount_mask.sum()
    )

    negative_amount_count += int(
        negative_amount_mask.sum()
    )

    zero_amount_fraud += int(
        (
            zero_amount_mask
            & chunk["isFraud"].eq(1)
        ).sum()
    )

    zero_amount_legitimate += int(
        (
            zero_amount_mask
            & chunk["isFraud"].eq(0)
        ).sum()
    )


    # --------------------------------------------------------
    # Step validation
    # --------------------------------------------------------

    invalid_step_mask = (
        chunk["step"].isna()
        | chunk["step"].lt(1)
        | chunk["step"].gt(743)
    )

    invalid_step_count += int(
        invalid_step_mask.sum()
    )


    # --------------------------------------------------------
    # Transaction-type validation
    # --------------------------------------------------------

    invalid_type_mask = (
        chunk["type"].isna()
        | ~chunk["type"].isin(ALLOWED_TYPES)
    )

    invalid_type_count += int(
        invalid_type_mask.sum()
    )


    # --------------------------------------------------------
    # Binary target/flag validation
    # --------------------------------------------------------

    invalid_isfraud_count += int(
        (
            chunk["isFraud"].isna()
            | ~chunk["isFraud"].isin([0, 1])
        ).sum()
    )

    invalid_isflagged_count += int(
        (
            chunk["isFlaggedFraud"].isna()
            | ~chunk["isFlaggedFraud"].isin([0, 1])
        ).sum()
    )


    # --------------------------------------------------------
    # Identifier checks
    # --------------------------------------------------------

    origin_blank_mask = (
        chunk["nameOrig"].isna()
        | chunk["nameOrig"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    destination_blank_mask = (
        chunk["nameDest"].isna()
        | chunk["nameDest"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    blank_origin_count += int(
        origin_blank_mask.sum()
    )

    blank_destination_count += int(
        destination_blank_mask.sum()
    )


    # --------------------------------------------------------
    # Identifier prefix profiling
    # --------------------------------------------------------

    origin_prefix = (
        chunk["nameOrig"]
        .astype(str)
        .str[0]
    )

    destination_prefix = (
        chunk["nameDest"]
        .astype(str)
        .str[0]
    )

    origin_prefix_counts = origin_prefix_counts.add(
        origin_prefix.value_counts(),
        fill_value=0
    )

    destination_prefix_counts = (
        destination_prefix_counts.add(
            destination_prefix.value_counts(),
            fill_value=0
        )
    )


    # --------------------------------------------------------
    # Non-finite numeric values
    # --------------------------------------------------------

    numeric_array = chunk[
        NUMERIC_COLUMNS
    ].to_numpy(dtype="float64")

    nonfinite_numeric_count += int(
        (~np.isfinite(numeric_array)).sum()
    )


    # --------------------------------------------------------
    # Balance checks
    # --------------------------------------------------------

    for col in BALANCE_COLUMNS:

        negative_balance_counts[col] += int(
            chunk[col].lt(0).sum()
        )

        zero_balance_counts[col] += int(
            chunk[col].eq(0).sum()
        )


    # --------------------------------------------------------
    # Origin balance consistency
    #
    # IMPORTANT:
    # These are diagnostic flags, NOT automatic errors.
    # --------------------------------------------------------

    outgoing_mask = chunk["type"].isin(
        [
            "CASH_OUT",
            "TRANSFER",
            "PAYMENT",
            "DEBIT"
        ]
    )

    outgoing_delta = (
        chunk["oldbalanceOrg"]
        - chunk["newbalanceOrig"]
    )

    outgoing_difference = (
        outgoing_delta
        - chunk["amount"]
    ).abs()

    outgoing_balance_checked += int(
        outgoing_mask.sum()
    )

    outgoing_balance_mismatch += int(
        (
            outgoing_mask
            & outgoing_difference.gt(
                BALANCE_TOLERANCE
            )
        ).sum()
    )


    # CASH_IN normally increases the origin balance

    cash_in_mask = chunk["type"].eq(
        "CASH_IN"
    )

    cash_in_delta = (
        chunk["newbalanceOrig"]
        - chunk["oldbalanceOrg"]
    )

    cash_in_difference = (
        cash_in_delta
        - chunk["amount"]
    ).abs()

    cash_in_balance_checked += int(
        cash_in_mask.sum()
    )

    cash_in_balance_mismatch += int(
        (
            cash_in_mask
            & cash_in_difference.gt(
                BALANCE_TOLERANCE
            )
        ).sum()
    )


    # --------------------------------------------------------
    # Transaction type profile
    # --------------------------------------------------------

    type_counts = type_counts.add(
        chunk["type"].value_counts(),
        fill_value=0
    )


    # --------------------------------------------------------
    # High-value transactions
    #
    # Used for profiling only.
    # NOT an outlier-deletion rule.
    # --------------------------------------------------------

    high_value_mask = chunk["amount"].gt(
        200_000
    )

    high_value_type_counts = (
        high_value_type_counts.add(
            chunk.loc[
                high_value_mask,
                "type"
            ].value_counts(),
            fill_value=0
        )
    )

    high_value_fraud_counts = (
        high_value_fraud_counts.add(
            chunk.loc[
                high_value_mask
                & chunk["isFraud"].eq(1),
                "type"
            ].value_counts(),
            fill_value=0
        )
    )


print("\nFull Step 4 scan completed.")


# ------------------------------------------------------------
# 5. Main data-quality audit table
# ------------------------------------------------------------

audit_df = pd.DataFrame({

    "Check": [

        "Total rows",

        "Zero transaction amounts",
        "Negative transaction amounts",

        "Zero-amount fraud transactions",
        "Zero-amount legitimate transactions",

        "Invalid step values",
        "Invalid transaction types",

        "Invalid isFraud values",
        "Invalid isFlaggedFraud values",

        "Blank origin IDs",
        "Blank destination IDs",

        "Non-finite numeric values",

        "Outgoing origin-balance mismatches",
        "CASH_IN origin-balance mismatches"
    ],

    "Count": [

        total_rows,

        zero_amount_count,
        negative_amount_count,

        zero_amount_fraud,
        zero_amount_legitimate,

        invalid_step_count,
        invalid_type_count,

        invalid_isfraud_count,
        invalid_isflagged_count,

        blank_origin_count,
        blank_destination_count,

        nonfinite_numeric_count,

        outgoing_balance_mismatch,
        cash_in_balance_mismatch
    ]
})


# ------------------------------------------------------------
# 6. Balance summary
# ------------------------------------------------------------

balance_summary_rows = []

for col in BALANCE_COLUMNS:

    balance_summary_rows.append({

        "Column": col,

        "Zero_Count":
            zero_balance_counts[col],

        "Zero_Percentage":
            zero_balance_counts[col]
            / total_rows
            * 100,

        "Negative_Count":
            negative_balance_counts[col]
    })


balance_summary_df = pd.DataFrame(
    balance_summary_rows
)


# ------------------------------------------------------------
# 7. Identifier-prefix summary
# ------------------------------------------------------------

all_prefixes = sorted(
    set(origin_prefix_counts.index)
    | set(destination_prefix_counts.index)
)

prefix_rows = []

for prefix in all_prefixes:

    prefix_rows.append({

        "Prefix": prefix,

        "Origin_Count":
            int(
                origin_prefix_counts.get(
                    prefix,
                    0
                )
            ),

        "Destination_Count":
            int(
                destination_prefix_counts.get(
                    prefix,
                    0
                )
            )
    })


prefix_df = pd.DataFrame(
    prefix_rows
)


# ------------------------------------------------------------
# 8. High-value transaction profile
# ------------------------------------------------------------

all_types = sorted(
    set(type_counts.index)
)

high_value_rows = []

for transaction_type in all_types:

    high_value_rows.append({

        "Transaction_Type":
            transaction_type,

        "Total_Transactions":
            int(
                type_counts.get(
                    transaction_type,
                    0
                )
            ),

        "Transactions_Above_200k":
            int(
                high_value_type_counts.get(
                    transaction_type,
                    0
                )
            ),

        "Fraud_Above_200k":
            int(
                high_value_fraud_counts.get(
                    transaction_type,
                    0
                )
            )
    })


high_value_df = pd.DataFrame(
    high_value_rows
)


# ------------------------------------------------------------
# 9. Balance-consistency summary
# ------------------------------------------------------------

balance_consistency_df = pd.DataFrame({

    "Check": [

        "Outgoing transactions checked",

        "Outgoing origin-balance mismatches",

        "CASH_IN transactions checked",

        "CASH_IN origin-balance mismatches"
    ],

    "Count": [

        outgoing_balance_checked,

        outgoing_balance_mismatch,

        cash_in_balance_checked,

        cash_in_balance_mismatch
    ]
})


# ------------------------------------------------------------
# 10. Field decision register
# ------------------------------------------------------------

field_decisions_df = pd.DataFrame({

    "Field": [

        "step",
        "type",
        "amount",

        "nameOrig",
        "nameDest",

        "oldbalanceOrg",
        "newbalanceOrig",

        "oldbalanceDest",
        "newbalanceDest",

        "isFraud",
        "isFlaggedFraud"
    ],

    "Business_Role": [

        "Transaction timing",
        "Transaction category",
        "Transaction value",

        "Origin identifier",
        "Destination identifier",

        "Pre-transaction origin balance",
        "Post-transaction origin balance",

        "Pre-transaction destination balance",
        "Post-transaction destination balance",

        "Fraud outcome",
        "Existing rule-based flag"
    ],

    "Data_Treatment": [

        "KEEP",
        "KEEP",
        "KEEP",

        "KEEP AS ID",
        "KEEP AS ID",

        "KEEP",
        "KEEP FOR EDA ONLY",

        "KEEP",
        "KEEP FOR EDA ONLY",

        "TARGET ONLY",
        "BENCHMARK ONLY"
    ],

    "Primary_Model_Use": [

        "YES",
        "YES",
        "YES",

        "NO - raw identifier",
        "NO - raw identifier",

        "YES - candidate predictor",
        "NO - post-transaction information",

        "YES - candidate predictor",
        "NO - post-transaction information",

        "NO - prediction target",
        "NO - existing rule output"
    ],

    "Notes": [

        "Later derive day/hour features",

        "Later encode categorical values",

        "Do not delete high values simply as outliers",

        "Can support investigation or historical features",

        "Can support destination/merchant features",

        "Compare models with and without balance variables",

        "Potential operational leakage",

        "Compare models with and without balance variables",

        "Potential operational leakage",

        "Ground-truth fraud label",

        "Used to benchmark current detection rule"
    ]
})


# ------------------------------------------------------------
# 11. Cleaning-rule register
# ------------------------------------------------------------

cleaning_rules_df = pd.DataFrame({

    "Issue": [

        "Missing values",
        "Exact duplicate rows",

        "Negative transaction amount",
        "Zero transaction amount",

        "Invalid transaction type",
        "Invalid step",

        "Invalid fraud label",
        "Invalid existing flag",

        "Blank account identifiers",

        "Zero balances",
        "Balance inconsistencies",

        "Very large transaction amounts",

        "Post-transaction balance fields",

        "Raw account identifiers",

        "CASH_IN / DEBIT / PAYMENT"
    ],

    "Proposed_Action": [

        "Investigate; remove/impute only if required",
        "Remove exact duplicates if any exist",

        "Treat as invalid and exclude if found",
        "Investigate before deciding",

        "Treat as invalid and exclude if found",
        "Treat as invalid and exclude if found",

        "Treat as invalid because target must be binary",
        "Treat as invalid if outside 0/1",

        "Investigate and exclude if truly missing",

        "RETAIN - zero is not the same as missing",
        "RETAIN and create diagnostic flags",

        "RETAIN - important for fraud analysis",

        "Keep for EDA but exclude from primary model",

        "Keep as identifiers; exclude raw IDs from model",

        "Keep in master analytical dataset"
    ],

    "Reason": [

        "No assumption should be made before profiling",

        "Duplicates can bias volume and model results",

        "Transaction amounts should not be negative",

        "Zero may represent a valid or unusual transaction",

        "Only five documented transaction types expected",

        "Dataset period is steps 1 through 743",

        "isFraud is the binary target",

        "Existing flag must be 0 or 1",

        "Identifiers support traceability",

        "Zero balances can be valid structural values",

        "Mismatch may reflect simulation/business behaviour, not bad data",

        "Fraud cases are disproportionately high value",

        "They describe balances after transaction completion",

        "High-cardinality raw IDs do not generalise well",

        "Required for overall reporting even though fraud is absent in them"
    ]
})


# ------------------------------------------------------------
# 12. Save outputs
# ------------------------------------------------------------

audit_df.to_csv(
    OUTPUT_DIR
    / "step4_data_quality_audit.csv",
    index=False
)

balance_summary_df.to_csv(
    OUTPUT_DIR
    / "step4_balance_summary.csv",
    index=False
)

prefix_df.to_csv(
    OUTPUT_DIR
    / "step4_identifier_prefix_summary.csv",
    index=False
)

high_value_df.to_csv(
    OUTPUT_DIR
    / "step4_high_value_profile.csv",
    index=False
)

balance_consistency_df.to_csv(
    OUTPUT_DIR
    / "step4_balance_consistency.csv",
    index=False
)

field_decisions_df.to_csv(
    DOCUMENTATION_DIR
    / "Step_4_Field_Decision_Register.csv",
    index=False
)

cleaning_rules_df.to_csv(
    DOCUMENTATION_DIR
    / "Step_4_Cleaning_Rule_Register.csv",
    index=False
)


# ------------------------------------------------------------
# 13. Create text report
# ------------------------------------------------------------

REPORT_FILE = (
    DOCUMENTATION_DIR
    / "Step_4_Data_Quality_and_Cleaning_Rules.txt"
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "PROJECT 5 - STEP 4\n"
    )

    report.write(
        "DATA QUALITY ASSESSMENT & CLEANING RULES\n"
    )

    report.write("=" * 72 + "\n\n")

    report.write(
        "DATA QUALITY AUDIT\n\n"
    )

    report.write(
        audit_df.to_string(index=False)
    )

    report.write("\n\n")

    report.write(
        "BALANCE SUMMARY\n\n"
    )

    report.write(
        balance_summary_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "BALANCE CONSISTENCY\n\n"
    )

    report.write(
        balance_consistency_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "IDENTIFIER PREFIXES\n\n"
    )

    report.write(
        prefix_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "HIGH VALUE PROFILE\n\n"
    )

    report.write(
        high_value_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "FIELD DECISION REGISTER\n\n"
    )

    report.write(
        field_decisions_df.to_string(
            index=False
        )
    )

    report.write("\n\n")

    report.write(
        "CLEANING RULE REGISTER\n\n"
    )

    report.write(
        cleaning_rules_df.to_string(
            index=False
        )
    )


# ------------------------------------------------------------
# 14. Display results
# ------------------------------------------------------------

print("\n" + "=" * 72)
print("DATA QUALITY AUDIT")
print("=" * 72)

print(
    audit_df.to_string(index=False)
)


print("\n" + "=" * 72)
print("BALANCE SUMMARY")
print("=" * 72)

print(
    balance_summary_df.to_string(
        index=False
    )
)


print("\n" + "=" * 72)
print("BALANCE CONSISTENCY")
print("=" * 72)

print(
    balance_consistency_df.to_string(
        index=False
    )
)


print("\n" + "=" * 72)
print("IDENTIFIER PREFIX SUMMARY")
print("=" * 72)

print(
    prefix_df.to_string(
        index=False
    )
)


print("\n" + "=" * 72)
print("HIGH VALUE PROFILE")
print("=" * 72)

print(
    high_value_df.to_string(
        index=False
    )
)


print("\nStep 4 output files created successfully.")

print("\nReport:")
print(REPORT_FILE)

print("\nStep 4 completed successfully.")