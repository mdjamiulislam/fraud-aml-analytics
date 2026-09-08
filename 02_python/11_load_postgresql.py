import pandas as pd
from pathlib import Path
from io import StringIO
from getpass import getpass

import pyarrow.parquet as pq
import psycopg2


# ============================================================
# PROJECT 5 - STEP 12
# LOAD FRAUD ANALYTICS OUTPUTS INTO POSTGRESQL
# ============================================================


# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "05_outputs"
)


SCORED_TEST_FILE = (
    OUTPUT_DIR
    / "step11_scored_high_risk_test.parquet"
)


ALERT_QUEUE_FILE = (
    OUTPUT_DIR
    / "step11_investigator_alert_queue.csv"
)


RISK_THRESHOLD_FILE = (
    OUTPUT_DIR
    / "step11_risk_tier_thresholds.csv"
)


RISK_SUMMARY_FILE = (
    OUTPUT_DIR
    / "step11_risk_tier_summary.csv"
)


OPERATIONAL_SUMMARY_FILE = (
    OUTPUT_DIR
    / "step11_operational_alert_summary.csv"
)


CAPACITY_FILE = (
    OUTPUT_DIR
    / "step11_investigation_capacity_summary.csv"
)


MODEL_COMPARISON_FILE = (
    OUTPUT_DIR
    / "step10_model_comparison.csv"
)


FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "step10_hgb_permutation_importance.csv"
)


# ------------------------------------------------------------
# 2. PostgreSQL settings
# ------------------------------------------------------------

DB_HOST = "localhost"

DB_PORT = 5432

DB_NAME = "fraud_aml_analytics"

DB_USER = "postgres"


print("=" * 82)

print(
    "PROJECT 5 - STEP 12"
)

print(
    "LOAD FRAUD ANALYTICS DATA INTO POSTGRESQL"
)

print("=" * 82)


# ------------------------------------------------------------
# 3. Request password securely
# ------------------------------------------------------------

DB_PASSWORD = getpass(
    "\nEnter PostgreSQL password: "
)


# ------------------------------------------------------------
# 4. Connect to PostgreSQL
# ------------------------------------------------------------

print(
    "\nConnecting to PostgreSQL..."
)


connection = psycopg2.connect(

    host=DB_HOST,

    port=DB_PORT,

    database=DB_NAME,

    user=DB_USER,

    password=DB_PASSWORD
)


connection.autocommit = False


print(
    "Connection successful."
)


# ============================================================
# PART A
# COPY HELPER
# ============================================================


# ------------------------------------------------------------
# 5. Bulk-copy DataFrame into PostgreSQL
# ------------------------------------------------------------

def copy_dataframe(
    conn,
    dataframe,
    table_name,
    columns
):

    buffer = StringIO()


    dataframe[
        columns
    ].to_csv(

        buffer,

        index=False,

        header=False,

        na_rep=""

    )


    buffer.seek(0)


    column_sql = ", ".join(
        columns
    )


    copy_sql = f"""

        COPY fraud_analytics.{table_name}
        ({column_sql})

        FROM STDIN

        WITH (
            FORMAT CSV,
            NULL '',
            QUOTE '"'
        )

    """


    with conn.cursor() as cursor:

        cursor.copy_expert(
            copy_sql,
            buffer
        )


# ============================================================
# PART B
# CLEAR TARGET TABLES
# ============================================================


# ------------------------------------------------------------
# 6. Truncate before reload
# ------------------------------------------------------------

print(
    "\nClearing analytical tables..."
)


with connection.cursor() as cursor:

    cursor.execute(
        """

        TRUNCATE TABLE

            fraud_analytics.fact_scored_transactions,

            fraud_analytics.investigator_alert_queue,

            fraud_analytics.risk_tier_thresholds,

            fraud_analytics.risk_tier_summary,

            fraud_analytics.operational_alert_summary,

            fraud_analytics.investigation_capacity_summary,

            fraud_analytics.model_comparison,

            fraud_analytics.hgb_feature_importance;

        """
    )


connection.commit()


print(
    "Tables cleared."
)


# ============================================================
# PART C
# LOAD MAIN SCORED FACT TABLE
# ============================================================


# ------------------------------------------------------------
# 7. PostgreSQL field mapping
# ------------------------------------------------------------

FACT_SOURCE_COLUMNS = [

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


FACT_DATABASE_COLUMNS = [

    "transaction_id",

    "step",

    "transaction_day",

    "hour_of_day",

    "transaction_type",

    "amount",

    "origin_account",

    "destination_account",

    "oldbalance_org",

    "oldbalance_dest",

    "high_value_200k_flag",

    "hgb_fraud_score",

    "risk_tier",

    "operational_alert",

    "legacy_rule_flag",

    "actual_fraud"
]


# ------------------------------------------------------------
# 8. Stream Parquet in batches
# ------------------------------------------------------------

print(
    "\nLoading fact_scored_transactions..."
)


parquet_file = pq.ParquetFile(
    SCORED_TEST_FILE
)


fact_rows_loaded = 0


for batch_number, batch in enumerate(

    parquet_file.iter_batches(
        batch_size=100_000
    ),

    start=1

):

    df = batch.to_pandas()


    df = df[
        FACT_SOURCE_COLUMNS
    ].copy()


    df.columns = (
        FACT_DATABASE_COLUMNS
    )


    copy_dataframe(

        connection,

        df,

        "fact_scored_transactions",

        FACT_DATABASE_COLUMNS

    )


    connection.commit()


    fact_rows_loaded += len(
        df
    )


    print(

        f"Fact batch {batch_number} "
        f"| loaded {len(df):,} "
        f"| cumulative {fact_rows_loaded:,}"

    )


print(
    "\nMain fact table loaded."
)


# ============================================================
# PART D
# LOAD INVESTIGATOR QUEUE
# ============================================================


print(
    "\nLoading investigator_alert_queue..."
)


queue_df = pd.read_csv(
    ALERT_QUEUE_FILE
)


QUEUE_RENAME = {

    "type":
        "transaction_type",

    "nameOrig":
        "origin_account",

    "nameDest":
        "destination_account",

    "oldbalanceOrg":
        "oldbalance_org",

    "oldbalanceDest":
        "oldbalance_dest",

    "isFlaggedFraud":
        "legacy_rule_flag"
}


queue_df = queue_df.rename(
    columns=QUEUE_RENAME
)


QUEUE_COLUMNS = [

    "investigation_rank",

    "transaction_id",

    "step",

    "transaction_day",

    "hour_of_day",

    "transaction_type",

    "amount",

    "origin_account",

    "destination_account",

    "oldbalance_org",

    "oldbalance_dest",

    "hgb_fraud_score",

    "risk_tier",

    "priority_band",

    "high_value_200k_flag",

    "legacy_rule_flag"
]


copy_dataframe(

    connection,

    queue_df,

    "investigator_alert_queue",

    QUEUE_COLUMNS
)


connection.commit()


print(
    f"Loaded {len(queue_df):,} investigator alerts."
)


# ============================================================
# PART E
# SMALL SUMMARY TABLES
# ============================================================


# ------------------------------------------------------------
# 9. Generic CSV load helper
# ------------------------------------------------------------

def load_small_csv(
    conn,
    filepath,
    table_name,
    rename_map,
    database_columns
):

    df = pd.read_csv(
        filepath
    )


    df = df.rename(
        columns=rename_map
    )


    copy_dataframe(

        conn,

        df,

        table_name,

        database_columns

    )


    conn.commit()


    print(

        f"{table_name}: "
        f"{len(df):,} rows loaded."

    )


# ------------------------------------------------------------
# 10. Risk thresholds
# ------------------------------------------------------------

load_small_csv(

    connection,

    RISK_THRESHOLD_FILE,

    "risk_tier_thresholds",

    {

        "Threshold_Type":
            "threshold_type",

        "Source":
            "source",

        "Score_Threshold":
            "score_threshold"

    },

    [

        "threshold_type",

        "source",

        "score_threshold"

    ]

)


# ------------------------------------------------------------
# 11. Risk-tier summary
# ------------------------------------------------------------

load_small_csv(

    connection,

    RISK_SUMMARY_FILE,

    "risk_tier_summary",

    {

        "Risk_Tier":
            "risk_tier",

        "Transaction_Count":
            "transaction_count",

        "Population_Pct":
            "population_pct",

        "Average_Model_Score":
            "average_model_score",

        "Transaction_Value":
            "transaction_value",

        "Fraud_Count":
            "fraud_count",

        "Fraud_Rate_Pct":
            "fraud_rate_pct",

        "Fraud_Capture_Pct":
            "fraud_capture_pct",

        "Fraud_Value":
            "fraud_value",

        "Fraud_Value_Capture_Pct":
            "fraud_value_capture_pct",

        "Operational_Alerts":
            "operational_alerts"

    },

    [

        "risk_tier",

        "transaction_count",

        "population_pct",

        "average_model_score",

        "transaction_value",

        "fraud_count",

        "fraud_rate_pct",

        "fraud_capture_pct",

        "fraud_value",

        "fraud_value_capture_pct",

        "operational_alerts"

    ]

)


# ------------------------------------------------------------
# 12. Operational summary
# ------------------------------------------------------------

load_small_csv(

    connection,

    OPERATIONAL_SUMMARY_FILE,

    "operational_alert_summary",

    {

        "Metric":
            "metric",

        "Value":
            "value"

    },

    [

        "metric",

        "value"

    ]

)


# ------------------------------------------------------------
# 13. Investigation capacity
# ------------------------------------------------------------

load_small_csv(

    connection,

    CAPACITY_FILE,

    "investigation_capacity_summary",

    {

        "Investigation_Capacity":
            "investigation_capacity",

        "Alerts_Reviewed":
            "alerts_reviewed",

        "Fraud_Found":
            "fraud_found",

        "Precision_Pct":
            "precision_pct",

        "Total_Test_Fraud_Captured_Pct":
            "total_test_fraud_captured_pct"

    },

    [

        "investigation_capacity",

        "alerts_reviewed",

        "fraud_found",

        "precision_pct",

        "total_test_fraud_captured_pct"

    ]

)


# ------------------------------------------------------------
# 14. Model comparison
# ------------------------------------------------------------

load_small_csv(

    connection,

    MODEL_COMPARISON_FILE,

    "model_comparison",

    {

        "Model":
            "model",

        "Threshold":
            "threshold",

        "ROC_AUC":
            "roc_auc",

        "Average_Precision":
            "average_precision",

        "Accuracy":
            "accuracy",

        "Precision":
            "precision",

        "Recall":
            "recall",

        "F1":
            "f1",

        "F2":
            "f2",

        "True_Negative":
            "true_negative",

        "False_Positive":
            "false_positive",

        "False_Negative":
            "false_negative",

        "True_Positive":
            "true_positive",

        "Alerts":
            "alerts",

        "Alert_Rate_Pct":
            "alert_rate_pct",

        "False_Positive_Rate_Pct":
            "false_positive_rate_pct"

    },

    [

        "model",

        "threshold",

        "roc_auc",

        "average_precision",

        "accuracy",

        "precision",

        "recall",

        "f1",

        "f2",

        "true_negative",

        "false_positive",

        "false_negative",

        "true_positive",

        "alerts",

        "alert_rate_pct",

        "false_positive_rate_pct"

    ]

)


# ------------------------------------------------------------
# 15. Feature importance
# ------------------------------------------------------------

load_small_csv(

    connection,

    FEATURE_IMPORTANCE_FILE,

    "hgb_feature_importance",

    {

        "Feature":
            "feature",

        "Importance_Mean":
            "importance_mean",

        "Importance_STD":
            "importance_std"

    },

    [

        "feature",

        "importance_mean",

        "importance_std"

    ]

)


# ============================================================
# PART F
# VALIDATE DATABASE LOAD
# ============================================================


print(
    "\nValidating database row counts..."
)


validation_queries = {

    "fact_scored_transactions":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.fact_scored_transactions
        """,

    "investigator_alert_queue":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.investigator_alert_queue
        """,

    "risk_tier_thresholds":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.risk_tier_thresholds
        """,

    "risk_tier_summary":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.risk_tier_summary
        """,

    "operational_alert_summary":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.operational_alert_summary
        """,

    "investigation_capacity_summary":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.investigation_capacity_summary
        """,

    "model_comparison":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.model_comparison
        """,

    "hgb_feature_importance":

        """
        SELECT COUNT(*)
        FROM fraud_analytics.hgb_feature_importance
        """
}


validation_rows = []


with connection.cursor() as cursor:

    for table_name, sql in (
        validation_queries.items()
    ):

        cursor.execute(
            sql
        )

        row_count = (
            cursor.fetchone()[0]
        )


        validation_rows.append({

            "Table":
                table_name,

            "Row_Count":
                row_count

        })


validation_df = pd.DataFrame(
    validation_rows
)


print(
    "\nDATABASE ROW COUNTS"
)


print(

    validation_df.to_string(
        index=False
    )

)


# ------------------------------------------------------------
# 16. Save validation result
# ------------------------------------------------------------

validation_df.to_csv(

    OUTPUT_DIR
    / "step12_postgresql_load_validation.csv",

    index=False
)


# ------------------------------------------------------------
# 17. Close connection
# ------------------------------------------------------------

connection.close()


print(
    "\nPostgreSQL connection closed."
)


print(
    "\nStep 12 completed successfully."
)
