import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "01_raw_data"
    / "PS_20174392719_1491204439457_log.csv"
)


# --------------------------------------------------
# 2. Confirm dataset exists
# --------------------------------------------------

print("=" * 60)
print("PROJECT 5 - INITIAL DATA LOADING AND PROFILING")
print("=" * 60)

print("\nDataset path:")
print(DATA_FILE)

print("\nDataset exists:")
print(DATA_FILE.exists())


# --------------------------------------------------
# 3. Load a safe sample
# --------------------------------------------------

sample_df = pd.read_csv(
    DATA_FILE,
    nrows=100000
)


# --------------------------------------------------
# 4. Basic structure
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET STRUCTURE")
print("=" * 60)

print("\nSample shape:")
print(sample_df.shape)

print("\nColumn names:")
print(sample_df.columns.tolist())

print("\nData types:")
print(sample_df.dtypes)

print("\nFirst five rows:")
print(sample_df.head())


# --------------------------------------------------
# 5. Memory usage
# --------------------------------------------------

memory_mb = sample_df.memory_usage(deep=True).sum() / 1024**2

print("\nApproximate sample memory usage:")
print(f"{memory_mb:.2f} MB")


# --------------------------------------------------
# 6. Data quality checks
# --------------------------------------------------

print("\n" + "=" * 60)
print("INITIAL DATA QUALITY CHECKS")
print("=" * 60)

print("\nMissing values:")
print(sample_df.isnull().sum())

print("\nDuplicate rows:")
print(sample_df.duplicated().sum())


# --------------------------------------------------
# 7. Initial category checks
# --------------------------------------------------

print("\nTransaction types:")
print(sample_df["type"].value_counts())

print("\nFraud distribution:")
print(sample_df["isFraud"].value_counts())

print("\nFlagged fraud distribution:")
print(sample_df["isFlaggedFraud"].value_counts())


# --------------------------------------------------
# 8. Count total rows without loading full dataset
# --------------------------------------------------

print("\n" + "=" * 60)
print("FULL DATASET ROW COUNT")
print("=" * 60)

with open(DATA_FILE, "r", encoding="utf-8") as file:
    total_rows = sum(1 for _ in file) - 1

print("\nTotal transaction records:")
print(f"{total_rows:,}")


print("\nStep 2 completed successfully.")