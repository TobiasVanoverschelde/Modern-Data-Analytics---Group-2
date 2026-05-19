import pandas as pd


class DataQualityReporter:

    def generate_report(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ):

        print("\n" + "=" * 80)
        print(f"DATA QUALITY REPORT: {dataset_name}")
        print("=" * 80)

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumn names:")
        print(list(df.columns))

        print("\nMissing values:")
        print(df.isnull().sum())

        duplicates = df.duplicated().sum()

        print(f"\nDuplicate rows: {duplicates:,}")
        print("=" * 80)
