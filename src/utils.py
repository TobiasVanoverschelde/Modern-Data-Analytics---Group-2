import pandas as pd


class DashboardSummaryBuilder:
    def create_summary(self, daily: pd.DataFrame, results_table: pd.DataFrame) -> pd.DataFrame:
        total_count = daily["daily_count"].sum()
        number_of_sites = daily["site_id"].nunique()
        number_of_municipalities = daily["municipality"].nunique()
        start_date = pd.to_datetime(daily["datetime"]).min()
        end_date = pd.to_datetime(daily["datetime"]).max()
        best_model = results_table.iloc[0]["model"]
        best_r2 = results_table.iloc[0]["R2"]
        best_rmse = results_table.iloc[0]["RMSE"]
        return pd.DataFrame([
            {"metric": "Total counted cyclists", "value": f"{total_count:,.0f}"},
            {"metric": "Counting sites", "value": f"{number_of_sites:,.0f}"},
            {"metric": "Municipalities", "value": f"{number_of_municipalities:,.0f}"},
            {"metric": "Start date", "value": str(start_date.date())},
            {"metric": "End date", "value": str(end_date.date())},
            {"metric": "Best model", "value": best_model},
            {"metric": "Best model R²", "value": f"{best_r2:.3f}"},
            {"metric": "Best model RMSE", "value": f"{best_rmse:,.2f}"}
        ])


def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {path}. Run `python main.py` first to generate processed data.")
