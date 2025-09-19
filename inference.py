# inference.py
import pandas as pd
import numpy as np
import json, joblib
from typing import Optional, List, Dict

class RFPowerForecaster:
    def __init__(self, model_path: str = "artifacts/model.joblib", meta_path: str = "artifacts/metadata.json"):
        self.model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.TARGET = meta["target"]
        self.TIME_COL = meta["time_col"]
        self.X_cols   = meta["feature_columns"]
        self.WEATHER_COLS = meta["weather_cols"]

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime, sort, hourly reindex, ffill exogenous only."""
        df = df.copy()
        df[self.TIME_COL] = pd.to_datetime(df[self.TIME_COL], errors="coerce")
        df = df.dropna(subset=[self.TIME_COL]).sort_values(self.TIME_COL)
        df = df.set_index(self.TIME_COL).sort_index()

        # Ensure all weather cols exist
        for c in self.WEATHER_COLS:
            if c not in df.columns:
                df[c] = np.nan

        # Hourly index & ffill exogenous
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="H")
        df = df.reindex(full_idx)
        df.index.name = self.TIME_COL
        exog = [c for c in df.columns if c != self.TARGET]
        df[exog] = df[exog].ffill()

        return df

    def _build_feature_row(self, ts: pd.Timestamp, ref: pd.DataFrame) -> Dict[str, float]:
        row = {}
        row["hour"] = ts.hour
        row["month"] = ts.month
        row["dow"] = ts.weekday()

        # lags for target
        for l in [1,2,3,6]:
            row[f"{self.TARGET}_lag{l}"] = ref[self.TARGET].iloc[-l] if len(ref) >= l else np.nan

        # exogenous + lags
        for col in self.WEATHER_COLS:
            row[col] = ref[col].iloc[-1]
            for l in [1,2,3]:
                row[f"{col}_lag{l}"] = ref[col].iloc[-l] if len(ref) >= l else np.nan

        # rolling stats of target
        hist_target = ref[self.TARGET]
        row[f"{self.TARGET}_roll3"] = hist_target.tail(3).mean() if len(hist_target)>=3 else hist_target.mean()
        row[f"{self.TARGET}_roll6"] = hist_target.tail(6).mean() if len(hist_target)>=6 else hist_target.mean()
        return row

    def forecast_next3(
        self,
        history_df: pd.DataFrame,
        future_weather_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        history_df: must include TARGET and WEATHER_COLS and a 'Date' column.
        future_weather_df: optional DataFrame indexed by the 3 future timestamps with WEATHER_COLS.
        """
        df = self._standardize(history_df)

        # Make sure TARGET column exists (if missing, you can still forecast but accuracy may drop
        # because autoregressive lags/rolls won’t have recent true values).
        if self.TARGET not in df.columns:
            df[self.TARGET] = np.nan

        last_time = df.index.max()
        future_times = pd.date_range(last_time + pd.Timedelta(hours=1), periods=3, freq="H")

        # Ensure future weather has index = timestamps
        if future_weather_df is not None:
            fw = future_weather_df.copy()
            if self.TIME_COL in fw.columns:
                fw[self.TIME_COL] = pd.to_datetime(fw[self.TIME_COL], errors="coerce")
                fw = fw.set_index(self.TIME_COL)
            fw = fw.reindex(future_times)  # keep only those 3 rows
        else:
            fw = None

        ref = df.copy()
        preds = []

        for ts in future_times:
            # append a new row using carry-forward exog
            temp = ref.iloc[[-1]].copy()
            temp.index = [ts]
            if fw is not None:
                for c in self.WEATHER_COLS:
                    if c in fw.columns and pd.notna(fw.loc[ts, c]):
                        temp[c] = fw.loc[ts, c]
            ref = pd.concat([ref, temp])

            feat = self._build_feature_row(ts, ref)
            x = pd.DataFrame([feat], index=[ts])[self.X_cols]
            y_hat = float(self.model.predict(x)[0])

            preds.append((ts, y_hat))
            # write back predicted target to build next-step lags
            ref.loc[ts, self.TARGET] = y_hat

        out = pd.DataFrame(preds, columns=["Timestamp", "Predicted_Power_Output"]).set_index("Timestamp")
        return out
