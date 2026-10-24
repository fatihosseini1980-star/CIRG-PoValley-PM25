"""Create the 101,913-row analysis file from the GRINS AQCLIM export."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

COLUMN_MAP = {
    "AirQualityStation": "StationID", "AQ_mean_PM2.5": "PM25",
    "CL_blh": "BLH", "CL_rh": "RH", "CL_t2m": "T2M",
    "CL_tp": "Precipitation", "CL_windspeed": "WindSpeed",
    "CL_winddir": "WindDirection", "CL_ssr": "SolarRadiation",
    "CL_lai_hv": "LAI_HighVegetation", "CL_lai_lv": "LAI_LowVegetation",
    "AirQualityStationType": "StationType", "AirQualityStationArea": "StationArea"
}
DAYS = {2018: 365, 2019: 365, 2020: 366, 2021: 365, 2022: 365}
SCALE_VARS = ["PM25", "BLH", "RH", "T2M", "Precipitation", "WindSpeed",
              "SolarRadiation", "LAI_HighVegetation", "LAI_LowVegetation",
              "Longitude", "Latitude", "Altitude"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = pd.read_csv(args.input).rename(columns=COLUMN_MAP)
    s = raw[raw.Year.between(2018, 2022)].copy()
    s["Observed"] = s.PM25.notna().astype(int)
    coverage = s.groupby(["StationID", "Year"]).Observed.sum().unstack()
    for year, days in DAYS.items(): coverage[year] = coverage[year] / days
    keep = coverage.index[coverage[list(DAYS)].ge(0.80).all(axis=1)]
    df = raw[raw.StationID.isin(keep) & raw.Year.between(2018, 2023) & raw.PM25.notna()].copy()
    df["Date"] = pd.to_datetime(df.Date)
    df = df.sort_values(["Date", "StationID"]).reset_index(drop=True)
    train = df[df.Year.between(2018, 2021)]
    # Exogenous variables remain scaled using the archived (all-site) rule;
    # the unused PM25_Z_Train column is computed only on development responses.
    config_path = Path(__file__).resolve().parents[1] / "config" / "study_design.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Study design must be available: {config_path}")
    design = json.loads(config_path.read_text())
    locked = set(design["locked_stations"])
    dev_train = train.loc[~train.StationID.isin(locked)]
    for var in SCALE_VARS:
        ref = dev_train if var == "PM25" else train
        df[f"{var}_Z_Train"] = (df[var] - ref[var].mean()) / ref[var].std(ddof=1)
    df["MonthSin"] = np.sin(2*np.pi*df.Month/12.0)
    df["MonthCos"] = np.cos(2*np.pi*df.Month/12.0)
    period = np.where(((df.Year % 4 == 0) & ((df.Year % 100 != 0) | (df.Year % 400 == 0))), 366.0, 365.0)
    df["DayOfYearSin"] = np.sin(2*np.pi*df.DayOfYear/period)
    df["DayOfYearCos"] = np.cos(2*np.pi*df.DayOfYear/period)
    df["QCFlag"] = np.where(df.PM25 >= 400, "Review_Extreme_400plus",
                     np.where(df.PM25 >= 200, "Review_High_200plus", ""))
    df["PrimaryAnalysisInclude"] = 1
    df["Date"] = df.Date.dt.strftime("%Y-%m-%d")
    raw_cols = ["StationID","Date","Year","Month","DayOfYear","Season","PM25","BLH","RH","T2M",
                "Precipitation","WindSpeed","WindDirection","SolarRadiation","LAI_HighVegetation",
                "LAI_LowVegetation","Longitude","Latitude","Altitude","StationType","StationArea"]
    z_cols = [f"{v}_Z_Train" for v in SCALE_VARS]
    out_cols = raw_cols + z_cols + ["MonthSin","MonthCos","DayOfYearSin","DayOfYearCos","QCFlag","PrimaryAnalysisInclude"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.output, index=False)
    checks = {"stations": df.StationID.nunique(), "rows": len(df),
              "train": int(df.Year.between(2018, 2021).sum()),
              "validation": int((df.Year == 2022).sum()), "external": int((df.Year == 2023).sum())}
    expected = {"stations": 50, "rows": 101913, "train": 69386, "validation": 16660, "external": 15867}
    if checks != expected: raise RuntimeError(f"Data audit failed: {checks} != {expected}")
    print(checks)

if __name__ == "__main__": main()
