"""Create the 101,913-row analysis file from the GRINS AQCLIM export."""
from __future__ import annotations
import argparse
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
    df = df.sort_values(["StationID", "Date"]).reset_index(drop=True)
    train = df[df.Year.between(2018, 2021)]
    for var in SCALE_VARS:
        df[f"{var}_Z_Train"] = (df[var] - train[var].mean()) / train[var].std(ddof=1)
    df["MonthSin"] = np.sin(2*np.pi*df.Month/12.0)
    df["MonthCos"] = np.cos(2*np.pi*df.Month/12.0)
    period = np.where(((df.Year % 4 == 0) & ((df.Year % 100 != 0) | (df.Year % 400 == 0))), 366.0, 365.0)
    df["DayOfYearSin"] = np.sin(2*np.pi*df.DayOfYear/period)
    df["DayOfYearCos"] = np.cos(2*np.pi*df.DayOfYear/period)
    df["QCFlag"] = np.where(df.PM25 >= 400, "Review_Extreme_400plus",
                     np.where(df.PM25 >= 200, "Review_High_200plus", ""))
    df["PrimaryAnalysisInclude"] = 1
    df["Split"] = np.select([df.Year.between(2018,2021), df.Year.eq(2022), df.Year.eq(2023)], ["Train","Validation","Test"], default="Other")
    df["PM25Observed"] = df.PM25.notna().astype(int)
    df["TimeIndexDays"] = (df.Date - pd.Timestamp("2018-01-01")).dt.days
    cov_summary = coverage.loc[keep, list(DAYS)]
    df["MinStationCoverage_2018_2022"] = df.StationID.map(cov_summary.min(axis=1))
    df["MeanStationCoverage_2018_2022"] = df.StationID.map(cov_summary.mean(axis=1))
    df["Date"] = df.Date.dt.strftime("%Y-%m-%d")
    raw_cols = ["StationID","Date","Year","Month","DayOfYear","Season","PM25","BLH","RH","T2M",
                "Precipitation","WindSpeed","WindDirection","SolarRadiation","LAI_HighVegetation",
                "LAI_LowVegetation","Longitude","Latitude","Altitude","StationType","StationArea"]
    z_cols = [f"{v}_Z_Train" for v in SCALE_VARS]
    out_cols = raw_cols + ["Split","PM25Observed","PrimaryAnalysisInclude","TimeIndexDays","MonthSin","MonthCos","DayOfYearSin","DayOfYearCos","QCFlag","MinStationCoverage_2018_2022","MeanStationCoverage_2018_2022"] + z_cols
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.output, index=False)
    checks = {"stations": df.StationID.nunique(), "rows": len(df),
              "train": int(df.Year.between(2018, 2021).sum()),
              "validation": int((df.Year == 2022).sum()), "external": int((df.Year == 2023).sum())}
    expected = {"stations": 50, "rows": 101913, "train": 69386, "validation": 16660, "external": 15867}
    if checks != expected: raise RuntimeError(f"Data audit failed: {checks} != {expected}")
    print(checks)

if __name__ == "__main__": main()
