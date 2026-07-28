"""Prepare and summarize the two prespecified 203 µg/m³ sensitivity analyses."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def metrics(d,mean='Mean'):
    e=d[mean]-d.Observed
    return {'n':len(d),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(abs(e))),'Bias':float(np.mean(e)),'CRPS':float(d.CRPS_row.mean()) if 'CRPS_row' in d else np.nan}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args(); root=args.root
    data=root/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv'; out=root/'Po_Valley_Sensitivity'; out.mkdir(exist_ok=True)
    d=pd.read_csv(data); mask=(d.StationID=='STA.IT1222A')&(d.Date=='2019-01-06')&(d.PM25==203)
    refit=d.loc[~mask]; refit.to_csv(out/'Po_Valley_without_2019_STA.IT1222A_203.csv',index=False)
    g=pd.read_csv(root/'Po_Valley_Final_Test_2023/final_probabilistic_predictions.csv'); r=pd.read_csv(root/'Po_Valley_Final_Test_2023/final_direct_kernel_probabilistic_predictions.csv')
    rows=[]
    for name,x in [('CIRG',g),('RBF25',r)]:
        z={'Model':name,'Scenario':'Primary',**metrics(x)}; rows.append(z)
        keep=~((x.StationID=='STA.IT1590A')&(x.Date=='2023-01-03')&(x.Observed==203)); rows.append({'Model':name,'Scenario':'Exclude locked-test 203',**metrics(x.loc[keep])})
    pd.DataFrame(rows).to_csv(out/'sensitivity_2023_203_metrics.csv',index=False)
    (out/'REFIT_COMMAND.txt').write_text('python scripts/run_full_real_data.py --data Po_Valley_Sensitivity/Po_Valley_without_2019_STA.IT1222A_203.csv --workspace work_no_2019_203\n',encoding='utf-8')
    print(pd.DataFrame(rows)); print('A full refit is required for the 2019 observation; the variant data file and command were created.')
if __name__=='__main__': main()
