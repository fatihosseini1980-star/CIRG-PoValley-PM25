import sys, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import simulation as sim
import os
OUT=Path(os.environ.get('CIRG_SIM_OUT', Path(__file__).resolve().parents[1]/'results'/'simulation'))
rows=[]
for source_n in [40,10]:
    for rep in range(int(os.environ.get('CIRG_SIM_SHIFT_REPS','50'))):
        rows += sim.run_one(rep,True,source_n,True)
shift=pd.DataFrame(rows)
shift.to_csv(OUT/'simulation_shift_replicates.csv',index=False)
base=pd.read_csv(OUT/'simulation_replicates.csv')
all_df=pd.concat([base,shift],ignore_index=True)
all_df.to_csv(OUT/'simulation_all_replicates.csv',index=False)
summary=(all_df.groupby(['Adaptive','SourceN','ScaleShift','Model','Calibration'])
         .agg(RMSE=('RMSE','mean'),RMSE_SD=('RMSE','std'),MAE=('MAE','mean'),MAE_SD=('MAE','std'),CRPS=('CRPS','mean'),CRPS_SD=('CRPS','std'),Cov90=('Cov90','mean'),Cov90_SD=('Cov90','std'),Width90=('Width90','mean'),Width90_SD=('Width90','std'),az=('az','mean'),ax=('ax','mean')).reset_index())
summary.to_csv(OUT/'simulation_all_summary.csv',index=False)
paired=[]
for keys,g in all_df.groupby(['Adaptive','SourceN','ScaleShift','Calibration']):
    w=g.pivot(index='Rep',columns='Model',values=['RMSE','MAE','CRPS','Cov90','Width90'])
    for metric in ['RMSE','MAE','CRPS','Cov90','Width90']:
        d=w[metric]['Graph']-w[metric]['RBF']
        paired.append({'Adaptive':keys[0],'SourceN':keys[1],'ScaleShift':keys[2],'Calibration':keys[3],'Metric':metric,'MeanDiff':d.mean(),'Q025':d.quantile(.025),'Q975':d.quantile(.975),'PGraphLower':float(np.mean(d<0))})
pd.DataFrame(paired).to_csv(OUT/'simulation_all_paired.csv',index=False)
print(summary[summary.ScaleShift==True].to_string(index=False))
