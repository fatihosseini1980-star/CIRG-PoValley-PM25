"""Paired day-cluster and 7/14-day moving-block bootstrap sensitivity."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

def bootstrap_means(values, block, reps, rng):
    """Vectorized moving-block bootstrap of an ordered daily-mean series."""
    values=np.asarray(values,float); n=len(values)
    if block==1:
        idx=rng.integers(0,n,size=(reps,n))
    else:
        nb=int(np.ceil(n/block))
        starts=rng.integers(0,n-block+1,size=(reps,nb))
        idx=(starts[:,:,None]+np.arange(block)[None,None,:]).reshape(reps,-1)[:,:n]
    return values[idx].mean(axis=1)

def class_metrics(y,p):
    y=np.asarray(y,bool); p=np.asarray(p,bool); tp=np.sum(y&p); fp=np.sum(~y&p); fn=np.sum(y&~p)
    rec=tp/(tp+fn) if tp+fn else np.nan; pre=tp/(tp+fp) if tp+fp else np.nan
    return rec,2*pre*rec/(pre+rec) if pre+rec else np.nan

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--reps',type=int,default=5000); args=ap.parse_args(); root=args.root
    out=root/'Po_Valley_Sensitivity'; out.mkdir(exist_ok=True); rng=np.random.default_rng(20260726+7000); records=[]
    # Internal learned edge minus fixed graph.
    fixed=[]; learned=[]
    for f in range(1,6):
        a=pd.read_csv(root/f'po_stage2_clim_resgraph_fold{f}'/'predictions.csv'); fixed.append(a[a.Model=='Climatology residual graph'])
        b=pd.read_csv(root/f'po_stage2_learned_edge_fold{f}'/'predictions.csv'); learned.append(b[b.Model=='Climatology learned-edge graph'])
    a=pd.concat(learned).rename(columns={'Mean':'New'}); b=pd.concat(fixed)[['Fold','StationID','Date','Mean']].rename(columns={'Mean':'Ref'}); internal=a.merge(b,on=['Fold','StationID','Date']); internal.Date=pd.to_datetime(internal.Date)
    internal['AE']=abs(internal.New-internal.Observed)-abs(internal.Ref-internal.Observed); internal['SE']=(internal.New-internal.Observed)**2-(internal.Ref-internal.Observed)**2
    # External graph minus RBF.
    g=pd.read_csv(root/'Po_Valley_Final_Test_2023/final_probabilistic_predictions.csv').rename(columns={'Mean':'New','CRPS_row':'NewCRPS'}); r=pd.read_csv(root/'Po_Valley_Final_Test_2023/final_direct_kernel_probabilistic_predictions.csv')[['StationID','Date','Mean','CRPS_row']].rename(columns={'Mean':'Ref','CRPS_row':'RefCRPS'}); external=g.merge(r,on=['StationID','Date']); external.Date=pd.to_datetime(external.Date)
    external['AE']=abs(external.New-external.Observed)-abs(external.Ref-external.Observed); external['SE']=(external.New-external.Observed)**2-(external.Ref-external.Observed)**2; external['CRPS']=external.NewCRPS-external.RefCRPS
    for label,df,mets in [('Internal learned-minus-fixed',internal,['AE','SE']),('External CIRG-minus-RBF25',external,['AE','SE','CRPS'])]:
        by={d:g for d,g in df.groupby('Date')}; dates=list(by)
        for block in [1,7,14]:
            for met in mets:
                daily=np.asarray([by[d][met].mean() for d in dates],float)
                vals=bootstrap_means(daily,block,args.reps,rng)
                records.append({'Comparison':label,'BlockDays':block,'Metric':met,'ObservedDifference':float(df[met].mean()),'CI_Lower':float(np.quantile(vals,.025)),'CI_Upper':float(np.quantile(vals,.975)),'ProbabilityNewBetter':float(np.mean(vals<0))})
    pd.DataFrame(records).to_csv(out/'moving_block_bootstrap.csv',index=False); print(pd.DataFrame(records))
if __name__=='__main__': main()
