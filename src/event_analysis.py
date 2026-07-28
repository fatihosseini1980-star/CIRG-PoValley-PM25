"""Event probabilities use the calibrated Student-t scale saved in final predictions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

def cls(y,p):
    y=np.asarray(y,bool); p=np.asarray(p,bool); tp=int(np.sum(y&p)); fp=int(np.sum(~y&p)); fn=int(np.sum(y&~p)); tn=int(np.sum(~y&~p))
    rec=tp/(tp+fn) if tp+fn else np.nan; pre=tp/(tp+fp) if tp+fp else np.nan; fpr=fp/(fp+tn) if fp+tn else np.nan
    return {'Recall':rec,'Precision':pre,'FPR':fpr,'F1':2*pre*rec/(pre+rec) if pre+rec else np.nan,'TP':tp,'FP':fp,'FN':fn,'TN':tn}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args(); root=args.root
    final=root/'Po_Valley_Final_Test_2023'; out=root/'Po_Valley_Stage4_Stress'; out.mkdir(exist_ok=True)
    g=pd.read_csv(final/'final_probabilistic_predictions.csv'); r=pd.read_csv(final/'final_direct_kernel_probabilistic_predictions.csv')
    design=json.loads((Path(__file__).resolve().parents[1]/'config/study_design.json').read_text()); source=set(sum(design['folds'].values(),[]))
    d=pd.read_csv(root/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv',usecols=['StationID','Year','PM25'])
    tr=d[d.StationID.isin(source)&d.Year.between(2018,2021)].PM25; thresholds={'P90':float(tr.quantile(.90)),'P95':float(tr.quantile(.95))}
    r=r.set_index(['StationID','Date']); rows=[]; metrics=[]
    for a in g.to_dict('records'):
        b=r.loc[(a['StationID'],a['Date'])]; row={'StationID':a['StationID'],'Date':a['Date'],'Observed':a['Observed'],'GraphMean':a['Mean'],'RBFMean':b['Mean'],'GraphScale':a['Scale'],'RBFScale':b['Scale'],'GraphDF':a['df'],'RBFDF':b['df'],'GraphCRPS':a['CRPS_row'],'RBFCRPS':b['CRPS_row']}
        for lab,thr in thresholds.items():
            row[f'Truth_{lab}']=int(a['Observed']>thr); row[f'GraphProb_{lab}']=1-student_t.cdf((thr-a['Mean'])/a['Scale'],df=a['df']); row[f'RBFProb_{lab}']=1-student_t.cdf((thr-b['Mean'])/b['Scale'],df=b['df']); row[f'GraphClass_{lab}']=int(a['Mean']>thr); row[f'RBFClass_{lab}']=int(b['Mean']>thr)
        rows.append(row)
    rows=pd.DataFrame(rows); rows.to_csv(out/'paired_event_rows.csv',index=False)
    for lab,thr in thresholds.items():
        y=rows[f'Truth_{lab}'].values
        for model in ['Graph','RBF']:
            pr=rows[f'{model}Prob_{lab}'].values; pred=rows[f'{model}Class_{lab}'].values
            metrics.append({'Model':model,'ThresholdLabel':lab,'Threshold':thr,'Events':int(y.sum()),'EventRate':float(y.mean()),'Brier':brier_score_loss(y,pr),'AUROC':roc_auc_score(y,pr),'AUPRC':average_precision_score(y,pr),**cls(y,pred)})
    pd.DataFrame(metrics).to_csv(out/'event_threshold_metrics.csv',index=False); print(pd.DataFrame(metrics))
if __name__=='__main__': main()
