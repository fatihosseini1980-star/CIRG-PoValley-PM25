import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

BASE=Path('/mnt/data')
FINAL=BASE/'Po_Valley_Final_Test_2023'
OUT=BASE/'Po_Valley_Stage4_Stress'
OUT.mkdir(exist_ok=True)
SEED=20260726


def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

g=read_csv(FINAL/'final_probabilistic_predictions.csv')
r=read_csv(FINAL/'final_direct_kernel_probabilistic_predictions.csv')
rd={(x['StationID'],x['Date']):x for x in r}

# Training thresholds from development stations only.
with open(FINAL/'final_model_parameters.json',encoding='utf-8') as f:
    import json
    par=json.load(f)
source=set(par['source_stations'])
train_vals=[]
with open(BASE/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv',encoding='utf-8-sig',newline='') as f:
    for x in csv.DictReader(f):
        if x['StationID'] in source and int(x['Year']) in [2018,2019,2020,2021]:
            train_vals.append(float(x['PM25']))
p90=float(np.quantile(train_vals,.90));p95=float(np.quantile(train_vals,.95))

rows=[]
for a in g:
    b=rd[(a['StationID'],a['Date'])]
    y=float(a['Observed']); mg=float(a['Mean']); mr=float(b['Mean'])
    sg=float(a['Scale']); sr=float(b['Scale']); nug=float(a['df']); nur=float(b['df'])
    row={
        'StationID':a['StationID'],'Date':a['Date'],'Season':a['Season'],'Month':int(a['Month']),
        'StationType':a['StationType'],'StationArea':a['StationArea'],
        'NearestSourceKm':float(a['NearestSourceKm']),'BLH_Z':float(a['BLH_Z_Train']),
        'RH_Z':float(a['RH_Z_Train']),'T2M_Z':float(a['T2M_Z_Train']),'Wind_Z':float(a['WindSpeed_Z_Train']),
        'Observed':y,'GraphMean':mg,'RBFMean':mr,'GraphScale':sg,'RBFScale':sr,
        'GraphCRPS':float(a['CRPS_row']),'RBFCRPS':float(b['CRPS_row']),
        'GraphLogScore':float(a['LogScore_row']),'RBFLogScore':float(b['LogScore_row']),
        'GraphAE':abs(mg-y),'RBFAE':abs(mr-y),'GraphSE':(mg-y)**2,'RBFSE':(mr-y)**2,
        'GraphError':mg-y,'RBFError':mr-y,
        'GraphDF':nug,'RBFDF':nur,
    }
    for label,thr in [('P90',p90),('P95',p95)]:
        row[f'Truth_{label}']=int(y>thr)
        row[f'GraphProb_{label}']=float(1-student_t.cdf((thr-mg)/sg,df=nug))
        row[f'RBFProb_{label}']=float(1-student_t.cdf((thr-mr)/sr,df=nur))
        row[f'GraphClass_{label}']=int(mg>thr)
        row[f'RBFClass_{label}']=int(mr>thr)
    rows.append(row)

# Quantile groups.
def assign_q(field,name):
    vals=np.array([x[field] for x in rows],float); cuts=np.quantile(vals,[0,.25,.5,.75,1])
    for x in rows: x[name]=int(np.searchsorted(cuts[1:-1],x[field],side='right')+1)
    return cuts
cuts_dist=assign_q('NearestSourceKm','DistanceQuartile')
cuts_blh=assign_q('BLH_Z','BLHQuartile')
cuts_obs=assign_q('Observed','PollutionQuartile')

# Generic group metrics.
def summarize_group(rr):
    y=np.array([x['Observed'] for x in rr]); mg=np.array([x['GraphMean'] for x in rr]); mr=np.array([x['RBFMean'] for x in rr])
    return {
        'n':len(rr),'ObservedMean':float(y.mean()),
        'Graph_RMSE':float(np.sqrt(np.mean((mg-y)**2))),'RBF_RMSE':float(np.sqrt(np.mean((mr-y)**2))),
        'Delta_RMSE':float(np.sqrt(np.mean((mg-y)**2))-np.sqrt(np.mean((mr-y)**2))),
        'Graph_MAE':float(np.mean(abs(mg-y))),'RBF_MAE':float(np.mean(abs(mr-y))),
        'Delta_MAE':float(np.mean(abs(mg-y))-np.mean(abs(mr-y))),
        'Graph_CRPS':float(np.mean([x['GraphCRPS'] for x in rr])),'RBF_CRPS':float(np.mean([x['RBFCRPS'] for x in rr])),
        'Delta_CRPS':float(np.mean([x['GraphCRPS']-x['RBFCRPS'] for x in rr])),
        'Graph_Bias':float(np.mean(mg-y)),'RBF_Bias':float(np.mean(mr-y)),
        'GraphWin_RMSE':int(np.sqrt(np.mean((mg-y)**2))<np.sqrt(np.mean((mr-y)**2))),
        'GraphWin_MAE':int(np.mean(abs(mg-y))<np.mean(abs(mr-y))),
        'GraphWin_CRPS':int(np.mean([x['GraphCRPS'] for x in rr])<np.mean([x['RBFCRPS'] for x in rr])),
    }

group_vars=['StationID','Season','StationType','StationArea','DistanceQuartile','BLHQuartile','PollutionQuartile']
group_results=[]
for gv in group_vars:
    groups=defaultdict(list)
    for x in rows: groups[str(x[gv])].append(x)
    for group,rr in sorted(groups.items()):
        z={'GroupVariable':gv,'Group':group};z.update(summarize_group(rr));group_results.append(z)

with open(OUT/'condition_subgroup_results.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(group_results[0].keys()));w.writeheader();w.writerows(group_results)

# Daily-block bootstrap for key subgroups and overall.
rng=np.random.default_rng(SEED+5000)
boot_groups=[('Overall','All',rows)]
for gv in ['Season','StationType','StationArea','BLHQuartile','PollutionQuartile']:
    groups=defaultdict(list)
    for x in rows: groups[str(x[gv])].append(x)
    for group,rr in groups.items(): boot_groups.append((gv,group,rr))
boot_out=[]
for gv,group,rr in boot_groups:
    dates=sorted({x['Date'] for x in rr}); bydate=defaultdict(list)
    for x in rr: bydate[x['Date']].append(x)
    daily={}
    for d in dates:
        z=bydate[d]
        daily[d]={
            'dAE':float(np.mean([x['GraphAE']-x['RBFAE'] for x in z])),
            'dSE':float(np.mean([x['GraphSE']-x['RBFSE'] for x in z])),
            'dCRPS':float(np.mean([x['GraphCRPS']-x['RBFCRPS'] for x in z])),
        }
    for met in ['dAE','dSE','dCRPS']:
        arr=np.array([daily[d][met] for d in dates],float)
        vals=np.empty(2000)
        for b in range(2000): vals[b]=np.mean(rng.choice(arr,size=len(arr),replace=True))
        boot_out.append({'GroupVariable':gv,'Group':group,'Metric':met,'MeanDifference':float(arr.mean()),'CI_Lower':float(np.quantile(vals,.025)),'CI_Upper':float(np.quantile(vals,.975)),'ProbabilityGraphBetter':float(np.mean(vals<0))})
with open(OUT/'condition_subgroup_bootstrap.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(boot_out[0].keys()));w.writeheader();w.writerows(boot_out)

# Event metrics P90/P95, probabilistic and deterministic.
def classification(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool)
    tp=int(np.sum(truth&pred));fp=int(np.sum(~truth&pred));fn=int(np.sum(truth&~pred));tn=int(np.sum(~truth&~pred))
    rec=tp/(tp+fn) if tp+fn else float('nan');prec=tp/(tp+fp) if tp+fp else float('nan');fpr=fp/(fp+tn) if fp+tn else float('nan');f1=2*prec*rec/(prec+rec) if prec+rec else float('nan')
    return {'Recall':rec,'Precision':prec,'FPR':fpr,'F1':f1,'TP':tp,'FP':fp,'FN':fn,'TN':tn}

event_results=[]
for label,thr in [('P90',p90),('P95',p95)]:
    truth=np.array([x[f'Truth_{label}'] for x in rows],int)
    for model in ['Graph','RBF']:
        probs=np.array([x[f'{model}Prob_{label}'] for x in rows],float)
        pred=np.array([x[f'{model}Class_{label}'] for x in rows],int)
        cls=classification(truth,pred)
        high=[x for x in rows if x[f'Truth_{label}']==1]; non=[x for x in rows if x[f'Truth_{label}']==0]
        prefix='Graph' if model=='Graph' else 'RBF'
        event_results.append({
            'Model':model,'ThresholdLabel':label,'Threshold':thr,'n':len(rows),'Events':int(truth.sum()),'EventRate':float(truth.mean()),
            'Brier':float(brier_score_loss(truth,probs)),'AUROC':float(roc_auc_score(truth,probs)),'AUPRC':float(average_precision_score(truth,probs)),
            **cls,
            'High_RMSE':float(np.sqrt(np.mean([x[f'{prefix}SE'] for x in high]))),'High_MAE':float(np.mean([x[f'{prefix}AE'] for x in high])),
            'High_Bias':float(np.mean([x[f'{prefix}Error'] for x in high])),'High_CRPS':float(np.mean([x[f'{prefix}CRPS'] for x in high])),
            'NonHigh_RMSE':float(np.sqrt(np.mean([x[f'{prefix}SE'] for x in non]))),'NonHigh_MAE':float(np.mean([x[f'{prefix}AE'] for x in non])),
            'NonHigh_CRPS':float(np.mean([x[f'{prefix}CRPS'] for x in non])),
        })
with open(OUT/'event_threshold_metrics.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(event_results[0].keys()));w.writeheader();w.writerows(event_results)

# Daily block bootstrap for event differences.
event_boot=[]
for label,thr in [('P90',p90),('P95',p95)]:
    dates=sorted({x['Date'] for x in rows}); bydate=defaultdict(list)
    for x in rows: bydate[x['Date']].append(x)
    vals=[]
    for b in range(2000):
        sampled=rng.choice(dates,size=len(dates),replace=True)
        sample=[x for d in sampled for x in bydate[d]]
        truth=np.array([x[f'Truth_{label}'] for x in sample],int)
        gp=np.array([x[f'GraphProb_{label}'] for x in sample]);rp=np.array([x[f'RBFProb_{label}'] for x in sample])
        gc=np.array([x[f'GraphClass_{label}'] for x in sample]);rc=np.array([x[f'RBFClass_{label}'] for x in sample])
        gm=classification(truth,gc);rm=classification(truth,rc)
        vals.append({
            'Brier':brier_score_loss(truth,gp)-brier_score_loss(truth,rp),
            'AUPRC':average_precision_score(truth,gp)-average_precision_score(truth,rp),
            'Recall':gm['Recall']-rm['Recall'],'F1':gm['F1']-rm['F1'],'FPR':gm['FPR']-rm['FPR'],
        })
    # observed differences
    truth=np.array([x[f'Truth_{label}'] for x in rows],int);gp=np.array([x[f'GraphProb_{label}'] for x in rows]);rp=np.array([x[f'RBFProb_{label}'] for x in rows]);gc=np.array([x[f'GraphClass_{label}'] for x in rows]);rc=np.array([x[f'RBFClass_{label}'] for x in rows]);gm=classification(truth,gc);rm=classification(truth,rc)
    observed={'Brier':brier_score_loss(truth,gp)-brier_score_loss(truth,rp),'AUPRC':average_precision_score(truth,gp)-average_precision_score(truth,rp),'Recall':gm['Recall']-rm['Recall'],'F1':gm['F1']-rm['F1'],'FPR':gm['FPR']-rm['FPR']}
    for met in observed:
        a=np.array([v[met] for v in vals],float)
        event_boot.append({'ThresholdLabel':label,'Threshold':thr,'Metric':met,'GraphMinusRBF':float(observed[met]),'CI_Lower':float(np.quantile(a,.025)),'CI_Upper':float(np.quantile(a,.975)),'ProbabilityGraphBetter':float(np.mean(a<0)) if met in ['Brier','FPR'] else float(np.mean(a>0))})
with open(OUT/'event_threshold_bootstrap.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(event_boot[0].keys()));w.writeheader();w.writerows(event_boot)

# Save row-wise paired analysis for reproducibility.
fields=list(rows[0].keys())
with open(OUT/'paired_condition_rows.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

with open(OUT/'analysis_cutpoints.txt','w',encoding='utf-8') as f:
    f.write(f'Training P90 threshold: {p90}\nTraining P95 threshold: {p95}\n')
    f.write(f'Distance quartile cuts: {cuts_dist.tolist()}\nBLH quartile cuts: {cuts_blh.tolist()}\nObserved quartile cuts: {cuts_obs.tolist()}\n')
print('Saved Stage 4 condition and event analysis to',OUT)
