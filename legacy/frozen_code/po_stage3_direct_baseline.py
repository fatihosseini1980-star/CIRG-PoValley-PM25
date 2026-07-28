import csv, numpy as np, pandas as pd
from pathlib import Path
CSV='/mnt/data/Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
OUT=Path('/mnt/data/Po_Valley_Final_Test_2023')
locked=['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[]))
use=['StationID','Date','Year','PM25','Longitude','Latitude']
d=pd.read_csv(CSV,usecols=use); stations=sorted(d.StationID.unique()); dates=sorted(d.Date.unique()); si={s:i for i,s in enumerate(stations)};di={x:i for i,x in enumerate(dates)}
y=np.full((len(dates),len(stations)),np.nan)
year=np.zeros(len(dates),int)
meta=d.drop_duplicates('StationID').set_index('StationID')
for r in d.itertuples():y[di[r.Date],si[r.StationID]]=r.PM25;year[di[r.Date]]=r.Year
lon=meta.loc[stations,'Longitude'].values;lat=meta.loc[stations,'Latitude'].values
def hav(lon1,lat1,lon2,lat2):
 R=6371.0088;a1=np.radians(lat1);a2=np.radians(lat2);dl=np.radians(lon2-lon1);dp=np.radians(lat2-lat1);a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
dist=np.stack([hav(lon[i],lat[i],lon,lat) for i in range(len(stations))])
src=np.array([si[s] for s in dev]); tar=np.array([si[s] for s in locked])
cands=['rbf25','rbf50','rbf100','rbf200','idw1','idw2','knn3','knn5','knn10']
def pred_one(t,j,name,exclude_self=True):
 ss=src.copy();v=y[t,ss];ok=np.isfinite(v)
 if exclude_self:ok &= ss!=j
 ss=ss[ok];v=v[ok];dd=dist[j,ss];o=np.argsort(dd);v2=v[o];d2=dd[o]
 if name.startswith('rbf'):
  h=float(name[3:]);w=np.exp(-.5*(dd/h)**2);return np.sum(w*v)/np.sum(w)
 if name.startswith('idw'):
  p=float(name[3:]);w=1/np.maximum(dd,1)**p;return np.sum(w*v)/np.sum(w)
 k=int(name[3:]);return np.mean(v2[:min(k,len(v2))])
# source 2022 validation
rows=[]
for name in cands:
 obs=[];pr=[]
 for t in np.where(year==2022)[0]:
  for j in src:
   if np.isfinite(y[t,j]):obs.append(y[t,j]);pr.append(pred_one(t,j,name,True))
 obs=np.array(obs);pr=np.array(pr);rows.append({'Candidate':name,'n':len(obs),'MAE':np.mean(abs(obs-pr)),'RMSE':np.sqrt(np.mean((obs-pr)**2))})
val=pd.DataFrame(rows).sort_values('MAE');best=val.iloc[0].Candidate
print(val.to_string(index=False));print('BEST',best)
# final 2023
out=[]
for t in np.where(year==2023)[0]:
 for j in tar:
  if np.isfinite(y[t,j]):out.append({'Model':'Source-validated direct kernel','SelectedKernel':best,'StationID':stations[j],'Date':dates[t],'Observed':y[t,j],'Mean':pred_one(t,j,best,False)})
pred=pd.DataFrame(out);e=pred.Mean-pred.Observed
met=pd.DataFrame([{'Model':'Source-validated direct kernel','SelectedKernel':best,'n':len(pred),'RMSE':np.sqrt(np.mean(e*e)),'MAE':np.mean(abs(e)),'Bias':np.mean(e),'R2':1-np.sum(e*e)/np.sum((pred.Observed-pred.Observed.mean())**2)}])
val.to_csv(OUT/'final_direct_kernel_selection_2022.csv',index=False);pred.to_csv(OUT/'final_direct_kernel_predictions.csv',index=False);met.to_csv(OUT/'final_direct_kernel_metrics.csv',index=False)
print(met.to_string(index=False))
