import csv, math, os, warnings, time
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
SEED=20260726
FOLD=int(os.environ.get('FOLD_ONLY','1'))
CSV='/mnt/data/Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
OUT=f'/mnt/data/po_stage2_clim_resgraph_fold{FOLD}'; os.makedirs(OUT,exist_ok=True)
locked=['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[])); held_ids=folds[FOLD]; source_ids=sorted(set(dev)-set(held_ids))
CONT=['BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos']
MET=['BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train']
print('load')
raw=[]
with open(CSV,newline='',encoding='utf-8-sig') as f:
 for d in csv.DictReader(f): raw.append(d)
stations=sorted({d['StationID'] for d in raw}); dates=sorted({d['Date'] for d in raw}); si={s:i for i,s in enumerate(stations)}; di={d:i for i,d in enumerate(dates)}
N=len(stations);T=len(dates); y=np.full((T,N),np.nan,np.float32); cont=np.full((T,N,len(CONT)),np.nan,np.float32); wind=np.zeros((T,N),np.int8); year=np.zeros(T,np.int16); static={}
for d in raw:
 t=di[d['Date']];j=si[d['StationID']];y[t,j]=float(d['PM25']);cont[t,j]=[float(d[c]) for c in CONT];wind[t,j]=int(float(d['WindDirection']));year[t]=int(d['Year'])
 if d['StationID'] not in static: static[d['StationID']]={'lon':float(d['Longitude']),'lat':float(d['Latitude']),'alt':float(d['Altitude']),'type':d['StationType'],'area':d['StationArea']}
assert set(locked).isdisjoint(dev)
source_idx=[si[s] for s in source_ids]; held_idx=[si[s] for s in held_ids]
lons=np.array([static[s]['lon'] for s in stations]);lats=np.array([static[s]['lat'] for s in stations])
def hav(lon1,lat1,lon2,lat2):
 R=6371.0088;a1=np.radians(lat1);a2=np.radians(lat2);dl=np.radians(lon2-lon1);dp=np.radians(lat2-lat1);a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
dist=np.stack([hav(lons[i],lats[i],lons,lats) for i in range(N)]).astype(np.float32)
TYPES=['background','traffic','industrial'];AREAS=['urban','suburban','rural']
# station climatology features from 2018-2021 covariates only
train_t=np.where(np.isin(year,[2018,2019,2020,2021]))[0]
def clim_features(j):
 vals=[]
 # static + polynomial coordinates
 lon=static[stations[j]]['lon']; lat=static[stations[j]]['lat']; alt=static[stations[j]]['alt']
 vals += [lon,lat,alt,lon*lon,lat*lat,lon*lat]
 # mean/sd of meteorology over train period and seasonal means
 for k in range(len(MET)):
  a=cont[train_t,j,k].astype(float); a=a[np.isfinite(a)]
  vals += [float(np.mean(a)),float(np.std(a))]
 # categorical
 vals += [float(static[stations[j]]['type']==k) for k in TYPES]
 vals += [float(static[stations[j]]['area']==k) for k in AREAS]
 return vals
Xclim=np.asarray([clim_features(j) for j in range(N)],np.float64)
train_mean=np.nanmean(y[train_t],axis=0)
# fixed preselected regularized KRR; source-station LOO predictions mimic unseen deployment
alpha=.01; gamma=.01
clim_src={}
for j in source_idx:
 others=[q for q in source_idx if q!=j]
 m=make_pipeline(StandardScaler(),KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma))
 m.fit(Xclim[others],train_mean[others])
 pp=float(m.predict(Xclim[j:j+1])[0])
 pp=float(np.clip(pp,np.nanmin(train_mean[others]),np.nanmax(train_mean[others])))
 clim_src[j]=pp
m_all=make_pipeline(StandardScaler(),KernelRidge(alpha=alpha,kernel='rbf',gamma=gamma))
m_all.fit(Xclim[source_idx],train_mean[source_idx])
clim_held={j:float(np.clip(m_all.predict(Xclim[j:j+1])[0],np.nanmin(train_mean[source_idx]),np.nanmax(train_mean[source_idx]))) for j in held_idx}
clim=np.full(N,np.nan,float)
for j,v in clim_src.items():clim[j]=v
for j,v in clim_held.items():clim[j]=v

def base_feat(t,j):
 return list(cont[t,j].astype(float))+[float(wind[t,j]==k) for k in [1,2,3,4]]+[float(static[stations[j]]['type']==k) for k in TYPES]+[float(static[stations[j]]['area']==k) for k in AREAS]+[float(clim[j])]
base_n=len(base_feat(0,source_idx[0]))
gnames=['net_mean','net_median','net_sd','net_q10','net_q90','nearest','nearest_dist','knn3','knn5','knn10']
for h in [25,50,100,200]:gnames += [f'rbf{h}_mean',f'rbf{h}_sd',f'rbf{h}_effn']
for p in [1,2]:gnames += [f'idw{p}_mean',f'idw{p}_sd']
def gfeat(field,t,j):
 src=np.asarray(source_idx);v=field[t,src].astype(float);ok=np.isfinite(v)&(src!=j);src=src[ok];v=v[ok]
 if len(v)==0:return [np.nan]*len(gnames)
 d=dist[j,src].astype(float);o=np.argsort(d);vv=v[o];dd=d[o];out=[v.mean(),np.median(v),v.std(),np.quantile(v,.1),np.quantile(v,.9),vv[0],dd[0],vv[:min(3,len(vv))].mean(),vv[:min(5,len(vv))].mean(),vv[:min(10,len(vv))].mean()]
 for h in [25,50,100,200]:
  w=np.exp(-.5*(d/h)**2);sw=w.sum();m=(w*v).sum()/sw;sd=np.sqrt((w*(v-m)**2).sum()/sw);eff=sw*sw/(w*w).sum();out += [m,sd,eff]
 for p in [1,2]:
  w=1/np.maximum(d,1)**p;sw=w.sum();m=(w*v).sum()/sw;sd=np.sqrt((w*(v-m)**2).sum()/sw);out += [m,sd]
 return out

def rows(target_idx,years,field=None):
 X=[];yy=[];meta=[];targets=set(target_idx)
 for t in np.where(np.isin(year,list(years)))[0]:
  for j in np.where(np.isfinite(y[t]))[0]:
   if j not in targets:continue
   z=base_feat(t,j)
   if field is not None:z+=gfeat(field,t,j)
   X.append(z);yy.append(float(y[t,j]-clim[j]));meta.append((t,j))
 return np.asarray(X,np.float32),np.asarray(yy,np.float32),meta

def lgb(seed,n=220): return LGBMRegressor(objective='regression_l2',n_estimators=n,learning_rate=.035,num_leaves=31,min_child_samples=100,colsample_bytree=.85,reg_alpha=.1,reg_lambda=2,random_state=seed,n_jobs=4,verbosity=-1)
# OOF baseline residual field (on original y scale) for 2018-2021
R=np.full_like(y,np.nan); base_oof=np.full_like(y,np.nan)
for yy in [2018,2019,2020,2021]:
 Xtr,ytr,mtr=rows(source_idx,set([2018,2019,2020,2021])-{yy}); Xv,yv,mv=rows(source_idx,{yy})
 m=lgb(SEED+yy,180);m.fit(Xtr,ytr);pred_an=m.predict(Xv)
 for k,(t,j) in enumerate(mv):
  pp=clim[j]+pred_an[k];base_oof[t,j]=pp;R[t,j]=y[t,j]-pp
 print('oof',yy,np.sqrt(np.mean((yv-pred_an)**2)))
# final baseline for 2022
Xtr,ytr,mtr=rows(source_idx,{2018,2019,2020,2021});Xcal,ycal_an,mcal=rows(source_idx,{2022});Xte,yte_an,mte=rows(held_idx,{2022})
ycal=np.array([y[t,j] for t,j in mcal],float);yte=np.array([y[t,j] for t,j in mte],float)
base=lgb(SEED+FOLD,350);base.fit(Xtr,ytr,eval_set=[(Xcal,ycal_an)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)])
pcal=np.array([clim[j] for t,j in mcal])+base.predict(Xcal);pte=np.array([clim[j] for t,j in mte])+base.predict(Xte)
for k,(t,j) in enumerate(mcal):R[t,j]=y[t,j]-pcal[k]
# correction training features based on OOF residual field
Xcorr=[];ycorr=[]
for yy in [2018,2019,2020,2021]:
 Xy,yt,my=rows(source_idx,{yy},R)
 for k,(t,j) in enumerate(my):
  Xcorr.append(np.concatenate([Xy[k],[base_oof[t,j]]]))
  ycorr.append(R[t,j])
Xcorr=np.asarray(Xcorr,np.float32);ycorr=np.asarray(ycorr,np.float32)
Xc,_,mc=rows(source_idx,{2022},R); Xt,_,mt=rows(held_idx,{2022},R)
Xc=np.column_stack([Xc,pcal]);Xt=np.column_stack([Xt,pte])
cor=lgb(SEED+100+FOLD,320);cor.fit(Xcorr,ycorr,eval_set=[(Xc,ycal-pcal)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)])
mu_res=pte+cor.predict(Xt)
models={'Climatology-anomaly baseline':pte,'Climatology residual graph':mu_res}
res=[];pred=[]
for name,mu in models.items():
 e=yte-mu;met={'Fold':FOLD,'Model':name,'n':len(yte),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(mu-yte))};res.append(met);print(met)
 for k,(t,j) in enumerate(mte):pred.append([FOLD,name,stations[j],dates[t],float(yte[k]),float(mu[k]),float(clim[j])])
with open(OUT+'/metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(res[0]));w.writeheader();w.writerows(res)
with open(OUT+'/predictions.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['Fold','Model','StationID','Date','Observed','Mean','Climatology']);w.writerows(pred)
with open(OUT+'/climatology.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow(['StationID','Set','PredictedTrainClimatology','TrueTrainMean_if_source'])
 for j in source_idx:w.writerow([stations[j],'source_LOO',clim[j],train_mean[j]])
 for j in held_idx:w.writerow([stations[j],'held',clim[j],''])
print('done')
