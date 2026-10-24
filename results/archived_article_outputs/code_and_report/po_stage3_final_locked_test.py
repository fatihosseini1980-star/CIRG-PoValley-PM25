import csv, math, os, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from scipy.optimize import minimize
from scipy.stats import t as student_t

warnings.filterwarnings('ignore')
SEED = 20260726
CSV = '/mnt/data/Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
STAGE2 = '/mnt/data/Po_Valley_Stage2'
OUT = Path('/mnt/data/Po_Valley_Final_Test_2023')
OUT.mkdir(exist_ok=True)

locked = ['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']
folds = {
1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],
2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],
3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],
4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],
5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev = sorted(sum(folds.values(), []))
assert set(dev).isdisjoint(locked) and len(dev) == 40 and len(locked) == 10

CONT = ['BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos']
TYPES = ['background','traffic','industrial']
AREAS = ['urban','suburban','rural']

print('Loading data...', flush=True)
raw = []
with open(CSV, newline='', encoding='utf-8-sig') as f:
    for d in csv.DictReader(f):
        raw.append(d)
stations = sorted({d['StationID'] for d in raw})
dates = sorted({d['Date'] for d in raw})
si = {s:i for i,s in enumerate(stations)}
di = {d:i for i,d in enumerate(dates)}
N, T = len(stations), len(dates)
y = np.full((T,N), np.nan, np.float32)
cont = np.full((T,N,len(CONT)), np.nan, np.float32)
wind = np.zeros((T,N), np.int8)
year = np.zeros(T, np.int16)
month = np.zeros(T, np.int8)
season = np.empty(T, dtype=object)
static = {}
for d in raw:
    t = di[d['Date']]; j = si[d['StationID']]
    y[t,j] = float(d['PM25'])
    cont[t,j] = [float(d[c]) for c in CONT]
    wind[t,j] = int(float(d['WindDirection']))
    year[t] = int(d['Year']); month[t] = int(d['Month']); season[t] = d['Season']
    if d['StationID'] not in static:
        static[d['StationID']] = {'lon':float(d['Longitude']),'lat':float(d['Latitude']),'alt':float(d['Altitude']),'type':d['StationType'],'area':d['StationArea']}
source_idx = [si[s] for s in dev]
target_idx = [si[s] for s in locked]

lons = np.array([static[s]['lon'] for s in stations])
lats = np.array([static[s]['lat'] for s in stations])
alts = np.array([static[s]['alt'] for s in stations])
def hav(lon1,lat1,lon2,lat2):
    R=6371.0088; a1=np.radians(lat1); a2=np.radians(lat2)
    dl=np.radians(lon2-lon1); dp=np.radians(lat2-lat1)
    a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
dist = np.stack([hav(lons[i],lats[i],lons,lats) for i in range(N)]).astype(np.float32)

# 1. Inductive station climatology
train_t = np.where(np.isin(year,[2018,2019,2020,2021]))[0]
def clim_features(j):
    lon=static[stations[j]]['lon']; lat=static[stations[j]]['lat']; alt=static[stations[j]]['alt']
    vals=[lon,lat,alt,lon*lon,lat*lat,lon*lat]
    for k in range(8):
        a=cont[train_t,j,k].astype(float); a=a[np.isfinite(a)]
        vals += [float(np.mean(a)),float(np.std(a))]
    vals += [float(static[stations[j]]['type']==k) for k in TYPES]
    vals += [float(static[stations[j]]['area']==k) for k in AREAS]
    return vals
Xclim = np.asarray([clim_features(j) for j in range(N)], float)
train_mean = np.nanmean(y[train_t], axis=0)
clim = np.full(N, np.nan, float)
for j in source_idx:
    others=[q for q in source_idx if q != j]
    m=make_pipeline(StandardScaler(),KernelRidge(alpha=.01,kernel='rbf',gamma=.01))
    m.fit(Xclim[others],train_mean[others])
    clim[j]=np.clip(m.predict(Xclim[j:j+1])[0],np.nanmin(train_mean[others]),np.nanmax(train_mean[others]))
m_all=make_pipeline(StandardScaler(),KernelRidge(alpha=.01,kernel='rbf',gamma=.01))
m_all.fit(Xclim[source_idx],train_mean[source_idx])
for j in target_idx:
    clim[j]=np.clip(m_all.predict(Xclim[j:j+1])[0],np.nanmin(train_mean[source_idx]),np.nanmax(train_mean[source_idx]))

# 2. Daily anomaly baseline
def base_feat(t,j):
    return list(cont[t,j].astype(float)) + [float(wind[t,j]==k) for k in [1,2,3,4]] + [float(static[stations[j]]['type']==k) for k in TYPES] + [float(static[stations[j]]['area']==k) for k in AREAS] + [float(clim[j])]

def rows_base(targets, years):
    X=[]; yy=[]; meta=[]; targets=set(targets)
    for t in np.where(np.isin(year,list(years)))[0]:
        for j in np.where(np.isfinite(y[t]))[0]:
            if j not in targets: continue
            X.append(base_feat(t,j)); yy.append(float(y[t,j]-clim[j])); meta.append((t,j))
    return np.asarray(X,np.float32),np.asarray(yy,np.float32),meta

def lgb(seed,n=220):
    return LGBMRegressor(objective='regression_l2',n_estimators=n,learning_rate=.035,num_leaves=31,min_child_samples=100,colsample_bytree=.85,reg_alpha=.1,reg_lambda=2,random_state=seed,n_jobs=4,verbosity=-1)

R = np.full_like(y, np.nan)
base_oof = np.full_like(y, np.nan)
for yy in [2018,2019,2020,2021]:
    Xtr,ytr,_ = rows_base(source_idx,set([2018,2019,2020,2021])-{yy})
    Xv,yv,mv = rows_base(source_idx,{yy})
    mod=lgb(SEED+yy,180); mod.fit(Xtr,ytr); pa=mod.predict(Xv)
    for k,(t,j) in enumerate(mv):
        pp=clim[j]+pa[k]; base_oof[t,j]=pp; R[t,j]=y[t,j]-pp
    print('OOF year',yy,'RMSE anomaly',float(np.sqrt(np.mean((yv-pa)**2))), flush=True)

Xtr,ytr,_=rows_base(source_idx,{2018,2019,2020,2021})
Xcal,ycal_an,mcal=rows_base(source_idx,{2022})
Xs23,ys23_an,ms23=rows_base(source_idx,{2023})
Xt23,yt23_an,mt23=rows_base(target_idx,{2023})
ycal=np.array([y[t,j] for t,j in mcal],float)
ys23=np.array([y[t,j] for t,j in ms23],float)
yt23=np.array([y[t,j] for t,j in mt23],float)
base=lgb(SEED,350)
base.fit(Xtr,ytr,eval_set=[(Xcal,ycal_an)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)])
pcal=np.array([clim[j] for t,j in mcal])+base.predict(Xcal)
ps23=np.array([clim[j] for t,j in ms23])+base.predict(Xs23)
pt23=np.array([clim[j] for t,j in mt23])+base.predict(Xt23)
for k,(t,j) in enumerate(mcal): R[t,j]=y[t,j]-pcal[k]
for k,(t,j) in enumerate(ms23): R[t,j]=y[t,j]-ps23[k]
base2022=np.full_like(y,np.nan); base2023=np.full_like(y,np.nan)
for k,(t,j) in enumerate(mcal): base2022[t,j]=pcal[k]
for k,(t,j) in enumerate(ms23): base2023[t,j]=ps23[k]
for k,(t,j) in enumerate(mt23): base2023[t,j]=pt23[k]
print('Base best iteration',base.best_iteration_, flush=True)

# 3. Learned edge function
pair_names=['log_distance','altitude_diff','climatology_diff','same_type','same_area','BLH_diff','RH_diff','T2M_diff','Wind_diff','base_prediction_diff']
static_pair=np.zeros((N,N,5),np.float32)
for i in range(N):
    for j in range(N):
        static_pair[i,j]=[
            np.log1p(dist[i,j]/25.0),abs(alts[i]-alts[j])/300.0,abs(clim[i]-clim[j])/10.0,
            float(static[stations[i]]['type']==static[stations[j]]['type']),float(static[stations[i]]['area']==static[stations[j]]['area'])]

def pair_tensor(meta, field, basefield):
    M=len(meta); K=len(source_idx); F=10
    pf=np.zeros((M,K,F),np.float32); rv=np.zeros((M,K),np.float32); mask=np.zeros((M,K),bool); target=np.zeros(M,np.float32)
    for a,(t,j) in enumerate(meta):
        target[a]=field[t,j]
        for b,q in enumerate(source_idx):
            if q==j or not np.isfinite(field[t,q]): continue
            mask[a,b]=True; rv[a,b]=field[t,q]; pf[a,b,:5]=static_pair[j,q]
            pf[a,b,5]=abs(cont[t,j,0]-cont[t,q,0]); pf[a,b,6]=abs(cont[t,j,1]-cont[t,q,1]); pf[a,b,7]=abs(cont[t,j,2]-cont[t,q,2]); pf[a,b,8]=abs(cont[t,j,4]-cont[t,q,4]); pf[a,b,9]=abs(basefield[t,j]-basefield[t,q])/10.0
    return pf,rv,mask,target

meta_all=[]
for t in np.where(np.isin(year,[2018,2019,2020,2021]))[0]:
    for j in source_idx:
        if np.isfinite(R[t,j]) and np.sum(np.isfinite(R[t,source_idx]))>=5: meta_all.append((t,j))
meta_tr=[m for m in meta_all if year[m[0]]<=2020]
meta_va=[m for m in meta_all if year[m[0]]==2021]
rng=np.random.default_rng(SEED)
if len(meta_tr)>28000: meta_tr=[meta_tr[i] for i in rng.choice(len(meta_tr),28000,replace=False)]
if len(meta_va)>9000: meta_va=[meta_va[i] for i in rng.choice(len(meta_va),9000,replace=False)]
ptr,rtr,mtr_att,ytr_att=pair_tensor(meta_tr,R,base_oof)
pva,rva,mva,yva_att=pair_tensor(meta_va,R,base_oof)

class Edge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_pen=torch.nn.Parameter(torch.tensor([0.8,0.2,0.8,0.1,0.1,0.1,0.1,0.1],dtype=torch.float32))
        self.bonus=torch.nn.Parameter(torch.tensor([0.2,0.2]))
        self.raw_shrink=torch.nn.Parameter(torch.tensor(0.0))
    def coeff(self):
        p=-torch.nn.functional.softplus(self.raw_pen)
        return torch.stack([p[0],p[1],p[2],self.bonus[0],self.bonus[1],p[3],p[4],p[5],p[6],p[7]])
    def forward(self,pf,rv,mask):
        sc=(pf*self.coeff()).sum(-1)
        sc=torch.where(mask,sc,torch.tensor(-1e9,device=sc.device))
        w=torch.softmax(sc,dim=1); mu=(w*rv).sum(1); sh=2*torch.sigmoid(self.raw_shrink)
        return sh*mu,w

def fit_edge():
    torch.manual_seed(SEED); mod=Edge(); opt=torch.optim.Adam(mod.parameters(),lr=.03,weight_decay=1e-3); lossfn=torch.nn.SmoothL1Loss(beta=5.0)
    trds=torch.utils.data.TensorDataset(torch.from_numpy(ptr),torch.from_numpy(rtr),torch.from_numpy(mtr_att),torch.from_numpy(ytr_att))
    loader=torch.utils.data.DataLoader(trds,batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(SEED))
    pva_t=torch.from_numpy(pva); rva_t=torch.from_numpy(rva); mva_t=torch.from_numpy(mva); yva_t=torch.from_numpy(yva_att)
    best=None; bad=0
    for ep in range(160):
        mod.train()
        for a,b,c,d in loader:
            opt.zero_grad(); pred,w=mod(a,b,c); loss=lossfn(pred,d)+.03*(w*w).sum(1).mean(); loss.backward(); opt.step()
        mod.eval()
        with torch.no_grad():
            pv,w=mod(pva_t,rva_t,mva_t); vl=float(lossfn(pv,yva_t)+.03*(w*w).sum(1).mean())
        if best is None or vl<best[0]-1e-5:
            best=(vl,{k:v.detach().clone() for k,v in mod.state_dict().items()},ep); bad=0
        else: bad+=1
        if bad>=20: break
    mod.load_state_dict(best[1]); return mod,best
edge,best=fit_edge()
coef=edge.coeff().detach().numpy(); shrink=float(2*torch.sigmoid(edge.raw_shrink).detach())
print('Edge best epoch',best[2],'shrink',shrink,'coefficients',dict(zip(pair_names,map(float,coef))), flush=True)

learn_names=['att_mean','att_sd','att_effn','att_top_weight','att_weighted_distance','att_weighted_climdiff']

def corr_rows_learned(targets, years):
    X=[]; yy=[]; meta=[]
    tids=np.where(np.isin(year,list(years)))[0]
    src=np.asarray(source_idx,dtype=int); coefv=coef.astype(np.float32); targets=np.asarray(list(targets),dtype=int)
    for t in tids:
        tg=targets[np.isfinite(y[t,targets])]
        if len(tg)==0: continue
        v=R[t,src].astype(np.float32); valid=np.isfinite(v)
        if valid.sum()==0: continue
        bf=base_oof if year[t]<=2021 else (base2022 if year[t]==2022 else base2023)
        pf=np.zeros((len(tg),len(src),10),np.float32)
        pf[:,:,:5]=static_pair[tg[:,None],src[None,:],:]
        pf[:,:,5]=np.abs(cont[t,tg,0][:,None]-cont[t,src,0][None,:])
        pf[:,:,6]=np.abs(cont[t,tg,1][:,None]-cont[t,src,1][None,:])
        pf[:,:,7]=np.abs(cont[t,tg,2][:,None]-cont[t,src,2][None,:])
        pf[:,:,8]=np.abs(cont[t,tg,4][:,None]-cont[t,src,4][None,:])
        pf[:,:,9]=np.abs(bf[t,tg][:,None]-bf[t,src][None,:])/10.0
        sc=np.tensordot(pf,coefv,axes=([2],[0]))
        mask=np.broadcast_to(valid[None,:],sc.shape).copy(); mask &= (tg[:,None]!=src[None,:])
        sc=np.where(mask,sc,-1e9); sc-=np.max(sc,axis=1,keepdims=True)
        w=np.exp(sc); w=np.where(mask,w,0.0); sw=w.sum(axis=1,keepdims=True); w=np.divide(w,sw,out=np.zeros_like(w),where=sw>0)
        vv=np.nan_to_num(v,nan=0.0)
        rawmu=(w*vv[None,:]).sum(axis=1); mu=shrink*rawmu
        sd=np.sqrt((w*(vv[None,:]-rawmu[:,None])**2).sum(axis=1))
        eff=1/np.maximum((w*w).sum(axis=1),1e-8); top=w.max(axis=1)
        wdist=(w*dist[tg[:,None],src[None,:]]).sum(axis=1)
        wclim=(w*np.abs(clim[tg[:,None]]-clim[src[None,:]])).sum(axis=1)
        for a,j in enumerate(tg):
            z=base_feat(t,int(j))+[float(mu[a]),float(sd[a]),float(eff[a]),float(top[a]),float(wdist[a]),float(wclim[a]),float(bf[t,j])]
            X.append(z); yy.append(float(R[t,j])); meta.append((t,int(j)))
    return np.asarray(X,np.float32),np.asarray(yy,np.float32),meta

models={'Climatology-anomaly baseline':pt23}
Xco=[]; yco=[]
for yy in [2018,2019,2020,2021]:
    Xa,ya,_=corr_rows_learned(source_idx,{yy}); Xco.append(Xa); yco.append(ya)
Xco=np.vstack(Xco); yco=np.concatenate(yco)
Xc,yc,_=corr_rows_learned(source_idx,{2022})
Xt,_,_=corr_rows_learned(target_idx,{2023})
cor=lgb(SEED+1010,350)
cor.fit(Xco,yco,eval_set=[(Xc,yc)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)])
models['Inductive climatology + learned edge']=pt23+cor.predict(Xt)
print('Inductive climatology + learned edge best iteration',cor.best_iteration_, flush=True)

def direct_spatial(meta, mode):
    out=[]; src=np.asarray(source_idx)
    for t,j in meta:
        v=y[t,src].astype(float); ok=np.isfinite(v); ss=src[ok]; vv=v[ok]
        d=dist[j,ss].astype(float); o=np.argsort(d)
        if mode=='nearest': out.append(vv[o[0]])
        else:
            w=1/np.maximum(d,1.0)**2; out.append(float(np.sum(w*vv)/np.sum(w)))
    return np.asarray(out,float)
models['Nearest observed source']=direct_spatial(mt23,'nearest')
models['IDW-2 observed sources']=direct_spatial(mt23,'idw')

# 4. Final point evaluation
records=[]; metrics=[]
for name,mu in models.items():
    e=yt23-mu
    metrics.append({'Model':name,'n':len(yt23),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(mu-yt23)),'R2':float(1-np.sum(e*e)/np.sum((yt23-yt23.mean())**2))})
    for k,(t,j) in enumerate(mt23):
        records.append({'Model':name,'StationID':stations[j],'Date':dates[t],'Observed':float(yt23[k]),'Mean':float(mu[k]),'Climatology':float(clim[j]),'Season':str(season[t]),'Month':int(month[t]),'NearestSourceKm':float(np.min(dist[j,source_idx])),'StationType':static[stations[j]]['type'],'StationArea':static[stations[j]]['area'],'Longitude':static[stations[j]]['lon'],'Latitude':static[stations[j]]['lat'],'Altitude':static[stations[j]]['alt']})
pred_long=pd.DataFrame(records)
point_metrics=pd.DataFrame(metrics).sort_values('RMSE')
point_metrics.to_csv(OUT/'final_point_metrics.csv',index=False)
pred_long.to_csv(OUT/'final_point_predictions_long.csv',index=False)

sel='Inductive climatology + learned edge'
boots=[]
for ref in [x for x in models if x != sel]:
    a=pred_long[pred_long.Model==sel][['StationID','Date','Observed','Mean']].rename(columns={'Mean':'new'})
    b=pred_long[pred_long.Model==ref][['StationID','Date','Mean']].rename(columns={'Mean':'ref'})
    d=a.merge(b,on=['StationID','Date']); d['Date']=pd.to_datetime(d.Date)
    for met in ['AE','SE']:
        d['diff']=np.abs(d.new-d.Observed)-np.abs(d.ref-d.Observed) if met=='AE' else (d.new-d.Observed)**2-(d.ref-d.Observed)**2
        daily=d.groupby('Date')['diff'].mean().values
        rng=np.random.default_rng(SEED+len(boots)); vals=np.array([rng.choice(daily,len(daily),replace=True).mean() for _ in range(5000)])
        boots.append({'Comparison':sel+' minus '+ref,'Metric':met,'MeanDifference':float(daily.mean()),'CI_Lower':float(np.quantile(vals,.025)),'CI_Upper':float(np.quantile(vals,.975)),'ProbabilitySelectedBetter':float(np.mean(vals<0))})
pd.DataFrame(boots).to_csv(OUT/'final_point_bootstrap.csv',index=False)

# 5. Frozen probabilistic layer trained only on internal 2022 CV predictions
internal=pd.read_csv(f'{STAGE2}/stage2_learned_mean_predictions.csv')
internal['Date']=pd.to_datetime(internal['Date'])
cols=['StationID','Date','Year','Month','Season','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','WindDirection','StationType','StationArea']
data=pd.read_csv(CSV,usecols=cols); data['Date']=pd.to_datetime(data['Date'])
internal=internal.merge(data,on=['StationID','Date'],how='left',validate='one_to_one')
rawcoords=pd.read_csv(CSV,usecols=['StationID','Longitude','Latitude']).drop_duplicates('StationID').set_index('StationID')
nearest_internal={}
for f,held in folds.items():
    src=sorted(set(dev)-set(held)); slon=rawcoords.loc[src,'Longitude'].values; slat=rawcoords.loc[src,'Latitude'].values
    for s in held: nearest_internal[(f,s)]=float(np.min(hav(rawcoords.loc[s,'Longitude'],rawcoords.loc[s,'Latitude'],slon,slat)))
internal['NearestSourceKm']=[nearest_internal[(int(f),s)] for f,s in zip(internal.Fold,internal.StationID)]
internal['Residual']=internal['Observed']-internal['Mean']; internal['LogAbsResidual']=np.log(np.abs(internal.Residual)+0.5)

selected=pred_long[pred_long.Model==sel].copy(); selected['Date']=pd.to_datetime(selected.Date)
final_cov=data[data.Year==2023].copy()
selected=selected.merge(final_cov,on=['StationID','Date'],how='left',validate='one_to_one',suffixes=('','_cov'))
selected['NearestSourceKm']=[float(np.min(dist[si[s],source_idx])) for s in selected.StationID]

num=['Mean','Climatology','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','NearestSourceKm']
cat=['WindDirection','StationType','StationArea']
Xall=pd.concat([internal[num+cat],selected[num+cat]],axis=0,ignore_index=True)
Xall=pd.get_dummies(Xall,columns=cat,drop_first=False,dtype=float).replace([np.inf,-np.inf],np.nan).fillna(0.0)
Xint=Xall.iloc[:len(internal)].copy(); Xfin=Xall.iloc[len(internal):].copy()
scale_model=LGBMRegressor(objective='regression_l1',n_estimators=180,learning_rate=.025,num_leaves=15,min_child_samples=120,colsample_bytree=.85,reg_alpha=.2,reg_lambda=3,random_state=SEED,n_jobs=4,verbosity=-1)
scale_model.fit(Xint,internal.LogAbsResidual)
s0int=np.exp(scale_model.predict(Xint)).clip(.2,100); s0fin=np.exp(scale_model.predict(Xfin)).clip(.2,100)

def fit_t(resid,s0):
    resid=np.asarray(resid,float); s0=np.clip(np.asarray(s0,float),.2,100)
    def obj(par):
        mult=np.exp(par[0]); nu=2.05+np.exp(par[1]); sc=mult*s0
        return -np.mean(student_t.logpdf(resid/sc,df=nu)-np.log(sc))
    opt=minimize(obj,[0.0,math.log(5.0)],method='Nelder-Mead',options={'maxiter':400})
    return float(np.exp(opt.x[0])),float(2.05+np.exp(opt.x[1]))
mult,nu=fit_t(internal.Residual.values,s0int)
base_int=mult*s0int; base_fin=mult*s0fin; qnom=student_t.ppf(.95,nu)
scores=np.abs(internal.Residual.values)/base_int
tmp=internal[['StationID']].copy(); tmp['Score']=scores
station_q=tmp.groupby('StationID').Score.quantile(.90).values
fac=float(np.quantile(station_q,.90,method='higher')/qnom)
scale=base_fin*fac
mu=selected.Mean.values.astype(float); yobs=selected.Observed.values.astype(float); res=yobs-mu
lower=mu+scale*student_t.ppf(.05,nu); upper=mu+scale*student_t.ppf(.95,nu)

def crps_mc(yv,muv,scv,nuv,seed,S=2000,chunk=1000):
    rng=np.random.default_rng(seed); z1=rng.standard_t(nuv,size=S); z2=rng.standard_t(nuv,size=S); c2=.5*np.mean(np.abs(z1-z2)); out=np.empty(len(yv))
    for a in range(0,len(yv),chunk):
        b=min(a+chunk,len(yv)); zz=(yv[a:b]-muv[a:b])/scv[a:b]
        out[a:b]=scv[a:b]*(np.mean(np.abs(z1[None,:]-zz[:,None]),axis=1)-c2)
    return out
crps=crps_mc(yobs,mu,scale,nu,SEED+3000)
logscore=-(student_t.logpdf(res/scale,df=nu)-np.log(scale))
interval_score=(upper-lower)+(2/.10)*(lower-yobs)*(yobs<lower)+(2/.10)*(yobs-upper)*(yobs>upper)
selected['Scale']=scale; selected['df']=nu; selected['Lower90']=lower; selected['Upper90']=upper; selected['CRPS_row']=crps; selected['LogScore_row']=logscore; selected['IntervalScore90']=interval_score
selected.to_csv(OUT/'final_probabilistic_predictions.csv',index=False)

prob_metrics={
    'n':len(selected),'RMSE':float(np.sqrt(np.mean(res*res))),'MAE':float(np.mean(np.abs(res))),'Bias':float(np.mean(mu-yobs)),
    'CRPS':float(np.mean(crps)),'LogScore':float(np.mean(logscore)),'Cov90':float(np.mean((yobs>=lower)&(yobs<=upper))),
    'AIW90':float(np.mean(upper-lower)),'IntervalScore90':float(np.mean(interval_score)),'df':nu,'ScaleMultiplier':mult,'ConformalFactor':fac}
pd.DataFrame([prob_metrics]).to_csv(OUT/'final_probabilistic_metrics.csv',index=False)

# 6. Applied diagnostics
def grouped_metrics(g):
    e=g.Mean-g.Observed
    out={'n':len(g),'ObservedMean':float(g.Observed.mean()),'PredictedMean':float(g.Mean.mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(e))}
    if 'Lower90' in g:
        out.update({'CRPS':float(g.CRPS_row.mean()),'Cov90':float(np.mean((g.Observed>=g.Lower90)&(g.Observed<=g.Upper90))),'AIW90':float(np.mean(g.Upper90-g.Lower90))})
    return pd.Series(out)
station_metrics=selected.groupby(['StationID','StationType','StationArea','NearestSourceKm']).apply(grouped_metrics,include_groups=False).reset_index()
season_metrics=selected.groupby('Season').apply(grouped_metrics,include_groups=False).reset_index()
month_metrics=selected.groupby('Month').apply(grouped_metrics,include_groups=False).reset_index()
type_metrics=selected.groupby('StationType').apply(grouped_metrics,include_groups=False).reset_index()
area_metrics=selected.groupby('StationArea').apply(grouped_metrics,include_groups=False).reset_index()
station_metrics.to_csv(OUT/'final_station_metrics.csv',index=False)
season_metrics.to_csv(OUT/'final_season_metrics.csv',index=False)
month_metrics.to_csv(OUT/'final_month_metrics.csv',index=False)
type_metrics.to_csv(OUT/'final_station_type_metrics.csv',index=False)
area_metrics.to_csv(OUT/'final_station_area_metrics.csv',index=False)

train_threshold=float(np.nanquantile(y[np.ix_(train_t,source_idx)],.90))
selected['HighEvent']=selected.Observed>train_threshold
selected['ProbHigh']=1-student_t.cdf((train_threshold-selected.Mean)/selected.Scale,df=nu)
try: auroc=float(roc_auc_score(selected.HighEvent,selected.ProbHigh))
except Exception: auroc=float('nan')
try: auprc=float(average_precision_score(selected.HighEvent,selected.ProbHigh))
except Exception: auprc=float('nan')
event={'ThresholdTrainP90':train_threshold,'EventRate':float(selected.HighEvent.mean()),'Brier':float(brier_score_loss(selected.HighEvent,selected.ProbHigh)),'AUROC':auroc,'AUPRC':auprc,'MeanPredictedProbability':float(selected.ProbHigh.mean()),'ObservedEvents':int(selected.HighEvent.sum())}
pd.DataFrame([event]).to_csv(OUT/'final_high_pollution_event_metrics.csv',index=False)
selected.to_csv(OUT/'final_probabilistic_predictions.csv',index=False)

with open(OUT/'final_model_parameters.json','w') as f:
    json.dump({'seed':SEED,'source_stations':dev,'locked_stations':locked,'base_best_iteration':int(base.best_iteration_),
               'edge_best_epoch':int(best[2]),'edge_shrink':shrink,'edge_coefficients':dict(zip(pair_names,map(float,coef))),
               'student_t_df':nu,'student_t_multiplier':mult,'station_block_conformal_factor':fac,'training_event_threshold_p90':train_threshold},f,indent=2)

print('\nPOINT METRICS')
print(point_metrics.to_string(index=False))
print('\nPROBABILISTIC',prob_metrics)
print('\nSTATION METRICS')
print(station_metrics.to_string(index=False))
print('\nSEASON METRICS')
print(season_metrics.to_string(index=False))
print('\nEVENT',event)
print('Saved to',OUT)
