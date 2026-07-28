import csv, math, os, warnings, time, json
import numpy as np
import torch
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
warnings.filterwarnings('ignore')
SEED=20260726
FOLD=int(os.environ.get('FOLD_ONLY','5'))
CSV='/mnt/data/Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
OUT=f'/mnt/data/po_stage2_learned_edge_fold{FOLD}'; os.makedirs(OUT,exist_ok=True)
locked=['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[])); held_ids=folds[FOLD]; source_ids=sorted(set(dev)-set(held_ids))
CONT=['BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos']
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
lons=np.array([static[s]['lon'] for s in stations]);lats=np.array([static[s]['lat'] for s in stations]);alts=np.array([static[s]['alt'] for s in stations])
def hav(lon1,lat1,lon2,lat2):
 R=6371.0088;a1=np.radians(lat1);a2=np.radians(lat2);dl=np.radians(lon2-lon1);dp=np.radians(lat2-lat1);a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
dist=np.stack([hav(lons[i],lats[i],lons,lats) for i in range(N)]).astype(np.float32)
TYPES=['background','traffic','industrial'];AREAS=['urban','suburban','rural']
train_t=np.where(np.isin(year,[2018,2019,2020,2021]))[0]
def clim_features(j):
 lon=static[stations[j]]['lon'];lat=static[stations[j]]['lat'];alt=static[stations[j]]['alt'];vals=[lon,lat,alt,lon*lon,lat*lat,lon*lat]
 for k in range(8):
  a=cont[train_t,j,k].astype(float);a=a[np.isfinite(a)];vals += [float(np.mean(a)),float(np.std(a))]
 vals += [float(static[stations[j]]['type']==k) for k in TYPES]+[float(static[stations[j]]['area']==k) for k in AREAS]
 return vals
Xclim=np.asarray([clim_features(j) for j in range(N)],float);train_mean=np.nanmean(y[train_t],axis=0)
clim=np.full(N,np.nan,float)
for j in source_idx:
 others=[q for q in source_idx if q!=j];m=make_pipeline(StandardScaler(),KernelRidge(alpha=.01,kernel='rbf',gamma=.01));m.fit(Xclim[others],train_mean[others]);clim[j]=np.clip(m.predict(Xclim[j:j+1])[0],np.nanmin(train_mean[others]),np.nanmax(train_mean[others]))
m=make_pipeline(StandardScaler(),KernelRidge(alpha=.01,kernel='rbf',gamma=.01));m.fit(Xclim[source_idx],train_mean[source_idx])
for j in held_idx:clim[j]=np.clip(m.predict(Xclim[j:j+1])[0],np.nanmin(train_mean[source_idx]),np.nanmax(train_mean[source_idx]))

def base_feat(t,j):return list(cont[t,j].astype(float))+[float(wind[t,j]==k) for k in [1,2,3,4]]+[float(static[stations[j]]['type']==k) for k in TYPES]+[float(static[stations[j]]['area']==k) for k in AREAS]+[float(clim[j])]
base_n=len(base_feat(0,source_idx[0]))
fixed_names=['net_mean','net_median','net_sd','net_q10','net_q90','nearest','nearest_dist','knn3','knn5','knn10']
for h in [25,50,100,200]:fixed_names += [f'rbf{h}_mean',f'rbf{h}_sd',f'rbf{h}_effn']
for p in [1,2]:fixed_names += [f'idw{p}_mean',f'idw{p}_sd']
def fixed_feat(field,t,j):
 src=np.asarray(source_idx);v=field[t,src].astype(float);ok=np.isfinite(v)&(src!=j);src=src[ok];v=v[ok]
 if len(v)==0:return [np.nan]*len(fixed_names)
 d=dist[j,src].astype(float);o=np.argsort(d);vv=v[o];dd=d[o];out=[v.mean(),np.median(v),v.std(),np.quantile(v,.1),np.quantile(v,.9),vv[0],dd[0],vv[:min(3,len(vv))].mean(),vv[:min(5,len(vv))].mean(),vv[:min(10,len(vv))].mean()]
 for h in [25,50,100,200]:
  w=np.exp(-.5*(d/h)**2);sw=w.sum();mm=(w*v).sum()/sw;sd=np.sqrt((w*(v-mm)**2).sum()/sw);eff=sw*sw/(w*w).sum();out += [mm,sd,eff]
 for p in [1,2]:
  w=1/np.maximum(d,1)**p;sw=w.sum();mm=(w*v).sum()/sw;sd=np.sqrt((w*(v-mm)**2).sum()/sw);out += [mm,sd]
 return out

def rows_base(target_idx,years):
 X=[];yy=[];meta=[];targets=set(target_idx)
 for t in np.where(np.isin(year,list(years)))[0]:
  for j in np.where(np.isfinite(y[t]))[0]:
   if j not in targets:continue
   X.append(base_feat(t,j));yy.append(float(y[t,j]-clim[j]));meta.append((t,j))
 return np.asarray(X,np.float32),np.asarray(yy,np.float32),meta

def lgb(seed,n=220):return LGBMRegressor(objective='regression_l2',n_estimators=n,learning_rate=.035,num_leaves=31,min_child_samples=100,colsample_bytree=.85,reg_alpha=.1,reg_lambda=2,random_state=seed,n_jobs=4,verbosity=-1)
# Base OOF and 2022 predictions
R=np.full_like(y,np.nan);base_oof=np.full_like(y,np.nan)
for yy in [2018,2019,2020,2021]:
 Xtr,ytr,mtr=rows_base(source_idx,set([2018,2019,2020,2021])-{yy});Xv,yv,mv=rows_base(source_idx,{yy});mod=lgb(SEED+yy,180);mod.fit(Xtr,ytr);pa=mod.predict(Xv)
 for k,(t,j) in enumerate(mv):pp=clim[j]+pa[k];base_oof[t,j]=pp;R[t,j]=y[t,j]-pp
Xtr,ytr,mtr=rows_base(source_idx,{2018,2019,2020,2021});Xcal,ycal_an,mcal=rows_base(source_idx,{2022});Xte,yte_an,mte=rows_base(held_idx,{2022});ycal=np.array([y[t,j] for t,j in mcal],float);yte=np.array([y[t,j] for t,j in mte],float)
base=lgb(SEED+FOLD,350);base.fit(Xtr,ytr,eval_set=[(Xcal,ycal_an)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)]);pcal=np.array([clim[j] for t,j in mcal])+base.predict(Xcal);pte=np.array([clim[j] for t,j in mte])+base.predict(Xte)
for k,(t,j) in enumerate(mcal):R[t,j]=y[t,j]-pcal[k]
# Pair features for learned edge
pair_names=['log_distance','altitude_diff','climatology_diff','same_type','same_area','BLH_diff','RH_diff','T2M_diff','Wind_diff','base_prediction_diff']
static_pair=np.zeros((N,N,5),np.float32)
for i in range(N):
 for j in range(N):
  static_pair[i,j]=[np.log1p(dist[i,j]/25.0),abs(alts[i]-alts[j])/300.0,abs(clim[i]-clim[j])/10.0,float(static[stations[i]]['type']==static[stations[j]]['type']),float(static[stations[i]]['area']==static[stations[j]]['area'])]
def pair_tensor(meta, field, basefield):
 M=len(meta);K=len(source_idx);F=10;pf=np.zeros((M,K,F),np.float32);rv=np.zeros((M,K),np.float32);mask=np.zeros((M,K),bool);target=np.zeros(M,np.float32)
 for a,(t,j) in enumerate(meta):
  target[a]=field[t,j]
  for b,q in enumerate(source_idx):
   if q==j or not np.isfinite(field[t,q]):continue
   mask[a,b]=True;rv[a,b]=field[t,q];pf[a,b,:5]=static_pair[j,q]
   pf[a,b,5]=abs(cont[t,j,0]-cont[t,q,0]);pf[a,b,6]=abs(cont[t,j,1]-cont[t,q,1]);pf[a,b,7]=abs(cont[t,j,2]-cont[t,q,2]);pf[a,b,8]=abs(cont[t,j,4]-cont[t,q,4]);pf[a,b,9]=abs(basefield[t,j]-basefield[t,q])/10.0
 return pf,rv,mask,target
# attention train rows: source OOF residuals; validation year 2021
meta_all=[]
for t in np.where(np.isin(year,[2018,2019,2020,2021]))[0]:
 for j in source_idx:
  if np.isfinite(R[t,j]) and np.sum(np.isfinite(R[t,source_idx]))>=5:meta_all.append((t,j))
meta_tr=[m for m in meta_all if year[m[0]]<=2020];meta_va=[m for m in meta_all if year[m[0]]==2021]
# sample for speed, deterministic
rng=np.random.default_rng(SEED+FOLD)
if len(meta_tr)>28000:meta_tr=[meta_tr[i] for i in rng.choice(len(meta_tr),28000,replace=False)]
if len(meta_va)>9000:meta_va=[meta_va[i] for i in rng.choice(len(meta_va),9000,replace=False)]
ptr,rtr,mtr_att,ytr_att=pair_tensor(meta_tr,R,base_oof);pva,rva,mva,yva_att=pair_tensor(meta_va,R,base_oof)
class Edge(torch.nn.Module):
 def __init__(self):
  super().__init__();self.raw_pen=torch.nn.Parameter(torch.tensor([0.8,0.2,0.8,0.1,0.1,0.1,0.1,0.1],dtype=torch.float32));self.bonus=torch.nn.Parameter(torch.tensor([0.2,0.2]));self.raw_shrink=torch.nn.Parameter(torch.tensor(0.0))
 def coeff(self):
  p=-torch.nn.functional.softplus(self.raw_pen);return torch.stack([p[0],p[1],p[2],self.bonus[0],self.bonus[1],p[3],p[4],p[5],p[6],p[7]])
 def forward(self,pf,rv,mask):
  sc=(pf*self.coeff()).sum(-1);sc=torch.where(mask,sc,torch.tensor(-1e9,device=sc.device));w=torch.softmax(sc,dim=1);mu=(w*rv).sum(1);sh=2*torch.sigmoid(self.raw_shrink);return sh*mu,w

def fit_edge(pftr,rvtr,mtrg,ytrg,pfva,rvva,mvag,yvag):
 torch.manual_seed(SEED+FOLD);mod=Edge();opt=torch.optim.Adam(mod.parameters(),lr=.03,weight_decay=1e-3);lossfn=torch.nn.SmoothL1Loss(beta=5.0)
 trds=torch.utils.data.TensorDataset(torch.from_numpy(pftr),torch.from_numpy(rvtr),torch.from_numpy(mtrg),torch.from_numpy(ytrg));loader=torch.utils.data.DataLoader(trds,batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(SEED+FOLD))
 pva_t=torch.from_numpy(pfva);rva_t=torch.from_numpy(rvva);mva_t=torch.from_numpy(mvag);yva_t=torch.from_numpy(yvag)
 best=None;bad=0
 for ep in range(160):
  mod.train()
  for a,b,c,d in loader:
   opt.zero_grad();pred,w=mod(a,b,c);loss=lossfn(pred,d)+.03*(w*w).sum(1).mean();loss.backward();opt.step()
  mod.eval()
  with torch.no_grad():pv,w=mod(pva_t,rva_t,mva_t);vl=float(lossfn(pv,yva_t)+.03*(w*w).sum(1).mean())
  if best is None or vl<best[0]-1e-5:best=(vl,{k:v.detach().clone() for k,v in mod.state_dict().items()},ep);bad=0
  else:bad+=1
  if bad>=20:break
 mod.load_state_dict(best[1]);return mod,best
edge,best=fit_edge(ptr,rtr,mtr_att,ytr_att,pva,rva,mva,yva_att)
coef=edge.coeff().detach().numpy();shrink=float(2*torch.sigmoid(edge.raw_shrink).detach());print('edge',dict(zip(pair_names,coef)),'shrink',shrink,'epoch',best[2])
learn_names=['att_mean','att_sd','att_effn','att_top_weight','att_weighted_distance','att_weighted_climdiff']
def learned_feat(field,basefield,t,j):
 src=np.asarray(source_idx);v=field[t,src].astype(np.float32);ok=np.isfinite(v)&(src!=j)
 if ok.sum()==0:return [np.nan]*len(learn_names)
 src2=src[ok];v2=v[ok];pf=np.zeros((len(src2),10),np.float32);pf[:,:5]=static_pair[j,src2];pf[:,5]=np.abs(cont[t,j,0]-cont[t,src2,0]);pf[:,6]=np.abs(cont[t,j,1]-cont[t,src2,1]);pf[:,7]=np.abs(cont[t,j,2]-cont[t,src2,2]);pf[:,8]=np.abs(cont[t,j,4]-cont[t,src2,4]);pf[:,9]=np.abs(basefield[t,j]-basefield[t,src2])/10.0
 sc=pf@coef;sc-=np.max(sc);w=np.exp(sc);w/=w.sum();mu=shrink*np.sum(w*v2);sd=np.sqrt(np.sum(w*(v2-np.sum(w*v2))**2));eff=1/np.sum(w*w);return [float(mu),float(sd),float(eff),float(w.max()),float(np.sum(w*dist[j,src2])),float(np.sum(w*np.abs(clim[j]-clim[src2])))]

def corr_rows(target_idx,years,mode):
 X=[];yy=[];meta=[]
 tids=np.where(np.isin(year,list(years)))[0]
 src=np.asarray(source_idx,dtype=int)
 coefv=coef.astype(np.float32)
 for t in tids:
  tg=np.asarray([j for j in target_idx if np.isfinite(y[t,j])],dtype=int)
  if len(tg)==0: continue
  v=R[t,src].astype(np.float32); valid=np.isfinite(v)
  if valid.sum()==0: continue
  # Pair features for all targets at this date
  pf=np.zeros((len(tg),len(src),10),np.float32)
  pf[:,:,:5]=static_pair[tg[:,None],src[None,:],:]
  pf[:,:,5]=np.abs(cont[t,tg,0][:,None]-cont[t,src,0][None,:])
  pf[:,:,6]=np.abs(cont[t,tg,1][:,None]-cont[t,src,1][None,:])
  pf[:,:,7]=np.abs(cont[t,tg,2][:,None]-cont[t,src,2][None,:])
  pf[:,:,8]=np.abs(cont[t,tg,4][:,None]-cont[t,src,4][None,:])
  bf=base_oof if year[t]<=2021 else base2022
  pf[:,:,9]=np.abs(bf[t,tg][:,None]-bf[t,src][None,:])/10.0
  sc=np.tensordot(pf,coefv,axes=([2],[0]))
  mask=np.broadcast_to(valid[None,:],sc.shape).copy()
  mask &= (tg[:,None] != src[None,:])
  sc=np.where(mask,sc,-1e9);sc-=np.max(sc,axis=1,keepdims=True)
  w=np.exp(sc);w=np.where(mask,w,0.0);sw=w.sum(axis=1,keepdims=True);w=np.divide(w,sw,out=np.zeros_like(w),where=sw>0)
  rawmu=(w*np.nan_to_num(v,nan=0.0)[None,:]).sum(axis=1)
  mu=shrink*rawmu
  sd=np.sqrt((w*(np.nan_to_num(v,nan=0.0)[None,:]-rawmu[:,None])**2).sum(axis=1))
  eff=1/np.maximum((w*w).sum(axis=1),1e-8);top=w.max(axis=1)
  wdist=(w*dist[tg[:,None],src[None,:]]).sum(axis=1)
  wclim=(w*np.abs(clim[tg[:,None]]-clim[src[None,:]])).sum(axis=1)
  for a,j in enumerate(tg):
   z=base_feat(t,int(j))+[float(mu[a]),float(sd[a]),float(eff[a]),float(top[a]),float(wdist[a]),float(wclim[a]),float(bf[t,j])]
   X.append(z);yy.append(float(R[t,j]));meta.append((t,int(j)))
 return np.asarray(X,np.float32),np.asarray(yy,np.float32),meta

# basefield for 2022 source and held, needed by pair feature
base2022=np.full_like(y,np.nan)
for k,(t,j) in enumerate(mcal):base2022[t,j]=pcal[k]
for k,(t,j) in enumerate(mte):base2022[t,j]=pte[k]
models={'Climatology-anomaly baseline':pte}
mode='learned';label='Climatology learned-edge graph'
Xco=[];yco=[]
for yy in [2018,2019,2020,2021]:
 Xa,ya,ma=corr_rows(source_idx,{yy},mode);Xco.append(Xa);yco.append(ya)
Xco=np.vstack(Xco);yco=np.concatenate(yco);Xc,yc,mc=corr_rows(source_idx,{2022},mode);Xt,yt,mt=corr_rows(held_idx,{2022},mode)
cor=lgb(SEED+1010+FOLD,350);cor.fit(Xco,yco,eval_set=[(Xc,yc)],eval_metric='l1',callbacks=[early_stopping(35,verbose=False),log_evaluation(0)]);models[label]=pte+cor.predict(Xt)
res=[];pred=[]
for name,mu in models.items():
 e=yte-mu;met={'Fold':FOLD,'Model':name,'n':len(yte),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(mu-yte))};res.append(met);print(met)
 for k,(t,j) in enumerate(mte):pred.append([FOLD,name,stations[j],dates[t],float(yte[k]),float(mu[k]),float(clim[j])])
with open(OUT+'/metrics.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(res[0]));w.writeheader();w.writerows(res)
with open(OUT+'/predictions.csv','w',newline='') as f:w=csv.writer(f);w.writerow(['Fold','Model','StationID','Date','Observed','Mean','Climatology']);w.writerows(pred)
with open(OUT+'/edge_parameters.json','w') as f:json.dump({'fold':FOLD,'coefficients':dict(zip(pair_names,map(float,coef))),'shrink':shrink,'best_epoch':best[2]},f,indent=2)
print('done')
