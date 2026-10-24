import os, glob, math, json, csv
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from lightgbm import LGBMRegressor

SEED=20260726
BASE='/mnt/data'
# combine stage2 fixed climatology-residual graph predictions
parts=[]
for f in range(1,6):
    p=f'{BASE}/po_stage2_learned_edge_fold{f}/predictions.csv'
    d=pd.read_csv(p)
    d=d[d.Model=='Climatology learned-edge graph'].copy()
    parts.append(d)
pred=pd.concat(parts,ignore_index=True)
pred['Date']=pd.to_datetime(pred['Date'])
# merge covariates
cols=['StationID','Date','Year','Month','Season','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','WindDirection','StationType','StationArea']
data=pd.read_csv(f'{BASE}/Po_Valley_PM25_Final_2018_2023_Observed(2).csv',usecols=cols)
data['Date']=pd.to_datetime(data['Date'])
pred=pred.merge(data,on=['StationID','Date'],how='left',validate='one_to_one')
# fold-specific nearest source distance
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[]))
meta=data.drop_duplicates('StationID').set_index('StationID')[['Longitude_Z_Train','Latitude_Z_Train']]
# use raw approximate great circle from original data to nearest source
rawmeta=pd.read_csv(f'{BASE}/Po_Valley_PM25_Final_2018_2023_Observed(2).csv',usecols=['StationID','Longitude','Latitude']).drop_duplicates('StationID').set_index('StationID')
def hav(lon1,lat1,lon2,lat2):
    R=6371.0088;a1=np.radians(lat1);a2=np.radians(lat2);dl=np.radians(lon2-lon1);dp=np.radians(lat2-lat1);a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2;return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
nearest={}
for f,held in folds.items():
    src=sorted(set(dev)-set(held))
    slon=rawmeta.loc[src,'Longitude'].values;slat=rawmeta.loc[src,'Latitude'].values
    for s in held:
        nearest[(f,s)]=float(np.min(hav(rawmeta.loc[s,'Longitude'],rawmeta.loc[s,'Latitude'],slon,slat)))
pred['NearestSourceKm']=[nearest[(int(f),s)] for f,s in zip(pred.Fold,pred.StationID)]
pred['Residual']=pred['Observed']-pred['Mean']
pred['LogAbsResidual']=np.log(np.abs(pred['Residual'])+0.5)
# encode features
num=['Mean','Climatology','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','NearestSourceKm']
cat=['WindDirection','StationType','StationArea']
X=pd.get_dummies(pred[num+cat].copy(),columns=cat,drop_first=False,dtype=float)
X=X.replace([np.inf,-np.inf],np.nan).fillna(0.0)

def fit_t(resid,s0):
    resid=np.asarray(resid,float);s0=np.clip(np.asarray(s0,float),0.2,100)
    def obj(par):
        mult=np.exp(par[0]);nu=2.05+np.exp(par[1]);sc=mult*s0
        return -np.mean(student_t.logpdf(resid/sc,df=nu)-np.log(sc))
    opt=minimize(obj,[0.0,math.log(5.0)],method='Nelder-Mead',options={'maxiter':400})
    return float(np.exp(opt.x[0])),float(2.05+np.exp(opt.x[1]))

def crps_mc(y,mu,scale,nu,seed,S=250,chunk=3000):
    rng=np.random.default_rng(seed);z1=rng.standard_t(nu,size=S);z2=rng.standard_t(nu,size=S);c2=.5*np.mean(np.abs(z1-z2));out=np.empty(len(y))
    for a in range(0,len(y),chunk):
        b=min(a+chunk,len(y));zz=(y[a:b]-mu[a:b])/scale[a:b];out[a:b]=scale[a:b]*(np.mean(np.abs(z1[None,:]-zz[:,None]),axis=1)-c2)
    return out

all_variants=[]; params=[]
for variant in ['global','station_block90','station_block_finite']:
    out=[]
    for f in range(1,6):
        tr=pred.Fold!=f;te=pred.Fold==f
        model=LGBMRegressor(objective='regression_l1',n_estimators=180,learning_rate=.025,num_leaves=15,min_child_samples=120,colsample_bytree=.85,reg_alpha=.2,reg_lambda=3,random_state=SEED+f,n_jobs=4,verbosity=-1)
        model.fit(X.loc[tr],pred.loc[tr,'LogAbsResidual'])
        s0tr=np.exp(model.predict(X.loc[tr])).clip(.2,100);s0te=np.exp(model.predict(X.loc[te])).clip(.2,100)
        mult,nu=fit_t(pred.loc[tr,'Residual'].values,s0tr)
        base_tr=mult*s0tr;base_te=mult*s0te
        qnom=student_t.ppf(.95,nu)
        factors={}
        scores=np.abs(pred.loc[tr,'Residual'].values)/base_tr
        if variant=='global':
            fac=float(np.quantile(scores,.90)/qnom);factors['All']=fac;scale=base_te*fac
        else:
            trdf=pred.loc[tr].copy();trdf['Score']=scores
            station_q=trdf.groupby('StationID')['Score'].quantile(.90).values
            if variant=='station_block90':
                level=.90
            else:
                m=len(station_q);level=min(1.0,math.ceil((m+1)*.90)/m)
            fac=float(np.quantile(station_q,level,method='higher')/qnom);factors['All']=fac;factors['StationQuantileLevel']=level;scale=base_te*fac
        yobs=pred.loc[te,'Observed'].values.astype(float);mu=pred.loc[te,'Mean'].values.astype(float);res=yobs-mu
        q05=mu+scale*student_t.ppf(.05,nu);q95=mu+scale*student_t.ppf(.95,nu)
        cr=crps_mc(yobs,mu,scale,nu,SEED+100*f+(0 if variant=='global' else 50))
        ls=-(student_t.logpdf(res/scale,df=nu)-np.log(scale))
        sub=pred.loc[te,['Fold','StationID','Date','Observed','Mean','Season','Climatology','NearestSourceKm']].copy();sub['Variant']=variant;sub['Scale']=scale;sub['df']=nu;sub['Lower90']=q05;sub['Upper90']=q95;sub['CRPS_row']=cr;sub['LogScore_row']=ls;out.append(sub)
        params.append({'Variant':variant,'TargetFold':f,'df':nu,'Multiplier':mult,'Factors':json.dumps(factors),'TrainN':int(tr.sum()),'TestN':int(te.sum())})
    all_variants.append(pd.concat(out,ignore_index=True))
resall=pd.concat(all_variants,ignore_index=True)
# metrics
metrics=[]
for (variant,fold),g in resall.groupby(['Variant','Fold']):
    e=g.Observed-g.Mean
    metrics.append({'Variant':variant,'Fold':int(fold),'n':len(g),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'CRPS':float(g.CRPS_row.mean()),'LogScore':float(g.LogScore_row.mean()),'Cov90':float(np.mean((g.Observed>=g.Lower90)&(g.Observed<=g.Upper90))),'AIW90':float(np.mean(g.Upper90-g.Lower90))})
for variant,g in resall.groupby('Variant'):
    e=g.Observed-g.Mean
    metrics.append({'Variant':variant,'Fold':'All','n':len(g),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'CRPS':float(g.CRPS_row.mean()),'LogScore':float(g.LogScore_row.mean()),'Cov90':float(np.mean((g.Observed>=g.Lower90)&(g.Observed<=g.Upper90))),'AIW90':float(np.mean(g.Upper90-g.Lower90))})
met=pd.DataFrame(metrics)
# daily block bootstrap for calibrated variants against pooled global
boots=[]
for alt in ['station_block90','station_block_finite']:
    wide=resall[resall.Variant.isin(['global',alt])].pivot_table(index=['Fold','StationID','Date'],columns='Variant',values='CRPS_row').reset_index()
    wide['delta']=wide[alt]-wide['global'];daily=wide.groupby('Date')['delta'].mean().values
    rng=np.random.default_rng(SEED+len(alt));vals=np.array([np.mean(rng.choice(daily,len(daily),replace=True)) for _ in range(3000)])
    boots.append({'Comparison':alt+' minus global','Mean_CRPS_Difference':float(np.mean(daily)),'CI_Lower':float(np.quantile(vals,.025)),'CI_Upper':float(np.quantile(vals,.975)),'Probability_Alternative_Better':float(np.mean(vals<0))})
# station metrics selected variants
station=[]
for (variant,sid),g in resall.groupby(['Variant','StationID']):
    e=g.Mean-g.Observed
    station.append({'Variant':variant,'StationID':sid,'Fold':int(g.Fold.iloc[0]),'n':len(g),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(e)),'Cov90':float(np.mean((g.Observed>=g.Lower90)&(g.Observed<=g.Upper90))),'AIW90':float(np.mean(g.Upper90-g.Lower90))})
# save
outdir=f'{BASE}/Po_Valley_Stage2'
os.makedirs(outdir,exist_ok=True)
pred.to_csv(f'{outdir}/stage2_learned_mean_predictions.csv',index=False)
resall.to_csv(f'{outdir}/stage2_learned_probabilistic_predictions_block.csv',index=False)
met.to_csv(f'{outdir}/stage2_learned_probabilistic_metrics_block.csv',index=False)
pd.DataFrame(params).to_csv(f'{outdir}/stage2_learned_uncertainty_parameters_block.csv',index=False)
pd.DataFrame(station).to_csv(f'{outdir}/stage2_learned_station_metrics_block.csv',index=False)
pd.DataFrame(boots).to_csv(f'{outdir}/stage2_learned_calibration_bootstrap_block.csv',index=False)
print(met.to_string(index=False))
print('BOOTS',boots)
