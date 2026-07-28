"""Matched Student-t uncertainty for the source-validated RBF25 comparator."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from scipy.stats import t as student_t

SEED=20260726

def hav(lon1,lat1,lon2,lat2):
    R=6371.0088; a1=np.radians(lat1); a2=np.radians(lat2)
    dl=np.radians(lon2-lon1); dp=np.radians(lat2-lat1)
    a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def fit_t(resid,s0):
    def obj(par):
        mult=np.exp(par[0]); nu=2.05+np.exp(par[1]); sc=mult*s0
        return -np.mean(student_t.logpdf(resid/sc,df=nu)-np.log(sc))
    o=minimize(obj,[0,math.log(5)],method='Nelder-Mead',options={'maxiter':400})
    return float(np.exp(o.x[0])),float(2.05+np.exp(o.x[1]))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    root=args.root; data_path=root/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv'; out=root/'Po_Valley_Final_Test_2023'
    design=json.loads((Path(__file__).resolve().parents[1]/'config/study_design.json').read_text())
    folds={int(k):v for k,v in design['folds'].items()}; dev=sorted(sum(folds.values(),[]))
    cols=['StationID','Date','Year','PM25','Longitude','Latitude','BLH_Z_Train','RH_Z_Train','T2M_Z_Train',
          'Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train',
          'LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos',
          'DayOfYearSin','DayOfYearCos','WindDirection','StationType','StationArea']
    d=pd.read_csv(data_path,usecols=cols); d.Date=pd.to_datetime(d.Date)
    meta=d.drop_duplicates('StationID').set_index('StationID')
    internal=[]
    for f,held in folds.items():
        src=sorted(set(dev)-set(held)); srcmeta=meta.loc[src]
        dd=d[(d.Year==2022)&d.StationID.isin(held)]
        bydate={date:g.set_index('StationID') for date,g in d[(d.Year==2022)&d.StationID.isin(src)].groupby('Date')}
        for r in dd.itertuples():
            g=bydate.get(r.Date)
            if g is None: continue
            ss=g.index.intersection(src); vals=g.loc[ss,'PM25'].astype(float).values
            dist=hav(r.Longitude,r.Latitude,meta.loc[ss,'Longitude'].values,meta.loc[ss,'Latitude'].values)
            w=np.exp(-.5*(dist/25.0)**2); mu=float(np.sum(w*vals)/np.sum(w))
            internal.append({'Fold':f,'StationID':r.StationID,'Date':r.Date,'Observed':r.PM25,'Mean':mu})
    internal=pd.DataFrame(internal)
    clim=pd.read_csv(root/'Po_Valley_Stage2/stage2_learned_mean_predictions.csv',usecols=['Fold','StationID','Date','Climatology','NearestSourceKm'])
    clim.Date=pd.to_datetime(clim.Date); internal=internal.merge(clim,on=['Fold','StationID','Date'],validate='one_to_one')
    internal=internal.merge(d.drop(columns=['PM25','Longitude','Latitude']),on=['StationID','Date'],validate='one_to_one')
    internal['Residual']=internal.Observed-internal.Mean; internal['LogAbsResidual']=np.log(abs(internal.Residual)+.5)
    fin=pd.read_csv(out/'final_direct_kernel_predictions.csv'); fin.Date=pd.to_datetime(fin.Date)
    sel=pd.read_csv(out/'final_probabilistic_predictions.csv',usecols=['StationID','Date','Climatology','NearestSourceKm','StationType','StationArea']); sel.Date=pd.to_datetime(sel.Date)
    fin=fin.merge(sel,on=['StationID','Date'],validate='one_to_one').merge(d[d.Year==2023].drop(columns=['Year','PM25','Longitude','Latitude','StationType','StationArea']),on=['StationID','Date'],validate='one_to_one')
    num=['Mean','Climatology','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','NearestSourceKm']; cat=['WindDirection','StationType','StationArea']
    X=pd.get_dummies(pd.concat([internal[num+cat],fin[num+cat]],ignore_index=True),columns=cat,dtype=float).replace([np.inf,-np.inf],np.nan).fillna(0)
    Xi=X.iloc[:len(internal)]; Xf=X.iloc[len(internal):]
    model=LGBMRegressor(objective='regression_l1',n_estimators=180,learning_rate=.025,num_leaves=15,min_child_samples=120,colsample_bytree=.85,reg_alpha=.2,reg_lambda=3,random_state=SEED,n_jobs=4,verbosity=-1)
    model.fit(Xi,internal.LogAbsResidual); s0i=np.exp(model.predict(Xi)).clip(.2,100); s0f=np.exp(model.predict(Xf)).clip(.2,100)
    mult,nu=fit_t(internal.Residual.values,s0i); basei=mult*s0i; qnom=student_t.ppf(.95,nu)
    scores=abs(internal.Residual.values)/basei; q=pd.DataFrame({'StationID':internal.StationID,'Score':scores}).groupby('StationID').Score.quantile(.9).values
    fac=float(np.quantile(q,.9,method='higher')/qnom); scale=mult*s0f*fac
    mu=fin.Mean.values; y=fin.Observed.values; lo=mu+scale*student_t.ppf(.05,nu); hi=mu+scale*student_t.ppf(.95,nu)
    rng=np.random.default_rng(SEED+4000); z1=rng.standard_t(nu,3000); z2=rng.standard_t(nu,3000); c=.5*np.mean(abs(z1-z2)); cr=np.empty(len(y))
    for a in range(0,len(y),1000):
        b=min(a+1000,len(y)); zz=(y[a:b]-mu[a:b])/scale[a:b]; cr[a:b]=scale[a:b]*(np.mean(abs(z1[None,:]-zz[:,None]),axis=1)-c)
    ls=-(student_t.logpdf((y-mu)/scale,df=nu)-np.log(scale)); is90=(hi-lo)+20*(lo-y)*(y<lo)+20*(y-hi)*(y>hi)
    fin['Scale']=scale; fin['df']=nu; fin['Lower90']=lo; fin['Upper90']=hi; fin['CRPS_row']=cr; fin['LogScore_row']=ls; fin['IntervalScore90']=is90
    fin.to_csv(out/'final_direct_kernel_probabilistic_predictions.csv',index=False)
    met={'Model':'Source-validated RBF25 + matched uncertainty','n':len(y),'RMSE':float(np.sqrt(np.mean((y-mu)**2))),'MAE':float(np.mean(abs(y-mu))),'Bias':float(np.mean(mu-y)),'CRPS':float(np.mean(cr)),'LogScore':float(np.mean(ls)),'Cov90':float(np.mean((y>=lo)&(y<=hi))),'AIW90':float(np.mean(hi-lo)),'IntervalScore90':float(np.mean(is90)),'df':nu,'ScaleMultiplier':mult,'ConformalFactor':fac}
    pd.DataFrame([met]).to_csv(out/'final_direct_kernel_probabilistic_metrics.csv',index=False); print(met)
if __name__=='__main__': main()
