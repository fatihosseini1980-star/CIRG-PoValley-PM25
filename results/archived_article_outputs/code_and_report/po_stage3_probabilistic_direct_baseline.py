import math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from lightgbm import LGBMRegressor
BASE=Path('/mnt/data');OUT=BASE/'Po_Valley_Final_Test_2023';CSV=BASE/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv';SEED=20260726
# Internal direct predictions, paired with Stage2 climatology and same covariates.
direct=pd.read_csv(BASE/'Po_Valley_Inductive_Graph_Pilot/predictions_crossfold_calibrated.csv')
direct=direct[direct.Model=='IDW/kernel direct'][['Fold','StationID','Date','Observed','Mean']].copy();direct['Date']=pd.to_datetime(direct.Date)
clim=pd.read_csv(BASE/'Po_Valley_Stage2/stage2_learned_mean_predictions.csv',usecols=['Fold','StationID','Date','Climatology','NearestSourceKm'])
clim['Date']=pd.to_datetime(clim.Date)
direct=direct.merge(clim,on=['Fold','StationID','Date'],how='left',validate='one_to_one')
cols=['StationID','Date','Year','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','WindDirection','StationType','StationArea']
data=pd.read_csv(CSV,usecols=cols);data['Date']=pd.to_datetime(data.Date)
direct=direct.merge(data,on=['StationID','Date'],how='left',validate='one_to_one')
direct['Residual']=direct.Observed-direct.Mean;direct['LogAbsResidual']=np.log(abs(direct.Residual)+.5)
# final direct mean
fin=pd.read_csv(OUT/'final_direct_kernel_predictions.csv');fin['Date']=pd.to_datetime(fin.Date)
# add climatology/covariates/nearest from selected final file
sel=pd.read_csv(OUT/'final_probabilistic_predictions.csv',usecols=['StationID','Date','Climatology','NearestSourceKm','StationType','StationArea']);sel['Date']=pd.to_datetime(sel.Date)
fin=fin.merge(sel,on=['StationID','Date'],how='left',validate='one_to_one').merge(data[data.Year==2023].drop(columns=['Year','StationType','StationArea']),on=['StationID','Date'],how='left',validate='one_to_one')
num=['Mean','Climatology','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','NearestSourceKm'];cat=['WindDirection','StationType','StationArea']
Xall=pd.concat([direct[num+cat],fin[num+cat]],ignore_index=True);Xall=pd.get_dummies(Xall,columns=cat,dtype=float).replace([np.inf,-np.inf],np.nan).fillna(0)
Xi=Xall.iloc[:len(direct)];Xf=Xall.iloc[len(direct):]
model=LGBMRegressor(objective='regression_l1',n_estimators=180,learning_rate=.025,num_leaves=15,min_child_samples=120,colsample_bytree=.85,reg_alpha=.2,reg_lambda=3,random_state=SEED,n_jobs=4,verbosity=-1);model.fit(Xi,direct.LogAbsResidual)
s0i=np.exp(model.predict(Xi)).clip(.2,100);s0f=np.exp(model.predict(Xf)).clip(.2,100)
def fit_t(r,s0):
 def obj(par):
  mult=np.exp(par[0]);nu=2.05+np.exp(par[1]);sc=mult*s0
  return -np.mean(student_t.logpdf(r/sc,df=nu)-np.log(sc))
 o=minimize(obj,[0,math.log(5)],method='Nelder-Mead',options={'maxiter':400});return float(np.exp(o.x[0])),float(2.05+np.exp(o.x[1]))
mult,nu=fit_t(direct.Residual.values,s0i);basei=mult*s0i;basef=mult*s0f;qnom=student_t.ppf(.95,nu);scores=abs(direct.Residual.values)/basei;q=pd.DataFrame({'StationID':direct.StationID,'Score':scores}).groupby('StationID').Score.quantile(.9).values;fac=float(np.quantile(q,.9,method='higher')/qnom);scale=basef*fac
mu=fin.Mean.values;y=fin.Observed.values;lo=mu+scale*student_t.ppf(.05,nu);hi=mu+scale*student_t.ppf(.95,nu)
def crps(y,mu,sc,nu,S=3000):
 rng=np.random.default_rng(SEED+4000);z1=rng.standard_t(nu,S);z2=rng.standard_t(nu,S);c=.5*np.mean(abs(z1-z2));out=[]
 for a in range(0,len(y),1000):
  b=min(a+1000,len(y));zz=(y[a:b]-mu[a:b])/sc[a:b];out.extend(sc[a:b]*(np.mean(abs(z1[None,:]-zz[:,None]),axis=1)-c))
 return np.array(out)
cr=crps(y,mu,scale,nu);ls=-(student_t.logpdf((y-mu)/scale,df=nu)-np.log(scale));is90=(hi-lo)+20*(lo-y)*(y<lo)+20*(y-hi)*(y>hi)
fin['Scale']=scale;fin['df']=nu;fin['Lower90']=lo;fin['Upper90']=hi;fin['CRPS_row']=cr;fin['LogScore_row']=ls;fin['IntervalScore90']=is90
met={'Model':'Source-validated direct kernel + matched uncertainty','n':len(y),'RMSE':float(np.sqrt(np.mean((y-mu)**2))),'MAE':float(np.mean(abs(y-mu))),'Bias':float(np.mean(mu-y)),'CRPS':float(np.mean(cr)),'LogScore':float(np.mean(ls)),'Cov90':float(np.mean((y>=lo)&(y<=hi))),'AIW90':float(np.mean(hi-lo)),'IntervalScore90':float(np.mean(is90)),'df':nu,'ScaleMultiplier':mult,'ConformalFactor':fac}
fin.to_csv(OUT/'final_direct_kernel_probabilistic_predictions.csv',index=False);pd.DataFrame([met]).to_csv(OUT/'final_direct_kernel_probabilistic_metrics.csv',index=False)
print(met)
