import math, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from lightgbm import LGBMRegressor

SEED=20260726
BASE=Path('/mnt/data')
CSV=BASE/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
STAGE2=BASE/'Po_Valley_Stage2'
OUT=BASE/'Po_Valley_Final_Test_2023'
sel='Inductive climatology + learned edge'
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[]))
locked=['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']

pred_long=pd.read_csv(OUT/'final_point_predictions_long.csv')
pred_long['Date']=pd.to_datetime(pred_long.Date)
selected=pred_long[pred_long.Model==sel].copy()
internal=pd.read_csv(STAGE2/'stage2_learned_mean_predictions.csv')
internal['Date']=pd.to_datetime(internal.Date)
# Stage 2 file already contains its training covariates and nearest-source distances.
internal['Residual']=internal['Observed']-internal['Mean']
internal['LogAbsResidual']=np.log(np.abs(internal.Residual)+0.5)

cov_cols=['StationID','Date','Year','BLH_Z_Train','RH_Z_Train','T2M_Z_Train','Precipitation_Z_Train','WindSpeed_Z_Train','SolarRadiation_Z_Train','LAI_HighVegetation_Z_Train','LAI_LowVegetation_Z_Train','Longitude_Z_Train','Latitude_Z_Train','Altitude_Z_Train','MonthSin','MonthCos','DayOfYearSin','DayOfYearCos','WindDirection']
data=pd.read_csv(CSV,usecols=cov_cols+['PM25'])
data['Date']=pd.to_datetime(data.Date)
selected=selected.merge(data[data.Year==2023].drop(columns=['Year','PM25']),on=['StationID','Date'],how='left',validate='one_to_one')

# Great-circle nearest source distance.
coords=pd.read_csv(CSV,usecols=['StationID','Longitude','Latitude']).drop_duplicates('StationID').set_index('StationID')
def hav(lon1,lat1,lon2,lat2):
    R=6371.0088; a1=np.radians(lat1); a2=np.radians(lat2); dl=np.radians(lon2-lon1); dp=np.radians(lat2-lat1)
    a=np.sin(dp/2)**2+np.cos(a1)*np.cos(a2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))
slon=coords.loc[dev,'Longitude'].values; slat=coords.loc[dev,'Latitude'].values
nearest={s:float(np.min(hav(coords.loc[s,'Longitude'],coords.loc[s,'Latitude'],slon,slat))) for s in locked}
selected['NearestSourceKm']=[nearest[s] for s in selected.StationID]

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

def crps_mc(yv,muv,scv,nuv,seed,S=3000,chunk=1000):
    rng=np.random.default_rng(seed); z1=rng.standard_t(nuv,size=S); z2=rng.standard_t(nuv,size=S); c2=.5*np.mean(np.abs(z1-z2)); out=np.empty(len(yv))
    for a in range(0,len(yv),chunk):
        b=min(a+chunk,len(yv)); zz=(yv[a:b]-muv[a:b])/scv[a:b]
        out[a:b]=scv[a:b]*(np.mean(np.abs(z1[None,:]-zz[:,None]),axis=1)-c2)
    return out
crps=crps_mc(yobs,mu,scale,nu,SEED+3000)
logscore=-(student_t.logpdf(res/scale,df=nu)-np.log(scale))
interval_score=(upper-lower)+(2/.10)*(lower-yobs)*(yobs<lower)+(2/.10)*(yobs-upper)*(yobs>upper)
selected['Scale']=scale; selected['df']=nu; selected['Lower90']=lower; selected['Upper90']=upper; selected['CRPS_row']=crps; selected['LogScore_row']=logscore; selected['IntervalScore90']=interval_score

prob_metrics={'n':len(selected),'RMSE':float(np.sqrt(np.mean(res*res))),'MAE':float(np.mean(np.abs(res))),'Bias':float(np.mean(mu-yobs)),'CRPS':float(np.mean(crps)),'LogScore':float(np.mean(logscore)),'Cov90':float(np.mean((yobs>=lower)&(yobs<=upper))),'AIW90':float(np.mean(upper-lower)),'IntervalScore90':float(np.mean(interval_score)),'df':nu,'ScaleMultiplier':mult,'ConformalFactor':fac}
pd.DataFrame([prob_metrics]).to_csv(OUT/'final_probabilistic_metrics.csv',index=False)

def grouped_metrics(g):
    e=g.Mean-g.Observed
    return pd.Series({'n':len(g),'ObservedMean':float(g.Observed.mean()),'PredictedMean':float(g.Mean.mean()),'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),'Bias':float(np.mean(e)),'CRPS':float(g.CRPS_row.mean()),'Cov90':float(np.mean((g.Observed>=g.Lower90)&(g.Observed<=g.Upper90))),'AIW90':float(np.mean(g.Upper90-g.Lower90))})
station_metrics=selected.groupby(['StationID','StationType','StationArea','NearestSourceKm']).apply(grouped_metrics,include_groups=False).reset_index()
season_metrics=selected.groupby('Season').apply(grouped_metrics,include_groups=False).reset_index()
month_metrics=selected.groupby('Month').apply(grouped_metrics,include_groups=False).reset_index()
type_metrics=selected.groupby('StationType').apply(grouped_metrics,include_groups=False).reset_index()
area_metrics=selected.groupby('StationArea').apply(grouped_metrics,include_groups=False).reset_index()
station_metrics.to_csv(OUT/'final_station_metrics.csv',index=False); season_metrics.to_csv(OUT/'final_season_metrics.csv',index=False); month_metrics.to_csv(OUT/'final_month_metrics.csv',index=False); type_metrics.to_csv(OUT/'final_station_type_metrics.csv',index=False); area_metrics.to_csv(OUT/'final_station_area_metrics.csv',index=False)

# Predefined high-pollution threshold from 2018-2021 development stations only.
train=pd.read_csv(CSV,usecols=['StationID','Year','PM25'])
threshold=float(train[(train.StationID.isin(dev))&(train.Year.isin([2018,2019,2020,2021]))].PM25.quantile(.90))
selected['HighEvent']=selected.Observed>threshold
selected['ProbHigh']=1-student_t.cdf((threshold-selected.Mean)/selected.Scale,df=nu)
event={'ThresholdTrainP90':threshold,'EventRate':float(selected.HighEvent.mean()),'Brier':float(brier_score_loss(selected.HighEvent,selected.ProbHigh)),'AUROC':float(roc_auc_score(selected.HighEvent,selected.ProbHigh)),'AUPRC':float(average_precision_score(selected.HighEvent,selected.ProbHigh)),'MeanPredictedProbability':float(selected.ProbHigh.mean()),'ObservedEvents':int(selected.HighEvent.sum())}
pd.DataFrame([event]).to_csv(OUT/'final_high_pollution_event_metrics.csv',index=False)
selected.to_csv(OUT/'final_probabilistic_predictions.csv',index=False)

params={'seed':SEED,'source_stations':dev,'locked_stations':locked,'base_best_iteration':282,'correction_best_iteration':350,'edge_best_epoch':9,'edge_shrink':0.9425285458564758,'edge_coefficients':{'log_distance':-2.854884386062622,'altitude_diff':-1.0905613899230957,'climatology_diff':-0.6082866191864014,'same_type':-0.05127391964197159,'same_area':-0.045867957174777985,'BLH_diff':-0.22228506207466125,'RH_diff':-0.48978081345558167,'T2M_diff':-0.8504548668861389,'Wind_diff':-0.4135601818561554,'base_prediction_diff':-1.4502899646759033},'student_t_df':nu,'student_t_multiplier':mult,'station_block_conformal_factor':fac,'training_event_threshold_p90':threshold}
(OUT/'final_model_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')

print('PROB',prob_metrics)
print('\nSTATION\n',station_metrics.to_string(index=False))
print('\nSEASON\n',season_metrics.to_string(index=False))
print('\nTYPE\n',type_metrics.to_string(index=False))
print('\nAREA\n',area_metrics.to_string(index=False))
print('\nEVENT',event)
