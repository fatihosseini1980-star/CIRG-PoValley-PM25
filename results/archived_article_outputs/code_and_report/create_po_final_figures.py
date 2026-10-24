from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

OUT=Path('/mnt/data/Po_Valley_Final_Test_2023')
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
DATA='/mnt/data/Po_Valley_PM25_Final_2018_2023_Observed(2).csv'
locked=['STA.IT0778A','STA.IT0912A','STA.IT1037A','STA.IT1128A','STA.IT1153A','STA.IT1388A','STA.IT1590A','STA.IT1826A','STA.IT2108A','STA.IT2121A']
folds={1:['STA.IT0480A','STA.IT0771A','STA.IT0777A','STA.IT1016A','STA.IT1650A','STA.IT1692A','STA.IT1743A','STA.IT1876A'],2:['STA.IT0591A','STA.IT1065A','STA.IT1222A','STA.IT1385A','STA.IT1696A','STA.IT1865A','STA.IT1868A','STA.IT1871A'],3:['STA.IT0470A','STA.IT0983A','STA.IT1130A','STA.IT1510A','STA.IT1532A','STA.IT1588A','STA.IT1877A','STA.IT2101A'],4:['STA.IT0594A','STA.IT0707A','STA.IT0840A','STA.IT1464A','STA.IT1737A','STA.IT1739A','STA.IT1967A','STA.IT2062A'],5:['STA.IT0692A','STA.IT1286A','STA.IT1883A','STA.IT1965A','STA.IT2038A','STA.IT2118A','STA.IT2119A','STA.IT2286A']}
dev=sorted(sum(folds.values(),[]))

# 1. Station design map.
meta=pd.read_csv(DATA,usecols=['StationID','Longitude','Latitude','StationType']).drop_duplicates('StationID')
plt.figure(figsize=(8,6))
a=meta[meta.StationID.isin(dev)];b=meta[meta.StationID.isin(locked)]
plt.scatter(a.Longitude,a.Latitude,label='Development stations (n=40)',s=38)
plt.scatter(b.Longitude,b.Latitude,label='Locked external-test stations (n=10)',s=65,marker='^')
for r in b.itertuples():
    plt.annotate(r.StationID.replace('STA.IT',''),(r.Longitude,r.Latitude),xytext=(3,3),textcoords='offset points',fontsize=7)
plt.xlabel('Longitude');plt.ylabel('Latitude');plt.title('Spatial design of the Po Valley external validation');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_station_design.png',dpi=300);plt.close()

# 2. RMSE model comparison.
point=pd.read_csv(OUT/'final_point_metrics.csv')
direct=pd.read_csv(OUT/'final_direct_kernel_metrics.csv')
allp=pd.concat([point,direct],ignore_index=True)
order=['Climatology-anomaly baseline','Nearest observed source','IDW-2 observed sources','Inductive climatology + learned edge','Source-validated direct kernel']
allp=allp.set_index('Model').loc[order].reset_index()
labels=['Covariate baseline','Nearest source','IDW-2','Learned-edge graph','Source-validated RBF25']
plt.figure(figsize=(8,5))
plt.barh(labels,allp.RMSE)
plt.xlabel(r'RMSE ($\mu$g m$^{-3}$)');plt.title('External 2023 point-prediction performance');plt.gca().invert_yaxis();plt.tight_layout();plt.savefig(FIG/'fig_po_final_rmse_comparison.png',dpi=300);plt.close()

# 3. Observed versus predicted for learned graph.
sel=pd.read_csv(OUT/'final_probabilistic_predictions.csv');sel['Date']=pd.to_datetime(sel.Date)
plt.figure(figsize=(6.5,6))
plt.scatter(sel.Observed,sel.Mean,s=10,alpha=.45)
lo=min(sel.Observed.min(),sel.Mean.min());hi=max(sel.Observed.max(),sel.Mean.max())
plt.plot([lo,hi],[lo,hi],linewidth=1.2,label='Identity line')
plt.xlabel(r'Observed PM$_{2.5}$ ($\mu$g m$^{-3}$)');plt.ylabel(r'Predicted PM$_{2.5}$ ($\mu$g m$^{-3}$)');plt.title('Learned-edge graph: observed versus predicted');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_observed_predicted.png',dpi=300);plt.close()

# 4. Station-level learned vs direct kernel RMSE.
comp=pd.read_csv(OUT/'final_group_comparison_direct.csv');comp=comp[comp.GroupVariable=='StationID'].sort_values('Learned_RMSE')
x=np.arange(len(comp));
plt.figure(figsize=(9,5))
plt.plot(x,comp.Learned_RMSE,marker='o',label='Learned-edge graph')
plt.plot(x,comp.Direct_RMSE,marker='s',label='Source-validated RBF25')
plt.xticks(x,[s.replace('STA.IT','') for s in comp.Group],rotation=45,ha='right')
plt.ylabel(r'RMSE ($\mu$g m$^{-3}$)');plt.xlabel('Locked station');plt.title('Station-level external-test heterogeneity');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_station_rmse_comparison.png',dpi=300);plt.close()

# 5. Representative time series and 90% interval for high-variability site.
sid='STA.IT1590A';z=sel[sel.StationID==sid].sort_values('Date')
plt.figure(figsize=(11,5))
plt.fill_between(z.Date,z.Lower90,z.Upper90,alpha=.22,label='90% predictive interval')
plt.plot(z.Date,z.Observed,linewidth=1,label='Observed')
plt.plot(z.Date,z.Mean,linewidth=1,label='Predicted mean')
plt.ylabel(r'PM$_{2.5}$ ($\mu$g m$^{-3}$)');plt.xlabel('Date');plt.title(f'External-test predictions at {sid}');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_timeseries_STA_IT1590A.png',dpi=300);plt.close()

# 6. PIT histogram.
pit=student_t.cdf((sel.Observed-sel.Mean)/sel.Scale,df=sel.df)
plt.figure(figsize=(7,4.5))
plt.hist(pit,bins=np.linspace(0,1,11),density=True,edgecolor='black')
plt.axhline(1.0,linewidth=1.2,label='Uniform density')
plt.xlabel('Probability integral transform');plt.ylabel('Density');plt.title('PIT diagnostic for the calibrated Student-t predictions');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_pit_histogram.png',dpi=300);plt.close()

# 7. Seasonal RMSE comparison.
season=pd.read_csv(OUT/'final_group_comparison_direct.csv');season=season[season.GroupVariable=='Season']
season_order=['Winter','Spring','Summer','Autumn'];season=season.set_index('Group').loc[season_order].reset_index();x=np.arange(4)
plt.figure(figsize=(7,4.5))
plt.plot(x,season.Learned_RMSE,marker='o',label='Learned-edge graph')
plt.plot(x,season.Direct_RMSE,marker='s',label='Source-validated RBF25')
plt.xticks(x,season.Group);plt.ylabel(r'RMSE ($\mu$g m$^{-3}$)');plt.title('Seasonal external-test performance');plt.legend();plt.tight_layout();plt.savefig(FIG/'fig_po_season_rmse.png',dpi=300);plt.close()
print('\n'.join(str(p) for p in sorted(FIG.glob('*.png'))))
