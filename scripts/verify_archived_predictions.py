"""Independent standard-library check of archived row-level locked-test outputs.
Usage: python verify_archived_predictions.py --archive frozen_repository/results/archived_article_outputs
"""
import argparse,csv,math,statistics,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);args=ap.parse_args();P=args.archive

def read(p):
 with (P/p).open(newline='') as f:return list(csv.DictReader(f))
def calc(R):
 n=len(R);obs=[float(r['Observed']) for r in R];mean=[float(r['Mean']) for r in R];e=[m-y for m,y in zip(mean,obs)]
 d={'n':n,'RMSE':math.sqrt(statistics.mean(t*t for t in e)),'MAE':statistics.mean(abs(t) for t in e),'Bias':statistics.mean(e)}
 if 'CRPS_row' in R[0]:
  d.update(CRPS=statistics.mean(float(r['CRPS_row']) for r in R),Coverage90=statistics.mean(float(r['Lower90'])<=float(r['Observed'])<=float(r['Upper90']) for r in R),Width90=statistics.mean(float(r['Upper90'])-float(r['Lower90']) for r in R))
 return d
r={}
for file in ['final_probabilistic_predictions.csv','final_direct_kernel_probabilistic_predictions.csv']:
 R=read(file);r[file]=calc(R)
assert len(read('final_probabilistic_predictions.csv'))==len(read('final_direct_kernel_probabilistic_predictions.csv'))==3237
assert abs(r['final_probabilistic_predictions.csv']['RMSE']-5.8973274370898)<1e-8
assert abs(r['final_direct_kernel_probabilistic_predictions.csv']['RMSE']-5.682746682624017)<1e-8
print(json.dumps(r,indent=2))
