"""Fast consistency check; does not refit model or recalibrate results."""
from pathlib import Path
import json,csv,hashlib,argparse,math
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path)
args=parser.parse_args()
files=['README.md','paper/CIRG_EMA_Manuscript.tex','paper/CIRG_EMA_Manuscript.pdf','config/study_design.json','src/prepare_data.py','scripts/run_full_real_data.py','results/archived_article_outputs/final_point_predictions_long.csv','results/archived_article_outputs/final_probabilistic_predictions.csv','results/archived_article_outputs/final_direct_kernel_probabilistic_predictions.csv']
missing=[p for p in files if not (root/p).is_file()]
assert not missing,f'Missing project files: {missing}'
design=json.loads((root/'config/study_design.json').read_text());folds=[s for sub in design['folds'].values() for s in sub]; locked=design['locked_stations'];assert len(folds)==len(set(folds))==40 and len(locked)==len(set(locked))==10 and not (set(folds)&set(locked))
if args.data:
 p=args.data; assert p.exists(),p
 counts={};stations=set();n=0;target=0;devtrain=0
 with p.open(newline='',encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):
   n+=1;stations.add(r['StationID']);y=int(r['Year']); counts[y]=counts.get(y,0)+1
   if y==2023 and r['StationID'] in locked:target+=1
   if 2018<=y<=2021 and r['StationID'] in folds:devtrain+=1
 assert (n,len(stations),target,devtrain)==(101913,50,3237,55444),(n,len(stations),target,devtrain)
 print('Processed data verified:',n,'rows, 50 stations, 3237 locked-2023, 55444 development-training')
print('Repository structure and fixed design verified.')
