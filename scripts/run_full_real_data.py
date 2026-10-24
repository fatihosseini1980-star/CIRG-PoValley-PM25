"""Execute the frozen real-data pipeline in an isolated, relocatable workspace."""
from __future__ import annotations
import argparse, os, shutil, subprocess, sys
from pathlib import Path

def run(cmd,env=None):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.run(cmd,check=True,env=env)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True); ap.add_argument('--workspace',type=Path,default=Path('work')); args=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]; work=args.workspace.resolve(); work.mkdir(parents=True,exist_ok=True)
    target=work/'Po_Valley_PM25_Final_2018_2023_Observed(2).csv'; shutil.copy2(args.data,target)
    patched=work/'patched_scripts'; patched.mkdir(exist_ok=True)
    names=['po_stage2_clim_resgraph_onefold.py','po_stage2_learned_edge_fast_onefold.py','po_stage2_uncertainty_learned.py','po_stage3_final_locked_test.py','po_stage3_direct_baseline.py']
    for name in names:
        text=(repo/'legacy/frozen_code'/name).read_text()
        text=text.replace('/mnt/data',str(work))
        # Stage-2 combined predictions already contain deployment covariates in
        # current repository builds. Avoid duplicate *_x/*_y columns at Stage 3.
        if name == 'po_stage3_final_locked_test.py':
            old_merge = "internal=internal.merge(data,on=['StationID','Date'],how='left',validate='one_to_one')"
            new_merge = ("missing_cov=[c for c in cols if c not in {'StationID','Date'} and c not in internal.columns]\n"
                         "if missing_cov:\n"
                         "    internal=internal.merge(data[['StationID','Date']+missing_cov],on=['StationID','Date'],how='left',validate='one_to_one')")
            text=text.replace(old_merge,new_merge)
        (patched/name).write_text(text)
    for fold in range(1,6):
        env=os.environ.copy(); env['FOLD_ONLY']=str(fold); run([sys.executable,patched/'po_stage2_clim_resgraph_onefold.py'],env); run([sys.executable,patched/'po_stage2_learned_edge_fast_onefold.py'],env)
    run([sys.executable,patched/'po_stage2_uncertainty_learned.py']); run([sys.executable,patched/'po_stage3_final_locked_test.py']); run([sys.executable,patched/'po_stage3_direct_baseline.py'])
    run([sys.executable,repo/'src/direct_uncertainty.py','--root',work]); run([sys.executable,repo/'src/event_analysis.py','--root',work]); run([sys.executable,repo/'src/moving_block_bootstrap.py','--root',work])
    print('Completed:',work)
if __name__=='__main__': main()
