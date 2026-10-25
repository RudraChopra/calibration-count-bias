"""Transparent aggregation; no new outcome-based selection or threshold tuning.
All descriptive averages across cells are labelled as descriptive; paired episode
intervals treat whole calibration+query episodes, not queries, as units.
"""
from pathlib import Path
import argparse,json,numpy as np,pandas as pd
from confirmation_audit import eb_interval

def main(root):
 root=Path(root);out=root/'results/summary';out.mkdir(parents=True,exist_ok=True)
 receipts=[json.loads((root/x/'SUMMARY.json').read_text()) for x in ('results/natural','results/extra_baselines')]
 if not all(v.get('complete') for v in receipts): raise RuntimeError('Incomplete input study; no completed summary permitted')
 roles=pd.read_csv(root/'results/natural/role_summary.csv')
 d=pd.read_csv(root/'results/natural/set_rows.csv')
 extra=root/'results/extra_baselines/set_rows.csv'
 if extra.exists():d=pd.concat([d,pd.read_csv(extra)],ignore_index=True)
 cell=['cache','arm','n','K','ratio','alpha']
 means=d.groupby(cell+['method']).agg(coverage=('coverage','mean'),mean_size=('mean_size','mean'),empty=('empty_fraction','mean'),singletons=('singleton_fraction','mean'),full=('full_fraction','mean')).reset_index()
 means.to_csv(out/'all_complete_method_cells.csv',index=False)
 # Registered family: two augmented cycles, five strong controls, both nominal
 # levels in every cache/arm/ratio. Intervals computed after execution are labelled
 # additional descriptive uncertainty, not a newly predeclared discovery test.
 comparators=['raw_lac','raw_randomized_aps','raw_randomized_raps','split_ridge_lac','full_ridge_lac','uniform_T1','uniform_T3','independent_T1','independent_T3']
 candidates=['augmented_T1','augmented_T3'];pairs=[]
 grouped=list(d.groupby(cell));ntests=len(grouped)*len(comparators)*2*2;delta=.05/ntests
 for keys,dd in grouped:
  pivot_size=dd.pivot(index='replicate',columns='method',values='mean_size');pivot_cov=dd.pivot(index='replicate',columns='method',values='coverage')
  K=keys[3]
  for m in candidates:
   for b in comparators:
    if b not in pivot_size:continue
    for endpoint,piv,rng in [('size_saved',pivot_size,(-K,K)),('coverage_change',pivot_cov,(-1,1))]:
     a=(piv[b]-piv[m]).to_numpy() if endpoint=='size_saved' else (piv[m]-piv[b]).to_numpy()
     lo,hi,rad,hoe=eb_interval(a,delta,*rng)
     pairs.append({**dict(zip(cell,keys)),'candidate':m,'comparator':b,'endpoint':endpoint,'mean':float(a.mean()),'lower':lo,'upper':hi,'delta':delta,'episodes':len(a),
       'interpretation':'simultaneous conservative bound on episode mean conditional on fixed cache; prespecified complete-method comparison; no outcome-selected success criterion'})
 pd.DataFrame(pairs).to_csv(out/'all_paired_episode_intervals.csv',index=False)
 c=roles[roles.endpoint=='coverage'].copy();c['nominal']=1-c.alpha;c['below_nominal']=c.upper<c.nominal
 c['deficit_points']=100*(c.nominal-c['mean']);c.to_csv(out/'all_coverage_conditions.csv',index=False)
 e=roles[roles.endpoint=='erosion'];sc=pd.read_csv(root/'results/natural/screening.csv')
 r=pd.read_csv(root/'results/natural/role_rows.csv')
 primary=c[(c.cycles==1)&(c.arm=='native')&(c.alpha==.1)]
 range_tab=[]
 for (cy,arm,alpha),dd in c.groupby(['cycles','arm','alpha']):
  range_tab.append({'cycles':int(cy),'arm':arm,'alpha':alpha,'conditions':len(dd),'simultaneous_upper_below_nominal':int(dd.below_nominal.sum()),'min_coverage':dd['mean'].min(),'max_coverage':dd['mean'].max(),'max_deficit_points':dd.deficit_points.max()})
 summary={'complete':True,'primary_native_one_cycle_90':primary[['cache','n','ratio','mean','lower','upper','deficit_points','below_nominal']].to_dict('records'),
 'coverage_ranges':range_tab,'all_one_cycle_gained_roles':int(r.loc[r.cycles==1,'gained_roles'].sum()),'three_cycle_gained_roles':int(r.loc[r.cycles==3,'gained_roles'].sum()),
 'screened_pair_fraction':float(sc.screened_pairs.sum()/sc.total_pairs.sum()),'candidate_fit_fraction':float(sc.candidate_fits.sum()/sc.candidate_total.sum()),
 'screening_speed_median_ratio':float(np.median(sc.full_time/sc.screen_time)),
 'complete_set_rows':len(d),'complete_method_count':d.method.nunique(),
 'bag_evaluations_including_shared_model_arms':int(len(r)//2),
 'target_role_evaluations_including_shared_model_arms_and_levels':int(r.roles.sum()),
 'scope':'New-condition GPU feature caches, reused conditional episodes; no new field acquisition. Shared arms are not independent environments.'}
 (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/Calibration_Count_v4');a=p.parse_args();main(a.root)
