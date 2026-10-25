"""Describe unadapted classifier strength on the fixed held-back evaluation pools."""
from pathlib import Path
import argparse,numpy as np,pandas as pd

def run(root):
 root=Path(root);cache=root/'data/cache'
 if not cache.exists():cache=root/'cache'
 ids=np.load(root/'results/natural/sampling_indices.npz',allow_pickle=False);rows=[]
 for path in sorted(cache.glob('*.npz')):
  z=np.load(path,allow_pickle=False);L=z['logits'].astype(float);y=z['labels'];pool=np.ones(len(y),dtype=bool)
  pool[ids[path.stem+'_temperature']]=False;L=L[pool];y=y[pool]
  top=np.sort(L,axis=1)[:,-2:]
  rows.append(dict(cache=path.stem,rows=len(y),top_one_accuracy=float(np.mean(L.argmax(1)==y)),median_top_two_logit_margin=float(np.median(top[:,1]-top[:,0])),scope='fixed evaluation pool; positive scalar temperatures preserve top one prediction'))
 out=root/'results/summary';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'score_diagnostics.csv',index=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));run(p.parse_args().root)
