"""Additional full conformal ridge and randomized RAPS on every saved episode.
These methods are declared before the new-condition predictions.
The original eleven-method experiment remains unchanged.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np,json,hashlib,time,sys,csv
from scipy.linalg import solve
from scipy.special import logsumexp
from transport_audit import rank_index

def full_ridge_masks(X,y,alphas=(.1,.05),temperature=.1):
 X=np.asarray(X,dtype=float);y=np.asarray(y,dtype=int);n=len(y);K=int(full_ridge_masks.K)
 V=X[:n];Q=X[n:];H=np.eye(K)[y];R=V@V.T+np.eye(n)
 dot=V@Q.T
 sol=solve(R,np.concatenate([H,dot],axis=1),assume_a='pos',check_finite=False)
 coef=sol[:,:K];b=sol[:,K:];Bc=H-coef;Bq=dot.T@coef
 lev=np.sum(Q*Q,axis=1)-np.sum(dot*b,axis=0)
 if np.any(lev<-1e-9):raise AssertionError('Negative leverage exceeds rounding tolerance')
 ans={str(a):np.empty((len(Q),K),bool) for a in alphas}
 for j in range(len(Q)):
  ac=b[:,j]/(1+lev[j]);aq=lev[j]/(1+lev[j])
  lc=(Bc-ac[:,None]*Bq[j])/temperature
  lq=((1-aq)*Bq[j])/temperature
  # Direct labelwise vector updates; no cancellation in probability normalization.
  for h in range(K):
   ch=lc.copy();ch[:,h]+=ac/temperature
   cal=logsumexp(ch,axis=1)-ch[np.arange(n),y]
   qh=lq.copy();qh[h]+=aq/temperature
   test=logsumexp(qh)-qh[h]
   for a in alphas:
    k=rank_index(n,a);cut=np.partition(cal,k-1)[k-1] if k<=n else np.inf
    ans[str(a)][j,h]=test<=cut
 return ans

def direct_ridge(X,y,q,h,K):
 V=np.vstack([X,q]);Y=np.eye(K)[np.r_[y,h]]
 W=solve(V.T@V+np.eye(V.shape[1]),V.T@Y,assume_a='pos')
 L=V@W/.1;return logsumexp(L,axis=1)-L[np.arange(len(V)),np.r_[y,h]]

def tests():
 rng=np.random.default_rng(381049);checks=0
 for K in (2,5,10):
  for n in (8,17):
   X=rng.normal(size=(n+4,6));X/=np.linalg.norm(X,axis=1,keepdims=True);y=rng.integers(K,size=n)
   full_ridge_masks.K=K;m=full_ridge_masks(X,y)
   for j in range(4):
    for h in range(K):
     s=direct_ridge(X[:n],y,X[n+j],h,K)
     for a in (.1,.05):
      k=rank_index(n,a);v=(s[-1]<=np.partition(s[:-1],k-1)[k-1]) if k<=n else True
      assert v==m[str(a)][j,h];checks+=1
 return checks

def run(root,out):
 root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
 summaryroot=root/'results/natural';cache=root/'cache'
 freeze={'purpose':__doc__,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'frozen_choices':{'methods':['full_ridge_lac','raw_randomized_raps'],'ridge_penalty':1,'ridge_temperature':.1,'raps_penalty':.01,'raps_kreg':3,'alphas':[.1,.05],'all_saved_episodes':True}}
 (out/'FREEZE.json').write_text(json.dumps(freeze,indent=2));testn=tests();start=time.perf_counter()
 # Explicitly wait for the completed, unchanged input experiment.
 while not (summaryroot/'SUMMARY.json').exists():time.sleep(3)
 ixall=np.load(summaryroot/'sampling_indices.npz',allow_pickle=False)
 import pandas as pd
 temps=pd.read_csv(summaryroot/'temperature.csv').set_index('cache')
 rows=[];masks={}
 def seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little')
 for path in sorted(cache.glob('*.npz')):
  z=np.load(path,allow_pickle=False);L=z['logits'].astype(float);y=z['labels'];X=z['image_features'].astype(float);X/=np.linalg.norm(X,axis=1,keepdims=True)
  K=L.shape[1];ds=path.stem.split('__')[-1];full_ridge_masks.K=K
  for ratio in (2,6,20):
   n=ratio*K;ixs=ixall[path.stem+f'_sets_{n}']
   for r,ix in enumerate(ixs):
    ridge=full_ridge_masks(X[ix],y[ix[:n]])
    for arm,tau in [('native',1.),('independently_fitted',float(temps.loc[path.stem,'fitted']))]:
     ll=L[ix]/tau;yy=y[ix];M=len(ix)-n
     p=np.exp(ll-logsumexp(ll,axis=1,keepdims=True));order=np.argsort(-p,axis=1,kind='stable');pp=np.take_along_axis(p,order,axis=1)
     rrng=np.random.default_rng(seed(f'v4aps:{ds}:{n}:{r}'))
     s=np.cumsum(pp,axis=1)-pp+rrng.random((len(ix),1))*pp+.01*np.maximum(np.arange(1,K+1)-3,0)
     s=np.take_along_axis(s,np.argsort(order,axis=1),axis=1);cal=s[np.arange(n),yy[:n]]
     raps={}
     for a in (.1,.05):
      k=rank_index(n,a);cut=np.partition(cal,k-1)[k-1] if k<=n else np.inf;raps[str(a)]=s[n:]<=cut
     for meth,m in [('full_ridge_lac',ridge),('raw_randomized_raps',raps)]:
      for a in (.1,.05):
       zz=m[str(a)];sz=zz.sum(axis=1)
       rows.append({'cache':path.stem,'arm':arm,'tau':tau,'n':n,'K':K,'ratio':ratio,'replicate':r,'method':meth,'alpha':a,'coverage':float(zz[np.arange(M),yy[n:]].mean()),
         'mean_size':float(sz.mean()),'empty_fraction':float((sz==0).mean()),'singleton_fraction':float((sz==1).mean()),'full_fraction':float((sz==K).mean())})
       key=f'{path.stem}__{arm}__{n}__{meth}__{a}';masks.setdefault(key,[]).append(np.packbits(zz))
   print(path.stem,n,'completed extra baselines',flush=True)
   pd.DataFrame(rows).to_csv(out/'set_rows.csv',index=False)
 np.savez_compressed(out/'all_complete_masks.npz',**{k:np.stack(v) for k,v in masks.items()})
 summ={'complete':True,'rows':len(rows),'direct_refit_checks':testn,'elapsed_seconds':time.perf_counter()-start}
 (out/'SUMMARY.json').write_text(json.dumps(summ,indent=2));print(summ)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/data/Calibration_Count_v4');p.add_argument('--out',default='/mnt/data/Calibration_Count_v4/results/extra_baselines');p.add_argument('--test-only',action='store_true');a=p.parse_args()
 if a.test_only: print(tests())
 else:run(a.root,a.out)
