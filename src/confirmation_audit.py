"""Frozen revision 4 experiments on recovered, previously studied feature caches.
No new image encoders are run. Repetitions are independent CONDITIONAL ON THE
FIXED CACHE; their roles/queries must not be treated as independent units.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time,platform
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.linalg import solve
from transport_audit import *

PROTOCOL={
 'version':'calibration_count_new_conditions_v4_20260919',
 'origin':'new randomized analysis on three previously unused conditions from the frozen confirmation protocol',
 'catalog':'all 10 CIFAR10, 101 Food101 and 100 FGVCAircraft labels; no retained/bank deletion',
 'temperature_fitting_rows':256,
 'temperature_rule':'first 256 IDs by SHA256(v4_confirm_temperature + dataset + ID); fit log temperature by bounded scalar NLL minimization, tau in [.25,8]',
 'temperature_arms':['native','independently_fitted'],
 'calibration_ratios':[2,6,20],
 'pseudocount':1.0,'alphas':[.1,.05],
 'role_audit_repetitions':2048,'multicycle_role_repetitions':512,
 'complete_set_repetitions':128,'queries_per_complete_episode':16,
 'cycles':[1,3],
 'sampling':'IID with replacement from the empirical cache after excluding temperature fitting rows; duplicates retained',
 'unit':'one complete independently generated bag or calibration plus query episode',
 'confidence':'simultaneous empirical Bernstein intervals on bag averages; simple Hoeffding intervals also retained',
 'negative_outcomes':'retain every cache, arm, ratio, method and numerical failure; no rescouting',
 'interpretation':'conditional cache evidence, not new environments, independent images or human replication',
 'complete_methods':['raw_lac','raw_randomized_aps','split_ridge_lac','empirical_T1','uniform_T1','independent_T1','augmented_T1','empirical_T3','uniform_T3','independent_T3','augmented_T3'],
 'split_ridge':'unit normalized cached image features; squared one hot loss, coefficient penalty 1, no intercept, temperature .1, first n/2 labels fit, remainder calibrate',
 'development_status':'Protocol fixed before any encoder predictions on these three datasets. Analysis logic was tested on older EuroSAT/Pets caches. This is not new field acquisition.',
 'primary_endpoint':'one cycle mean lost-role fraction and empirical coverage relative to nominal; native and fitted temperatures separately; no predeclared minimum success fraction',
 'screening':'finite cycle NLL bound T log(1+1/(c_h+1)); exact candidate fallback, inclusive rank, compare every mask to unscreened reference'
}

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stable_seed(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8],'little')
def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def eb_interval(a,delta,lo=0.,hi=1.):
    a=np.asarray(a,dtype=float);R=len(a);r=hi-lo
    rad=np.sqrt(2*np.var(a,ddof=1)*np.log(4/delta)/R)+7*r*np.log(4/delta)/(3*(R-1))
    hoe=r*np.sqrt(np.log(2/delta)/(2*R))
    return max(lo,float(a.mean()-rad)),min(hi,float(a.mean()+rad)),rad,hoe

def mask_for_probs(p,y,alphas):
    n=len(y);s=-p[np.arange(n),y];res={}
    for a in alphas:
        k=rank_index(n,a);q=np.partition(s,k-1)[k-1] if k<=n else np.inf
        res[str(a)]=-p[n:]<=q
    return res

def aps_masks(L,y,alphas,rng):
    p=np.exp(L-logsumexp(L,axis=1,keepdims=True))
    order=np.argsort(-p,axis=1,kind='stable');srt=np.take_along_axis(p,order,axis=1)
    sc=np.cumsum(srt,axis=1)-srt+rng.random((len(L),1))*srt
    inv=np.argsort(order,axis=1);sc=np.take_along_axis(sc,inv,axis=1)
    n=len(y);cal=sc[np.arange(n),y];out={}
    for a in alphas:
        k=rank_index(n,a);q=np.partition(cal,k-1)[k-1] if k<=n else np.inf
        out[str(a)]=sc[n:]<=q
    return out

def run(cachedir,out,quick=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=sorted(Path(cachedir).glob('*.npz'))
    if len(paths)!=9:raise ValueError(f'Expected nine frozen new-condition VLM caches, got {len(paths)}')
    proto=PROTOCOL.copy()
    if quick:proto.update(role_audit_repetitions=8,multicycle_role_repetitions=4,complete_set_repetitions=2,origin='contract smoke test, not scientific new-condition confirmation')
    freeze={'protocol':proto,'inputs':{p.name:digest(p) for p in paths},
            'code':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
            'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'environment':{'python':sys.version,'platform':platform.platform(),'cpu_affinity':len(os.sched_getaffinity(0))}}
    (out/'FREEZE.json').write_text(json.dumps(freeze,indent=2))
    rows=[];setrows=[];summ=[];fits=[];checks=[];allidx={};allmask={};start=time.perf_counter()
    alph=proto['alphas'];roleR=proto['role_audit_repetitions'];multiR=proto['multicycle_role_repetitions']
    setR=proto['complete_set_repetitions'];M=proto['queries_per_complete_episode']
    delta=.05/(9*2*3*2*2*3)
    for path in paths:
        cellstart=time.perf_counter();cache=np.load(path,allow_pickle=False)
        L=cache['logits'].astype(float);y=cache['labels'];X=cache['image_features'].astype(float)
        X/=np.linalg.norm(X,axis=1,keepdims=True)
        ids=cache['image_ids'];dataset=path.stem.split('__')[-1];K=L.shape[1]
        order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('v4_confirm_temperature:'+dataset+':'+str(ids[i])).encode()).digest())
        ti=np.array(order[:256]);pool=np.array(order[256:]);allidx[path.stem+'_temperature']=ti
        def objective(logtau):
            ll=L[ti]/np.exp(logtau)
            return float(np.mean(logsumexp(ll,axis=1)-ll[np.arange(len(ti)),y[ti]]))
        fit=minimize_scalar(objective,bounds=(np.log(.25),np.log(8)),method='bounded',options={'xatol':1e-9})
        if not fit.success:raise RuntimeError('Temperature fit failed')
        fitted=float(np.exp(fit.x));fits.append({'cache':path.stem,'native':1.,'fitted':fitted,
                    'fit_nll_native':objective(0),'fit_nll_fitted':objective(fit.x),'fit_labels':256,'pool_size':len(pool)})
        for ratio in proto['calibration_ratios']:
            n=int(ratio*K)
            rng=np.random.default_rng(stable_seed('v4confirm_roles:'+dataset+':'+str(n)))
            bagidx=pool[rng.integers(len(pool),size=(roleR,n+1))];allidx[path.stem+f'_role_{n}']=bagidx
            erng=np.random.default_rng(stable_seed('v4confirm_sets:'+dataset+':'+str(n)))
            episodeidx=pool[erng.integers(len(pool),size=(setR,n+M))];allidx[path.stem+f'_sets_{n}']=episodeidx
            for arm,tau in [('native',1.),('independently_fitted',fitted)]:
                vals={(T,a,key):[] for T in (1,3) for a in alph for key in ('coverage','augmented_coverage','erosion')}
                for r,ix in enumerate(bagidx):
                    for T in (1,3):
                        if T==3 and r>=multiR:continue
                        ans=bag_audit(L[ix]/tau,y[ix],alph,pseudocount=1.,cycles=T)
                        for a in alph:
                            v=ans[str(a)];ero=(v['lost_roles']-v['gained_roles'])/v['roles']
                            rows.append({'cache':path.stem,'arm':arm,'tau':tau,'n':n,'K':K,'ratio':ratio,'cycles':T,'alpha':a,'replicate':r,**v,'erosion':ero})
                            for key in ('coverage','augmented_coverage'):vals[T,a,key].append(v[key])
                            vals[T,a,'erosion'].append(ero)
                            if T==1 and v['gained_roles']!=0:raise AssertionError('One cycle nesting failed')
                for (T,a,key),v in vals.items():
                    lo,hi,rad,hoe=eb_interval(v,delta,lo=(-1 if key=='erosion' and T!=1 else 0),hi=1)
                    summ.append({'cache':path.stem,'arm':arm,'tau':tau,'n':n,'K':K,'ratio':ratio,'cycles':T,'alpha':a,
                        'endpoint':key,'mean':float(np.mean(v)),'lower':lo,'upper':hi,'replicates':len(v),'variance':float(np.var(v,ddof=1)),
                        'eb_radius':rad,'hoeffding_radius':hoe,'delta':delta})
                for r,ix in enumerate(episodeidx):
                    ll=L[ix]/tau;yy=y[ix];cal=yy[:n];test=yy[n:];nf=n//2
                    p=np.exp(ll-logsumexp(ll,axis=1,keepdims=True));methods={'raw_lac':mask_for_probs(p,cal,alph)}
                    arng=np.random.default_rng(stable_seed(f'v4aps:{dataset}:{n}:{r}'))
                    methods['raw_randomized_aps']=aps_masks(ll,cal,alph,arng)
                    Xf=X[ix[:nf]];Xt=X[ix[nf:]];H=np.eye(K)[cal[:nf]]
                    coef=solve(Xf@Xf.T+np.eye(nf),H,assume_a='pos',check_finite=False)
                    rl=(Xt@Xf.T)@coef/.1;rp=np.exp(rl-logsumexp(rl,axis=1,keepdims=True))
                    methods['split_ridge_lac']=mask_for_probs(rp,cal[nf:],alph)
                    for T in (1,3):
                        d=np.bincount(cal,minlength=K)+1
                        methods[f'empirical_T{T}']=mask_for_probs(transport(ll,d,T),cal,alph)
                        methods[f'uniform_T{T}']=mask_for_probs(transport(ll,np.ones(K),T),cal,alph)
                        df=np.bincount(cal[:nf],minlength=K)+1
                        methods[f'independent_T{T}']=mask_for_probs(transport(ll[nf:],df,T),cal[nf:],alph)
                        st=time.perf_counter()
                        if T==1:
                            full=one_cycle_sets(ll,cal,alph)
                            if r<2:
                                check,_,_=prediction_sets(ll,cal,alph,cycles=1)
                                for a in alph:assert np.array_equal(full[str(a)],check[str(a)])
                        else:
                            full,_=augmented_probability_sets(ll,cal,alph,cycles=T)
                            if r<2:
                                check,_,_=prediction_sets(ll,cal,alph,cycles=T)
                                for a in alph:assert np.array_equal(full[str(a)],check[str(a)])
                        fulltime=time.perf_counter()-st;methods[f'augmented_T{T}']=full
                        if T==3:
                            st=time.perf_counter();scr,stat=augmented_probability_sets(ll,cal,alph,cycles=T,screen=True)
                            screentime=time.perf_counter()-st
                            for a in alph:assert np.array_equal(full[str(a)],scr[str(a)])
                            checks.append({'cache':path.stem,'arm':arm,'n':n,'replicate':r,'full_time':fulltime,'screen_time':screentime,**stat})
                    for method,mask in methods.items():
                        for a in alph:
                            z=mask[str(a)];sz=z.sum(axis=1);cv=z[np.arange(M),test]
                            setrows.append({'cache':path.stem,'arm':arm,'tau':tau,'n':n,'K':K,'ratio':ratio,'replicate':r,
                                      'method':method,'alpha':a,'coverage':float(cv.mean()),'mean_size':float(sz.mean()),
                                      'empty_fraction':float(np.mean(sz==0)),'singleton_fraction':float(np.mean(sz==1)),
                                      'full_fraction':float(np.mean(sz==K))})
                            key=f'{path.stem}__{arm}__{n}__{method}__{a}'
                            allmask.setdefault(key,[]).append(np.packbits(z,axis=None))
                    for a in alph:assert not np.any(methods['empirical_T1'][str(a)] & ~methods['augmented_T1'][str(a)])
                print(f'Finished {path.stem} n={n} {arm} elapsed={time.perf_counter()-start:.1f}s',flush=True)
                save_csv(out/'role_rows.csv',rows);save_csv(out/'role_summary.csv',summ)
                save_csv(out/'set_rows.csv',setrows);save_csv(out/'screening.csv',checks);save_csv(out/'temperature.csv',fits)
        print(f'Cache complete seconds={time.perf_counter()-cellstart:.1f}',flush=True)
    np.savez_compressed(out/'sampling_indices.npz',**allidx)
    np.savez_compressed(out/'all_complete_masks.npz',**{k:np.stack(v) for k,v in allmask.items()})
    summary={'elapsed_seconds':time.perf_counter()-start,'role_rows':len(rows),'complete_set_rows':len(setrows),
              'screening_comparisons':len(checks),'cache_count':len(paths),'protocol':proto,
              'complete':True,'new_encoder_runs_this_analysis':0,'gpu_runs_this_analysis':0,'encoder_provenance':'parent EXTRACTION_COMPLETE.json and verified cache receipts'}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--quick',action='store_true')
    a=p.parse_args();run(a.cache,a.out,a.quick)
