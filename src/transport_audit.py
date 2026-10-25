"""Finite sample audits and exact candidate augmentation for count-dependent transport.

Independent implementation for revision 4. Scores are -log probabilities.
Mathematical statements concern exact arithmetic; float comparisons use explicitly
reported tolerances only in diagnostic checks, never to declare an unsafe set valid.
"""
from __future__ import annotations
from fractions import Fraction
from math import ceil
import numpy as np
from scipy.special import logsumexp


def rank_index(n: int, alpha: float) -> int:
    a = Fraction(str(alpha))
    v = (n + 1) * (1 - a)
    return (v.numerator + v.denominator - 1) // v.denominator


def _validate(logits: np.ndarray, weights: np.ndarray, cycles: int):
    L = np.asarray(logits, dtype=np.float64)
    d = np.asarray(weights, dtype=np.float64)
    if L.ndim != 2 or d.shape != (L.shape[1],) or not np.isfinite(L).all():
        raise ValueError('Finite example-by-class logits and one weight per class required.')
    if not np.isfinite(d).all() or (d < 0).any() or d.sum() <= 0 or cycles < 1:
        raise ValueError('Weights must be nonnegative with positive total; cycles >= 1.')
    return L, d


def log_transport(logits: np.ndarray, weights: np.ndarray, cycles: int = 1) -> np.ndarray:
    L, d = _validate(logits, weights, cycles)
    support = d > 0
    q = L[:, support].copy()
    ld = np.log(d[support])
    for _ in range(cycles):
        q += ld - logsumexp(q, axis=0)
        q -= logsumexp(q, axis=1, keepdims=True)
    out = np.full(L.shape, -np.inf)
    out[:, support] = q
    return out


def transport(logits: np.ndarray, weights: np.ndarray, cycles: int = 1) -> np.ndarray:
    L, d = _validate(logits, weights, cycles)
    if np.ptp(L) > 650:
        return np.exp(log_transport(L,d,cycles))
    q = np.exp(L - L.max())
    support = d > 0
    q[:,~support] = 0
    for _ in range(cycles):
        den = q.sum(axis=0)
        q *= np.divide(d,den,out=np.zeros_like(d),where=support)
        q /= q.sum(axis=1,keepdims=True)
    return q


def batch_transport(logits: np.ndarray, weights: np.ndarray, cycles: int) -> np.ndarray:
    """H independent marginal vectors, same pooled features; H x P x K output."""
    L = np.asarray(logits, dtype=np.float64)
    D = np.atleast_2d(np.asarray(weights,dtype=np.float64))
    if np.ptp(L)>650:
        return np.stack([np.exp(log_transport(L,d,cycles)) for d in D])
    q=np.broadcast_to(np.exp(L-L.max()),(len(D),*L.shape)).copy()
    for _ in range(cycles):
        den=q.sum(axis=1)
        q*=np.divide(D,den,out=np.zeros_like(D),where=den>0)[:,None,:]
        q/=q.sum(axis=2,keepdims=True)
    return q


def leave_one_out_accept(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Inclusive kth-of-other-scores rule via strict lower rank, including ties."""
    s=np.asarray(scores,dtype=float)
    k=rank_index(len(s)-1,alpha)
    return np.searchsorted(np.sort(s),s,side='left') < k


def bag_audit(logits: np.ndarray, labels: np.ndarray, alphas=(.1,.05),
              pseudocount: float=1., cycles: int=1) -> dict:
    """Integrate every possible target role in one exchangeable labeled bag.

    This is an OFFLINE labeled audit, not a deployable certificate from test
    features alone. One independent bag, not its individual roles, is the unit.
    """
    L=np.asarray(logits,dtype=float); y=np.asarray(labels,dtype=int)
    P,K=L.shape
    if y.shape!=(P,) or np.any((y<0)|(y>=K)) or pseudocount<=0:
        raise ValueError('Matching catalog labels and a positive pseudocount required.')
    d=np.bincount(y,minlength=K)+pseudocount
    full=transport(L,d,cycles)
    fulltrue=full[np.arange(P),y]
    fullscores=-np.log(fulltrue)
    present=np.unique(y)
    empirical={a:np.zeros(P,dtype=bool) for a in alphas}
    fullaccept={a:leave_one_out_accept(fullscores,a) for a in alphas}
    max_formula_error=0.
    # Remove one label of the held-out class. The feature pool does not change.
    if cycles==1:
        for h in present:
            ratio=full[:,h]/d[h]
            true=fulltrue/(1-ratio)
            mask=y==h
            true[mask]*=(d[h]-1)/d[h]
            s=-np.log(true)
            for a in alphas:
                empirical[a][mask]=leave_one_out_accept(s,a)[mask]
    else:
        D=np.broadcast_to(d,(len(present),K)).copy()
        D[np.arange(len(present)),present]-=1
        Q=batch_transport(L,D,cycles)
        for j,h in enumerate(present):
            s=-np.log(Q[j,np.arange(P),y]);mask=y==h
            for a in alphas:
                empirical[a][mask]=leave_one_out_accept(s,a)[mask]
    result={}
    for a in alphas:
        fa=fullaccept[a];ea=empirical[a]
        loss=int(np.count_nonzero(fa & ~ea))
        gain=int(np.count_nonzero(~fa & ea))
        result[str(a)]={'coverage':float(ea.mean()),'augmented_coverage':float(fa.mean()),
                        'lost_roles':loss,'gained_roles':gain,'roles':P,
                        'rounding_target':rank_index(P-1,a)/P,
                        'ties':P-len(np.unique(fullscores))}
    return result


def prediction_sets(logits: np.ndarray, calibration_labels: np.ndarray, alphas=(.1,.05),
                    cycles=1, pseudocount=1., method='augmented', independent_counts=None,
                    screening=False) -> tuple[dict,np.ndarray,dict]:
    """Complete sets for all query rows, sharing each candidate map across queries.

    The independent method expects a disjoint fitting count vector; the caller
    supplies only final calibration and query features in logits.
    """
    L=np.asarray(logits,dtype=float);y=np.asarray(calibration_labels,dtype=int)
    n=len(y);P,K=L.shape; M=P-n
    if M<1 or n<1:raise ValueError('At least one calibration and query observation required.')
    if method=='uniform': d=np.ones(K)
    elif method=='independent':
        if independent_counts is None:raise ValueError('Disjoint fitting counts required.')
        d=np.asarray(independent_counts,dtype=float)+pseudocount
    else:d=np.bincount(y,minlength=K)+pseudocount
    lp=log_transport(L,d,cycles); s=-lp[np.arange(n),y]
    Q={a:float(np.partition(s,rank_index(n,a)-1)[rank_index(n,a)-1])
       if rank_index(n,a)<=n else np.inf for a in alphas}
    masks={str(a):(-lp[n:]<=Q[a]) for a in alphas}
    stats={'candidate_fits':0,'candidates':K,'screened_pairs':0,'total_pairs':M*K*len(alphas),
           'screened_candidates':0}
    if method!='augmented':return masks,lp,stats
    masks={str(a):np.zeros((M,K),dtype=bool) for a in alphas}
    pending=np.ones((len(alphas),M,K),dtype=bool)
    if screening:
        # Span of the candidate log prior ratio; common prior normalizers cancel.
        gamma=cycles*np.log1p(1/d)
        for j,a in enumerate(alphas):
            margin=Q[a]+lp[n:]
            safein=margin>=2*gamma[None,:]+1e-12
            safeout=margin < -2*gamma[None,:]-1e-12
            masks[str(a)][safein]=True
            pending[j]=~(safein|safeout)
            stats['screened_pairs']+=int(np.count_nonzero(safein|safeout))
    for h in range(K):
        if not pending[:,:,h].any():
            stats['screened_candidates']+=1;continue
        dh=d.copy();dh[h]+=1
        lph=log_transport(L,dh,cycles)
        sh=-lph[np.arange(n),y]
        for j,a in enumerate(alphas):
            k=rank_index(n,a);qh=np.partition(sh,k-1)[k-1] if k<=n else np.inf
            exact=-lph[n:,h]<=qh
            m=pending[j,:,h]
            masks[str(a)][m,h]=exact[m]
        stats['candidate_fits']+=1
    return masks,lp,stats


def one_cycle_sets(logits: np.ndarray, calibration_labels: np.ndarray,
                   alphas=(.1,.05),pseudocount=1.) -> dict:
    """Exact O((n+M)K) candidate score construction for ONE complete cycle."""
    L=np.asarray(logits,dtype=float);y=np.asarray(calibration_labels,dtype=int)
    n=len(y);P,K=L.shape
    d=np.bincount(y,minlength=K)+pseudocount
    lp=log_transport(L,d,1)
    p=np.exp(lp); true=-lp[np.arange(n),y]
    delta=np.log1p(p/d[None,:])
    scal=true[:,None]+delta[:n]
    scal[np.arange(n),y]-=np.log1p(1/d[y])
    stest=-lp[n:]+delta[n:]-np.log1p(1/d)[None,:]
    out={}
    for a in alphas:
        k=rank_index(n,a)
        q=np.partition(scal,k-1,axis=0)[k-1] if k<=n else np.full(K,np.inf)
        out[str(a)]=stest<=q[None,:]
    return out


def augmented_probability_sets(logits, calibration_labels, alphas=(.1,.05), cycles=3,
                               pseudocount=1.,screen=False):
    """Vectorized positive-kernel implementation used for complete-set experiments.
    Returned masks are checked against separately written log-domain arithmetic.
    """
    L=np.asarray(logits,dtype=float);y=np.asarray(calibration_labels,dtype=int)
    n=len(y);P,K=L.shape;M=P-n
    d=np.bincount(y,minlength=K)+pseudocount
    p=transport(L,d,cycles)
    pending=np.ones((len(alphas),M,K),dtype=bool)
    masks={str(a):np.zeros((M,K),dtype=bool) for a in alphas}
    if screen:
        sc=-np.log(p[np.arange(n),y]);gam=cycles*np.log1p(1/d)
        for j,a in enumerate(alphas):
            k=rank_index(n,a);thr=np.partition(sc,k-1)[k-1] if k<=n else np.inf
            margin=thr+np.log(p[n:])
            si=margin>=2*gam+1e-12;so=margin < -2*gam-1e-12
            masks[str(a)][si]=True;pending[j]=~(si|so)
    active=np.flatnonzero(pending.any(axis=(0,1)))
    if len(active):
        D=np.broadcast_to(d,(len(active),K)).copy();D[np.arange(len(active)),active]+=1
        Q=batch_transport(L,D,cycles)
        for j,h in enumerate(active):
            sc=-Q[j,np.arange(n),y]
            for t,a in enumerate(alphas):
                k=rank_index(n,a);thr=np.partition(sc,k-1)[k-1] if k<=n else np.inf
                act=pending[t,:,h]
                masks[str(a)][act,h]=(-Q[j,n:,h]<=thr)[act]
    return masks,{'candidate_fits':len(active),'candidate_total':K,
                  'screened_pairs':int((~pending).sum()),'total_pairs':int(pending.size)}
