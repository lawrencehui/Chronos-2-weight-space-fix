import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
taus = np.arange(1,10)/10
def crps(y,Q):
    ok=~np.isnan(y); y=y[ok]; Q=Q[ok]
    if len(y)==0: return np.nan
    return float(np.mean((2*np.abs((y[:,None]-Q)*((y[:,None]<=Q)-taus))).sum(0)/np.abs(y).sum()))
segs=[(0,1),(1,3),(3,8),(8,16),(16,32),(32,57)]
out={s:[] for s in segs}
gains={s:{} for s in segs}
for cfg,v in d.items():
    Q=B[cfg]; y=v['label'].astype(float); pl=v['pl']
    locs=[];scs=[]
    for c in v['ctx']:
        c=c[-2048:].astype(np.float64)
        l=np.nanmean(c) if np.any(~np.isnan(c)) else 0.; s=np.sqrt(np.nanmean((c-l)**2)) if np.any(~np.isnan(c)) else 1.
        s = s if s>0 else 1e-5; locs.append(l); scs.append(s)
    locs=np.array(locs)[:,None,None]; scs=np.array(scs)[:,None,None]
    Qn=np.arcsinh((Q-locs)/scs); med=Qn[:,:,4:5]
    patch=np.arange(pl)//16
    for s in segs:
        sel=(patch>=s[0])&(patch<s[1])
        if not sel.any(): continue
        yy=y[:,sel]; QQ=Q[:,sel]
        ok=~np.isnan(yy)
        out[s].append([np.mean(yy[ok]<QQ[...,0][ok]),np.mean(yy[ok]<QQ[...,4][ok]),np.mean(yy[ok]<QQ[...,8][ok])])
        b=crps(yy.reshape(-1),QQ.reshape(-1,9))
        for sc in [0.85,0.95,1.05,1.15]:
            Qs=np.sinh(med[:,sel]+sc*(Qn[:,sel]-med[:,sel]))*scs+locs
            gains[s].setdefault(sc,[]).append(np.log(crps(yy.reshape(-1),Qs.reshape(-1,9))/b))
for s in segs:
    a=np.array(out[s]); print(s, len(a), 'median cov', np.round(np.median(a,0),3), 'mean', np.round(a.mean(0),3), ' spread geomean:', {k:round(float(np.exp(np.nanmean(v))),4) for k,v in gains[s].items()})
