import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
cats={}
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]; pl=v['pl']; m=v['m']
    for i,x in enumerate(v['ctx']):
        x=x[-2048:].astype(float); y=Y[i]; ok=~np.isnan(y)
        if not ok.any(): continue
        xs=x[~np.isnan(x)]
        if len(xs)==0: cat='allnan'
        else:
            last16nan=np.mean(np.isnan(x[-16:]))
            nanfrac=np.mean(np.isnan(x))
            zf=np.mean(xs==0)
            if last16nan>0: cat='nan_in_last_patch'
            elif nanfrac>0.3: cat='mostly_nan'
            elif np.std(xs[-min(len(xs),256):])==0: cat='flat_recent'
            elif zf>0.3: cat='many_zeros'
            elif len(xs)<64: cat='short<64'
            else: cat='normal'
        mm=m if len(xs)>2*m else 1
        if len(xs)>mm:
            base=np.array([xs[-mm+(h%mm)] for h in range(pl)])
        else: base=np.full(pl, xs[-1] if len(xs) else 0.)
        ql_m=np.sum(2*np.abs((y[ok,None]-Q[i][ok])*((y[ok,None]<=Q[i][ok])-TAUS)))
        ql_n=np.sum(2*np.abs((y[ok]-base[ok]))*0.5)*9  # point forecast CRPS-ish
        cats.setdefault(cat,[]).append((np.log((ql_m+1e-9)/(ql_n+1e-9)),c))
for k,v in cats.items():
    a=np.array([x[0] for x in v]); print(f'{k:20s} n={len(v):5d} median log(model/snaive) {np.median(a):+.3f}  frac model worse {np.mean(a>0):.2f}  configs {len(set(x[1] for x in v))}')
