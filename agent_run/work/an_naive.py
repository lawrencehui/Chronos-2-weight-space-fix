import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
out=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]; pl=v['pl']; m=v['m']
    QS=np.zeros_like(Q)
    for i,x in enumerate(v['ctx']):
        x=x[-2048:].astype(float)
        xs=x[~np.isnan(x)]
        if len(xs)==0: QS[i]=0; continue
        mm=m if len(xs)>2*m else 1
        # seasonal naive: repeat last season
        base=np.array([xs[-mm + (h % mm)] for h in range(pl)])
        res=xs[mm:]-xs[:-mm]
        if len(res)<5: res=np.array([0.0])
        qs=np.quantile(res,TAUS)
        hscale=np.sqrt(1+np.arange(pl)//mm)
        QS[i]=base[:,None]+qs[None,:]*hscale[:,None]
    r=crps(Y.reshape(-1),QS.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))
    out.append((c,r))
out.sort(key=lambda t:t[1])
for c,r in out[:15]: print(f'{c:40s} snaive/chronos CRPS ratio {r:.3f}')
print('geomean snaive/chronos',np.exp(np.mean([np.log(r) for c,r in out])))
