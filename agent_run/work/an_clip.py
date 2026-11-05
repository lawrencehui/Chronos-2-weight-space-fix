import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
logs=[];logs2=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c].copy(); Q0=B[c]
    nonneg=np.array([np.nanmin(x[-2048:])>=0 if np.any(~np.isnan(x[-2048:])) else False for x in v['ctx']])
    Q1=Q.copy(); Q1[nonneg]=np.maximum(Q1[nonneg],0)
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Q2=Q1.copy(); Q2[an]=0
    b=crps(Y.reshape(-1),Q0.reshape(-1,9)); r=crps(Y.reshape(-1),Q1.reshape(-1,9))/b; r2=crps(Y.reshape(-1),Q2.reshape(-1,9))/b
    logs.append(np.log(r)); logs2.append(np.log(r2))
    negfrac=np.mean(Q0[nonneg]<0) if nonneg.any() else 0
    if r<0.995: print(f'{c:40s} clip ratio {r:.4f}  nonneg windows {nonneg.mean():.2f} frac quantiles<0 {negfrac:.3f}')
print('clip geomean',np.exp(np.mean(logs)),' clip+nanfix',np.exp(np.mean(logs2)))
