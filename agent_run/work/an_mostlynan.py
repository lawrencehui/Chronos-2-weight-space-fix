import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
for lo,hi in [(0.3,0.7),(0.7,0.95),(0.95,0.9999)]:
  for sc in [0.6,0.8,1.2]:
    logs=[];n=0
    for c,v in d.items():
        Y=v['label'].astype(float); Q=B[c]; Q1=Q.copy(); hit=False
        for i,x in enumerate(v['ctx']):
            f=np.mean(np.isnan(x[-2048:]))
            if lo<f<hi:
                l,s=locscale(x); Qn=np.arcsinh((Q[i]-l)/s); med=Qn[:,4:5]
                Q1[i]=np.sinh(med+sc*(Qn-med))*s+l; hit=True; n+=1
        if hit: logs.append(np.log(crps(Y.reshape(-1),Q1.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))))
    print(f'nan frac {lo}-{hi} spread {sc}: windows {n} configs {len(logs)} geomean-over-touched {np.exp(np.mean(logs)):.4f}  overall-equiv {np.exp(np.sum(logs)/96):.4f}')
