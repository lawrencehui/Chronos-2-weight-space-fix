import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
for W in [64,128,256,512]:
    logs=[]; det=[]
    for c,v in d.items():
        Y=v['label'].astype(float); Q=B[c]; Q1=Q.copy(); n=0
        for i,x in enumerate(v['ctx']):
            x=x[-2048:].astype(float); xs=x[~np.isnan(x)]
            if len(xs)>=W and np.std(xs[-W:])==0:
                Q1[i]=xs[-1]; n+=1
        if n:
            r=crps(Y.reshape(-1),Q1.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9)); logs.append(np.log(r)); det.append((c,n,round(r,4)))
        else: logs.append(0)
    print(W,'geomean',round(np.exp(np.mean(logs)),4),det if W==256 else '')
