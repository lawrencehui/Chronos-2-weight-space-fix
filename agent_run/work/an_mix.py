import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
SN={}
for c,v in d.items():
    Q=B[c]; pl=v['pl']; m=v['m']; QS=np.zeros_like(Q)
    for i,x in enumerate(v['ctx']):
        x=x[-2048:].astype(float); xs=x[~np.isnan(x)]
        if len(xs)==0: QS[i]=Q[i]; continue
        mm=m if len(xs)>2*m else 1
        base=np.array([xs[-mm+(h%mm)] for h in range(pl)])
        # median-only shift: keep model's spread around seasonal naive point
        QS[i]=base[:,None]+(Q[i]-Q[i][:,4:5])
    SN[c]=QS
for w in [0.05,0.1,0.2]:
    logs=[]
    for c,v in d.items():
        Y=v['label'].astype(float); Q=B[c]
        Qm=(1-w)*Q+w*SN[c]
        logs.append(np.log(crps(Y.reshape(-1),Qm.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))))
    logs=np.array(logs); print(w,round(np.exp(logs.mean()),4),'wins',(logs<-.005).sum(),'losses',(logs>.005).sum())
