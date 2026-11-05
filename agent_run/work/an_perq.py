import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
data=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]
    L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Qn=np.arcsinh((Q-L)/S); med=Qn[:,:,4:5]
    data.append((c,Y,Q,Qn,med,L,S,an))
def score(f):
    logs=[]
    for c,Y,Q,Qn,med,L,S,an in data:
        Qe=f(Qn,med); Qe[an]=Qn[an]
        Q1=np.sinh(Qe)*S+L
        logs.append(np.log(crps(Y.reshape(-1),Q1.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))))
    return np.exp(np.mean(logs))
# per-level spread scale: q_k' = med + s_k (q_k - med)
for k in [0,1,2,3,5,6,7,8]:
    out=[]
    for s in [0.9,0.95,1.05,1.1]:
        def f(Qn,med,k=k,s=s):
            Qe=Qn.copy(); Qe[...,k]=med[...,0]+s*(Qn[...,k]-med[...,0]); return Qe
        out.append(f'{s}:{score(f):.4f}')
    print('level',k+1,' '.join(out))
# tails only (q10,q90) widening, symmetric
for s in [0.95,1.05,1.1]:
    def f(Qn,med,s=s):
        Qe=Qn.copy()
        for k in [0,8]: Qe[...,k]=med[...,0]+s*(Qn[...,k]-med[...,0])
        return Qe
    print('tails',s,round(score(f),4))
# global median shift
for dl in [-0.02,0.02]:
    print('shift',dl,round(score(lambda Qn,med,dl=dl: Qn+dl),4))
