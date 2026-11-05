import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
data=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]
    L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Qn=np.arcsinh((Q-L)/S); data.append((c,v['pl'],Y,Q,Qn,L,S,an))
for kap in [-0.05,0.05,0.1]:
  for prof in ['ramp','const']:
    logs=[];grp={}
    for c,pl,Y,Q,Qn,L,S,an in data:
        p=(np.arange(pl)/max(pl-1,1)) if prof=='ramp' else np.ones(pl)
        Qe=Qn-kap*p[None,:,None]*Qn[:,:,4:5]; Qe[an]=Qn[an]
        lg=np.log(crps(Y.reshape(-1),(np.sinh(Qe)*S+L).reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9)))
        logs.append(lg); grp.setdefault(c.split('/')[-1],[]).append(lg)
    logs=np.array(logs); print(kap,prof,round(np.exp(logs.mean()),4),'wins',(logs<-.005).sum(),'losses',(logs>.005).sum(),{k:round(float(np.exp(np.mean(v))),4) for k,v in grp.items()})
