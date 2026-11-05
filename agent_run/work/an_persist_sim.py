import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
data=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]
    L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Qn=np.arcsinh((Q-L)/S)
    last=np.array([np.nanmean(np.arcsinh((x[-16:]-l)/s)) if np.any(~np.isnan(x[-16:])) else np.nan for x,l,s in zip(v['ctx'],L[:,0,0],S[:,0,0])])
    last=np.nan_to_num(last - Qn[:,:,4].mean(1))
    data.append((c,Y,Q,Qn,L,S,an,last))
for kap in [0.03,0.06,0.1]:
  for mode in ['const','ramp']:
    logs=[]
    for c,Y,Q,Qn,L,S,an,last in data:
        pl=Q.shape[1]
        prof=np.ones(pl) if mode=='const' else np.exp(-np.arange(pl)/16.)
        Qe=Qn+kap*last[:,None,None]*prof[None,:,None]; Qe[an]=Qn[an]
        logs.append(np.log(crps(Y.reshape(-1),(np.sinh(Qe)*S+L).reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))))
    logs=np.array(logs); print(kap,mode,round(np.exp(logs.mean()),4),'wins',(logs<-.005).sum(),'losses',(logs>.005).sum())
