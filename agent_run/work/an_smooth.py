import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
for k in [3,5]:
  for mode in ['all','tails']:
    logs=[]
    for c,v in d.items():
        Y=v['label'].astype(float); Q=B[c]
        L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
        Qn=np.arcsinh((Q-L)/S)
        ker=np.ones(k)/k
        Qs=Qn.copy()
        for i in range(Qn.shape[0]):
            for q in range(9):
                if mode=='tails' and q in (3,4,5): continue
                x=np.pad(Qn[i,:,q],(k//2,k//2),mode='edge'); Qs[i,:,q]=np.convolve(x,ker,'valid')
        Q1=np.sinh(Qs)*S+L
        logs.append(np.log(crps(Y.reshape(-1),Q1.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))))
    logs=np.array(logs); print(k,mode,round(np.exp(logs.mean()),4),'wins',(logs<-.005).sum(),'losses',(logs>.005).sum())
