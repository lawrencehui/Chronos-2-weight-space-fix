import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
res={}
for maxobs in [64,128,256]:
  for kap in [0.1,0.2,0.3]:
    logs=[];changed=[]
    for c,v in d.items():
        Y=v['label'].astype(float); Q=B[c]; pl=v['pl']
        Q1=Q.copy(); n=0
        for i,x in enumerate(v['ctx']):
            x=x[-2048:]; ok=~np.isnan(x)
            if ok.sum()==0 or ok.sum()>maxobs: continue
            l,s=locscale(x); xn=np.arcsinh((x[ok]-l)/s); nn=len(xn)
            k=min(nn,max(2*pl,8)); t=np.arange(k); a,b=np.polyfit(t,xn[-k:],1)
            ext=a*(k+np.arange(pl))+b
            Qn=np.arcsinh((Q[i]-l)/s); Qn=Qn+kap*(ext-Qn[:,4])[:,None]
            Q1[i]=np.sinh(Qn)*s+l; n+=1
        lg=np.log(crps(Y.reshape(-1),Q1.reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9))); logs.append(lg)
        if n: changed.append((c,n,round(float(np.exp(lg)),3)))
    logs=np.array(logs); print(maxobs,kap,round(np.exp(logs.mean()),4),changed if kap==0.2 else '')
