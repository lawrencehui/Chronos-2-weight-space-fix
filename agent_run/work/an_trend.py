import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
data=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]; pl=v['pl']
    L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Qn=np.arcsinh((Q-L)/S)
    corr=np.zeros((len(Q),pl))
    for i,x in enumerate(v['ctx']):
        x=x[-2048:]; xn=np.arcsinh((x-L[i,0,0])/S[i,0,0]); ok=~np.isnan(xn)
        n=min(ok.sum(), max(2*pl, 2*v['m']))
        if n<4: continue
        xs=xn[ok][-n:]; t=np.arange(n)
        a,b=np.polyfit(t,xs,1)
        ext=a*(n+np.arange(pl))+b   # linear trend extrapolation
        corr[i]=ext-Qn[i,:,4]
    data.append((c,Y,Q,Qn,L,S,an,corr))
for kap in [0.05,0.1,0.2]:
    logs=[];names=[]
    for c,Y,Q,Qn,L,S,an,corr in data:
        Qe=Qn+kap*corr[:,:,None]; Qe[an]=Qn[an]
        logs.append(np.log(crps(Y.reshape(-1),(np.sinh(Qe)*S+L).reshape(-1,9))/crps(Y.reshape(-1),Q.reshape(-1,9)))); names.append(c)
    logs=np.array(logs); o=np.argsort(logs)
    print(kap,round(np.exp(logs.mean()),4),'wins',(logs<-.005).sum(),'losses',(logs>.005).sum(),' best:',[(names[i],round(np.exp(logs[i]),3)) for i in o[:6]])
