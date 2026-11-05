import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
grid=np.linspace(-0.3,0.3,13); sgrid=np.linspace(0.7,1.4,15)
A=[];Bv=[];As=[];Bs=[]
for c,v in d.items():
    Y=v['label'].astype(float); Q=B[c]; W=v['window']
    if len(np.unique(W))<2: continue
    L=np.array([locscale(x)[0] for x in v['ctx']])[:,None,None]; S=np.array([locscale(x)[1] for x in v['ctx']])[:,None,None]
    an=np.array([np.all(np.isnan(x[-2048:])) for x in v['ctx']])
    Qn=np.arcsinh((Q-L)/S); med=Qn[:,:,4:5]
    res=[]
    for sel in [(W<W.max())&~an, (W==W.max())&~an]:
        y=Y[sel].reshape(-1)
        if np.all(np.isnan(y)): res=None; break
        b=crps(y,Q[sel].reshape(-1,9))
        sh=grid[np.argmin([crps(y,(np.sinh(Qn[sel]+g)*S[sel]+L[sel]).reshape(-1,9)) for g in grid])]
        sp=sgrid[np.argmin([crps(y,(np.sinh(med[sel]+s*(Qn[sel]-med[sel]))*S[sel]+L[sel]).reshape(-1,9)) for s in sgrid])]
        res.append((sh,sp))
    if res: A.append(res[0][0]);Bv.append(res[1][0]);As.append(res[0][1]);Bs.append(res[1][1])
print('configs',len(A),' corr of best shift (early windows vs last window):',np.corrcoef(A,Bv)[0,1],' sign agreement',np.mean(np.sign(A)==np.sign(Bv)))
print(' corr of best spread:',np.corrcoef(As,Bs)[0,1], ' agreement (both >1 or both <1)', np.mean((np.array(As)>1)==(np.array(Bs)>1)))
