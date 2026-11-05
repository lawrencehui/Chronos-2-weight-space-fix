import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
cf=z['config']
d,B,rows=table()
allnan=np.array([r['allnan'] for r in rows])
A=z['block11_mlp_act'].astype(np.float32)[~allnan]
frac=(A>0).mean(0); print('block11 most-often-active neurons frac:',np.sort(frac)[-10:].round(3))
for k in ['block8_out','block9_out','block10_out','block11_out']:
    X=z[k].astype(np.float64)[~allnan]
    Xn=X/np.sqrt((X**2).mean(1,keepdims=True))
    mu=Xn.mean(0); S=np.cov(Xn.T)+1e-6*np.eye(768)
    c=np.linalg.solve(S,mu); c/=np.linalg.norm(c)
    p=Xn@c
    # simple: best single dim
    r=np.abs(mu)/Xn.std(0); j=np.argmax(r)
    print(k,'fisher dir: mean %.3f sd %.4f ratio %.1f'%(p.mean(),p.std(),p.mean()/p.std()),' best dim',j,'mean %.3f sd %.3f'%(mu[j],Xn[:,j].std()), ' top dims |mu|:',np.argsort(-np.abs(mu))[:5], np.sort(-np.abs(mu))[:5].round(2))
