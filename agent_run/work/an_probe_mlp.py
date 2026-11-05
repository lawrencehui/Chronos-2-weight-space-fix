import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
win=np.concatenate([d[c]['window'] for c in d]); item=np.concatenate([d[c]['item'] for c in d])
cfgs=np.array([r['cfg'] for r in rows]); ucf=list(dict.fromkeys(cfgs)); gid=np.array([ucf.index(c) for c in cfgs])
keep=np.array([(not r['allnan']) and r['nlab']>0 for r in rows])
lastw=np.zeros(len(rows),bool)
for c in ucf:
    m=np.where(cfgs==c)[0]
    for it in np.unique(item[m]):
        mm=m[item[m]==it]; lastw[mm[np.argmax(win[mm])]]=True
w=1.0/np.bincount(gid)[gid]
T={'t50':np.array([r['below'][4]-0.5 if 'below' in r else 0 for r in rows]),
   'miss80':np.array([(1-(r['below'][8]-r['below'][0]))-0.2 if 'below' in r else 0 for r in rows])}
def r2(X,y,train,test,lam):
    mu=np.average(X[train],0,w[train]); sd=np.sqrt(np.average((X[train]-mu)**2,0,w[train]))+1e-3
    A=(X-mu)/sd; ym=np.average(y[train],weights=w[train]); Aw=A[train]*w[train,None]
    beta=np.linalg.solve(Aw.T@A[train]+lam*w[train].sum()*np.eye(A.shape[1]),Aw.T@(y[train]-ym))
    p=A[test]@beta+ym
    return 1-np.sum(w[test]*(y[test]-p)**2)/np.sum(w[test]*(y[test]-np.average(y[test],weights=w[test]))**2)
for k,y in T.items():
    for b in [9,10,11]:
        for nm in ['mlp_act','timeattn_heads']:
            X=z[f'block{b}_{nm}'].astype(np.float64)
            X=X[:,X[keep].std(0)>1e-4]
            print(k,b,nm,X.shape[1],' '.join(f'lam{l}:{r2(X,y,keep&~lastw,keep&lastw,l):+.3f}' for l in [10,100,1000]))
