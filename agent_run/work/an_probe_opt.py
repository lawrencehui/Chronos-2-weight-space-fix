import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
win=np.concatenate([d[c]['window'] for c in d]); item=np.concatenate([d[c]['item'] for c in d])
cfgs=np.array([r['cfg'] for r in rows]); ucf=list(dict.fromkeys(cfgs)); gid=np.array([ucf.index(c) for c in cfgs])
keep=np.array([(not r['allnan']) and r['nlab']>0 for r in rows])
grid=np.linspace(-0.4,0.4,17)
opt=np.zeros(len(rows)); W=[]
cache=[]
for j,r in enumerate(rows):
    v=d[r['cfg']]; l,s=locscale(v['ctx'][r['i']]); Q=B[r['cfg']][r['i']]; y=v['label'][r['i']].astype(float)
    Qn=np.arcsinh((Q-l)/s); ok=~np.isnan(y)
    cache.append((Qn,l,s,y))
    if not keep[j]: continue
    losses=[np.sum(2*np.abs((y[ok,None]-(np.sinh(Qn[ok]+g)*s+l))*((y[ok,None]<=(np.sinh(Qn[ok]+g)*s+l))-TAUS))) for g in grid]
    opt[j]=grid[int(np.argmin(losses))]
lastw=np.zeros(len(rows),bool)
for c in ucf:
    m=np.where(cfgs==c)[0]
    for it in np.unique(item[m]):
        mm=m[item[m]==it]; lastw[mm[np.argmax(win[mm])]]=True
w=1.0/np.bincount(gid)[gid]
def fit(X,y,train,lam):
    mu=np.average(X[train],0,w[train]); sd=np.sqrt(np.average((X[train]-mu)**2,0,w[train]))+1e-6
    A=(X-mu)/sd; ym=np.average(y[train],weights=w[train]); Aw=A[train]*w[train,None]
    beta=np.linalg.solve(Aw.T@A[train]+lam*w[train].sum()*np.eye(A.shape[1]),Aw.T@(y[train]-ym))
    return A@beta+ym
te=keep&lastw; tr=keep&~lastw
for L in [9,10,11]:
    X=z[f'block{L}_out'].astype(np.float64); X=X/np.sqrt((X**2).mean(1,keepdims=True))
    for lam in [10,100]:
        p=fit(X,opt,tr,lam)
        r2=1-np.sum(w[te]*(opt[te]-p[te])**2)/np.sum(w[te]*(opt[te]-np.average(opt[te],weights=w[te]))**2)
        for kap in [0.5,1.0]:
            logs=[]
            for c in ucf:
                idx=np.where((cfgs==c)&te)[0]
                if len(idx)==0: continue
                Y=[];Q0=[];Q1=[]
                for j in idx:
                    Qn,l,s,y=cache[j]; Y.append(y); Q0.append(np.sinh(Qn)*s+l); Q1.append(np.sinh(Qn+kap*p[j])*s+l)
                Y=np.concatenate(Y);Q0=np.concatenate(Q0);Q1=np.concatenate(Q1)
                logs.append(np.log(crps(Y,Q1)/crps(Y,Q0)))
            logs=np.array(logs)
            print(f'L{L} lam{lam} R2 {r2:+.3f} kap {kap}: {np.exp(logs.mean()):.4f} wins {(logs<-.005).sum()} losses {(logs>.005).sum()}')
