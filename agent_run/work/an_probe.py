import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
win=np.concatenate([d[c]['window'] for c in d]); item=np.concatenate([d[c]['item'] for c in d])
print('window values',np.unique(win)[:10])
keep=np.array([(not r['allnan']) and r['nlab']>0 for r in rows])
cfgs=np.array([r['cfg'] for r in rows]); ucf=list(dict.fromkeys(cfgs)); gid=np.array([ucf.index(c) for c in cfgs])
T={}
T['t10']=np.array([r['below'][0]-0.1 if 'below' in r else np.nan for r in rows])
T['t50']=np.array([r['below'][4]-0.5 if 'below' in r else np.nan for r in rows])
T['t90']=np.array([r['below'][8]-0.9 if 'below' in r else np.nan for r in rows])
T['zerr']=np.array([r['err']/max(r['spread'],1e-3) if 'err' in r else np.nan for r in rows])
w=1.0/np.bincount(gid)[gid]
def r2(X,y,train,test,lam):
    mu=np.average(X[train],0,w[train]); sd=np.sqrt(np.average((X[train]-mu)**2,0,w[train]))+1e-6
    A=(X-mu)/sd; ym=np.average(y[train],weights=w[train])
    Aw=A[train]*w[train,None]
    beta=np.linalg.solve(Aw.T@A[train]+lam*w[train].sum()*np.eye(A.shape[1]),Aw.T@(y[train]-ym))
    p=A[test]@beta+ym
    sse=np.sum(w[test]*(y[test]-p)**2); sst=np.sum(w[test]*(y[test]-np.average(y[test],weights=w[test]))**2)
    return 1-sse/sst
# temporal split: last window index of each item as test
lastw=np.zeros(len(rows),bool)
for c in ucf:
    m=np.where(cfgs==c)[0]
    for it in np.unique(item[m]):
        mm=m[item[m]==it]; lastw[mm[np.argmax(win[mm])]]=True
for k in ['t10','t50','t90','zerr']:
    y=T[k]; ok=keep&np.isfinite(y)
    if k=='zerr': y=np.clip(y,-3,3)
    out=[]
    for L in [3,6,8,9,10,11]:
        X=z[f'block{L}_out'].astype(np.float64)
        X=X/np.sqrt((X**2).mean(1,keepdims=True))
        best=max(r2(X,y,ok&~lastw,ok&lastw,lam) for lam in [1,10,100])
        # config-held-out (5 folds)
        cv=[]
        for f in range(5):
            te=ok&(gid%5==f); tr=ok&(gid%5!=f)
            cv.append(max(r2(X,y,tr,te,lam) for lam in [10,100]))
        out.append(f'L{L}: time {best:+.3f} cfg {np.mean(cv):+.3f}')
    print(k,' | '.join(out))
