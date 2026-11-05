import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
win=np.concatenate([d[c]['window'] for c in d]); item=np.concatenate([d[c]['item'] for c in d])
keep=np.array([(not r['allnan']) and r['nlab']>0 for r in rows])
cfgs=np.array([r['cfg'] for r in rows]); ucf=list(dict.fromkeys(cfgs)); gid=np.array([ucf.index(c) for c in cfgs])
w=1.0/np.bincount(gid)[gid]
t50=np.array([r['below'][4]-0.5 if 'below' in r else np.nan for r in rows])
lastw=np.zeros(len(rows),bool)
for c in ucf:
    m=np.where(cfgs==c)[0]
    for it in np.unique(item[m]):
        mm=m[item[m]==it]; lastw[mm[np.argmax(win[mm])]]=True
def fit(X,y,train,lam):
    mu=np.average(X[train],0,w[train]); sd=np.sqrt(np.average((X[train]-mu)**2,0,w[train]))+1e-6
    A=(X-mu)/sd; ym=np.average(y[train],weights=w[train]); Aw=A[train]*w[train,None]
    beta=np.linalg.solve(Aw.T@A[train]+lam*w[train].sum()*np.eye(A.shape[1]),Aw.T@(y[train]-ym))
    return A@beta+ym
L=int(sys.argv[1]) if len(sys.argv)>1 else 10
X=z[f'block{L}_out'].astype(np.float64); X=X/np.sqrt((X**2).mean(1,keepdims=True))
ok=keep&np.isfinite(t50)
pred=np.full(len(rows),0.0)
pred_t=fit(X,np.nan_to_num(t50),ok&~lastw,10)   # temporal: train windows 0,1 -> test last
# also all-fold by config
pred_c=np.zeros(len(rows))
for f in range(5):
    pc=fit(X,np.nan_to_num(t50),ok&(gid%5!=f),10); pred_c[gid%5==f]=pc[gid%5==f]
def evaluate(pred,mask,kappa):
    logs=[]
    for c in ucf:
        idx=np.where((cfgs==c)&mask)[0]
        if len(idx)==0: continue
        Y=[];Q0=[];Q1=[]
        for j in idx:
            r=rows[j]; v=d[c]; l,s=locscale(v['ctx'][r['i']]); Q=B[c][r['i']]; y=v['label'][r['i']].astype(float)
            Qn=np.arcsinh((Q-l)/s); sp=np.mean(Qn[:,8]-Qn[:,0])
            dl=-kappa*pred[j]*sp if not r['allnan'] else 0
            Y.append(y);Q0.append(Q);Q1.append(np.sinh(Qn+dl)*s+l)
        Y=np.concatenate(Y);Q0=np.concatenate(Q0);Q1=np.concatenate(Q1)
        if np.all(np.isnan(Y)): continue
        logs.append(np.log(crps(Y,Q1)/crps(Y,Q0)))
    return np.exp(np.mean(logs)), np.sum(np.array(logs)<-0.005), np.sum(np.array(logs)>0.005)
for kappa in [0.25,0.5,1.0,1.5]:
    print('L',L,'kappa',kappa,'temporal-heldout(last windows):',evaluate(pred_t,lastw,kappa),' cfg-heldout(all):',evaluate(pred_c,np.ones(len(rows),bool),kappa))
