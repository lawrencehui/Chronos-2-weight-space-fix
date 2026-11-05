import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
assert len(rows)==len(z['config'])
assert all(r['cfg']==c for r,c in zip(rows,z['config']))
A=z['block11_mlp_act'].astype(np.float32)
allnan=np.array([r['allnan'] for r in rows])
An=A[~allnan]
mn=An.min(0); mu=An.mean(0); sd=An.std(0)
on=np.where(mn>0)[0]
print('block11 neurons active (window-mean>0) in all non-empty windows:',len(on))
o=on[np.argsort(sd[on]/mu[on])][:15]
for j in o: print(' n',j,'mean %.3f sd %.3f cv %.3f min %.3f'%(mu[j],sd[j],sd[j]/mu[j],mn[j]))
dead=np.where(An.max(0)==0)[0]; print('never active at forecast tokens:',len(dead))
for b in [9,10]:
    A2=z[f'block{b}_mlp_act'].astype(np.float32)[~allnan]
    on2=np.where(A2.min(0)>0)[0]; cv=A2.std(0)/np.maximum(A2.mean(0),1e-9)
    print(f'block{b}: always-on',len(on2),' best cv',np.sort(cv[on2])[:5].round(3), 'never active', (A2.max(0)==0).sum())
# empty context states
for k in ['block0_out','block5_out','block11_out']:
    X=z[k].astype(np.float32)
    print(k,'empty-ctx rows norm',np.linalg.norm(X[allnan],axis=1)[:5].round(2),' normal rows median norm',np.median(np.linalg.norm(X[~allnan],axis=1)).round(2))
