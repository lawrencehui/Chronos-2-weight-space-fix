import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
from chronos import Chronos2Pipeline
m=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
E=np.array([r['allnan'] for r in rows]); pl=np.array([r['pl'] for r in rows]); nop=np.ceil(pl/16).astype(int)
print('empty windows by nop:',{int(k):int(v) for k,v in zip(*np.unique(nop[E],return_counts=True))})
Wo=m.encoder.block[11].layer[2].mlp.wo.weight.detach().numpy()
g=m.encoder.block[11].layer[2].layer_norm.weight.detach().numpy()
H11=z['block11_out'].astype(np.float64); A=z['block11_mlp_act'].astype(np.float64)
Hin=H11-A@Wo.T     # mean MLP input of block 11
for name,X in [('block10_out',z['block10_out'].astype(np.float64)),('mlp11_in',Hin),('block11_out',H11)]:
    Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
    for k in np.unique(nop[E]):
        e=Xn[E&(nop==k)]; n=Xn[~E]
        ce=e@e.mean(0)/np.linalg.norm(e.mean(0))
        cn=n@e.mean(0)/np.linalg.norm(e.mean(0))
        print(f'{name:12s} nop={k:2d} empty-vs-centroid cos min {ce.min():.3f} | normal max cos {cn.max():.3f} (99.9pct {np.quantile(cn,0.999):.3f}) | same-nop normal max {(Xn[(~E)&(nop==k)]@e.mean(0)/np.linalg.norm(e.mean(0))).max():.3f}')
