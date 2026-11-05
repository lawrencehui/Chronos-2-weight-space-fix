"""Empty-context fix: repurpose one dead neuron j in block 11's MLP as a detector of the all-masked-context state,
writing a vector that moves the final state toward a typical 'flat, confident' forecast state."""
import numpy as np, sys, json
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
from chronos import Chronos2Pipeline
m=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
E=np.array([r['allnan'] for r in rows]); pl=np.array([r['pl'] for r in rows]); nop=np.ceil(pl/16).astype(int)
blk=m.encoder.block[11].layer[2]
Wi=blk.mlp.wi.weight.detach().numpy().copy(); Wo=blk.mlp.wo.weight.detach().numpy().copy(); g=blk.layer_norm.weight.detach().numpy()
print('LN g min/med', g.min(), np.median(g), (np.abs(g)<0.05).sum())
H11=z['block11_out'].astype(np.float64); A=z['block11_mlp_act'].astype(np.float64)
Hin=H11-A@Wo.T
X=g*Hin/np.sqrt((Hin**2).mean(1,keepdims=True))        # approx MLP input after LN (what wi sees)
Xu=X/np.linalg.norm(X,axis=1,keepdims=True)
# empty centroid (unit) averaged over nop groups (equal weight per nop)
ks=np.unique(nop[E]); cents=np.array([Xu[E&(nop==k)].mean(0) for k in ks]); cents/=np.linalg.norm(cents,axis=1,keepdims=True)
print('cos between empty centroids of different nop:\n',np.round(cents@cents.T,3))
e=cents.mean(0); e/=np.linalg.norm(e)
nhat=Xu[~E].mean(0); nhat/=np.linalg.norm(nhat)
pe=Xu[E]@e; pn=Xu[~E]@e; qn=Xu[~E]@nhat; qe=Xu[E]@nhat
print('empty: cos(e) min %.3f, cos(nhat) %.3f..%.3f | normal: cos(e) max %.3f, cos(nhat) min %.3f med %.3f'%(pe.min(),qe.min(),qe.max(),pn.max(),qn.min(),np.median(qn)))
for beta in [0.5,0.75,1.0,1.25]:
    sn=pn-beta*qn; se=pe-beta*qe
    print(f'beta {beta}: normal max {sn.max():+.3f} (99.9% {np.quantile(sn,0.999):+.3f})  empty min {se.min():+.3f}')
# dead neurons: tiny weights and never active
dead=np.where((np.abs(Wi).max(1)<1e-4)&(np.abs(Wo).max(0)<1e-4)&(A.max(0)==0))[0]
print('candidate dead neurons',len(dead), dead[:10])
np.save('/Users/lawrencehui/Dev/MLX/wsp/astra/work/nanfix_geom.npy',dict(e=e,nhat=nhat,dead=dead,ks=ks))
