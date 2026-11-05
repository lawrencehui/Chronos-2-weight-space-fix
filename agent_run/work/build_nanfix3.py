import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
from chronos import Chronos2Pipeline
m=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
E=np.array([r['allnan'] for r in rows]); pl=np.array([r['pl'] for r in rows]); nop=np.ceil(pl/16).astype(int)
blk=m.encoder.block[11].layer[2]
Wi=blk.mlp.wi.weight.detach().numpy().astype(np.float32).copy(); Wo=blk.mlp.wo.weight.detach().numpy().astype(np.float32).copy(); g=blk.layer_norm.weight.detach().numpy()
H11=z['block11_out'].astype(np.float64); A=z['block11_mlp_act'].astype(np.float64)
Hin=H11-A@Wo.T
X=g*Hin/np.sqrt((Hin**2).mean(1,keepdims=True))
N=X[~E]; Em=X[E]; mu=N.mean(0); S=np.cov(N.T)
shrink=float(sys.argv[1]) if len(sys.argv)>1 else 3.0
Sr=S+shrink*np.trace(S)/768*np.eye(768)
w=np.linalg.solve(Sr,Em.mean(0)-mu); c=np.linalg.solve(Sr,mu)
pn=N@w; pe=Em@w; pcn=N@c; pce=Em@c
thr=(pn.max()+pe.min())/2; t=thr/pcn.mean()
u=w-t*c
un=N@u; ue=Em@u
print(f'shrink {shrink}: normal max {un.max():.3f} mean {un.mean():.3f} sd {un.std():.3f}; empty min {ue.min():.3f} max {ue.max():.3f}  margin(sd) {-un.max()/un.std():.1f} / {ue.min()/un.std():.1f}')
# scale u so that empty activation ~ 1
u=u/ue.mean()
# target: normal windows with confident flat forecasts, same nop as empties
spread=np.array([r.get('spread',np.nan) for r in rows]); err=np.array([r.get('err',np.nan) for r in rows])
deltas=[]
for k in np.unique(nop[E]):
    sel=(~E)&(nop==k)
    medn=[]
    idx=np.where(sel)[0]
    # normalized median level of the forecast (mean over horizon) from base quantiles
    lv=[]
    for j in idx:
        r=rows[j]; v=d[r['cfg']]; l,s=locscale(v['ctx'][r['i']]); lv.append(np.mean(np.arcsinh((B[r['cfg']][r['i']][:,4]-l)/s)))
    lv=np.array(lv); sp=spread[idx]
    good=idx[(np.abs(lv)<0.5)&(sp<np.nanquantile(sp,0.25))]
    T=H11[good].mean(0); Ek=H11[E&(nop==k)].mean(0)
    deltas.append(T-Ek)
    print(f'nop {k}: target from {len(good)} windows; |T| {np.linalg.norm(T):.1f} |E| {np.linalg.norm(Ek):.1f} cos(T,E) {T@Ek/np.linalg.norm(T)/np.linalg.norm(Ek):.3f}')
delta=np.mean(deltas,0)
j=int(np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/nanfix_geom.npy',allow_pickle=True).item()['dead'][0])
Wi2=Wi.copy(); Wo2=Wo.copy()
Wi2[j]=u.astype(np.float32); Wo2[:,j]=delta.astype(np.float32)
print('neuron',j,'old wi/wo max',np.abs(Wi[j]).max(),np.abs(Wo[:,j]).max(),' new norms',np.linalg.norm(Wi2[j]),np.linalg.norm(Wo2[:,j]))
out=sys.argv[2] if len(sys.argv)>2 else '/Users/lawrencehui/Dev/MLX/wsp/astra/work/nanfix.npz'
np.savez(out,**{'encoder.block.11.layer.2.mlp.wi.weight':Wi2,'encoder.block.11.layer.2.mlp.wo.weight':Wo2})
