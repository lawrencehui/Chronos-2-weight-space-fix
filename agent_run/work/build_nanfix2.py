import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
from chronos import Chronos2Pipeline
m=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model
z=np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/read_base.npz')
d,B,rows=table()
E=np.array([r['allnan'] for r in rows]); pl=np.array([r['pl'] for r in rows]); nop=np.ceil(pl/16).astype(int)
blk=m.encoder.block[11].layer[2]
Wo=blk.mlp.wo.weight.detach().numpy(); g=blk.layer_norm.weight.detach().numpy()
H11=z['block11_out'].astype(np.float64); A=z['block11_mlp_act'].astype(np.float64)
Hin=H11-A@Wo.T
X=g*Hin/np.sqrt((Hin**2).mean(1,keepdims=True))   # LN output, norm ~ ||g||-ish
N=X[~E]; Em=X[E]
mu=N.mean(0); S=np.cov(N.T)
for shrink in [0.1,0.3,1.0]:
    Sr=S+shrink*np.trace(S)/768*np.eye(768)
    w=np.linalg.solve(Sr,Em.mean(0)-mu)
    pn=N@w; pe=Em@w
    # constant direction
    c=np.linalg.solve(Sr,mu); pc_n=N@c; pc_e=Em@c
    # threshold: midpoint between max normal and min empty, in w units; express via c: u=w - t*c
    thr=(pn.max()+pe.min())/2
    t=thr/pc_n.mean()
    un=pn-t*pc_n; ue=pe-t*pc_e
    sd=un.std()
    print(f'shrink {shrink}: normal proj max {pn.max():.2f} empty min {pe.min():.2f} | c-proj normal mean {pc_n.mean():.2f} sd {pc_n.std():.3f}, empty {pc_e.mean():.2f} | after bias: normal max {un.max()/sd:+.1f} sd, mean {un.mean()/sd:+.1f} sd; empty min {ue.min()/sd:+.1f} sd')
