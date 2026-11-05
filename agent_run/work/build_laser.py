import numpy as np, sys
from chronos import Chronos2Pipeline
m=Chronos2Pipeline.from_pretrained('amazon/chronos-2', device_map='cpu').model
blk,nm,r,out=int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),sys.argv[4]
key=f'encoder.block.{blk}.layer.2.mlp.{nm}.weight'
W=m.state_dict()[key].numpy().astype(np.float64)
U,s,Vt=np.linalg.svd(W,full_matrices=False)
Wr=(U[:,:r]*s[:r])@Vt[:r]
print(key,'rank',r,'rel change',np.linalg.norm(Wr-W)/np.linalg.norm(W))
np.savez(out,**{key:Wr.astype(np.float32)})
