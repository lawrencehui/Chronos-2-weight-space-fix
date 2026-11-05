import numpy as np
z=dict(np.load('/Users/lawrencehui/Dev/MLX/wsp/astra/work/nanfix.npz'))
ki='encoder.block.11.layer.2.mlp.wi.weight'; ko='encoder.block.11.layer.2.mlp.wo.weight'
j=11; rng=np.random.default_rng(0)
wi=z[ki]; wo=z[ko]
# C1 random write
r=rng.standard_normal(768).astype(np.float32); r*=np.linalg.norm(wo[:,j])/np.linalg.norm(r)
wo1=wo.copy(); wo1[:,j]=r; np.savez('/Users/lawrencehui/Dev/MLX/wsp/astra/work/ctl_randwrite.npz',**{ki:wi,ko:wo1})
# C2 random detector
r=rng.standard_normal(768).astype(np.float32); r*=np.linalg.norm(wi[j])/np.linalg.norm(r)
wi2=wi.copy(); wi2[j]=r; np.savez('/Users/lawrencehui/Dev/MLX/wsp/astra/work/ctl_randdet.npz',**{ki:wi2,ko:wo})
for s in [0.5,2.0]:
    wo3=wo.copy(); wo3[:,j]*=s; np.savez(f'/Users/lawrencehui/Dev/MLX/wsp/astra/work/nanfix_x{s}.npz',**{ki:wi,ko:wo3})
