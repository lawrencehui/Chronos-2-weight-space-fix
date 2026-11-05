import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
for r in rows:
    v=d[r['cfg']]; x=v['ctx'][r['i']][-2048:]; nobs=np.sum(~np.isnan(x))
    if 0<nobs<=48 or (nobs>0 and np.isnan(x[-16:]).all()):
        Q=B[r['cfg']][r['i']]; l,s=locscale(v['ctx'][r['i']]); Qn=np.arcsinh((Q-l)/s)
        print(f"{r['cfg']:30s} win {r['i']:3d} nobs {nobs:4d} last16 all nan {np.isnan(x[-16:]).all()} q10/q50/q90 norm {Qn[:,0].mean():+.2f} {Qn[:,4].mean():+.2f} {Qn[:,8].mean():+.2f} nlab {r['nlab']}")
