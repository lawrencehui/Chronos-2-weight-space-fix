import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
L=[]
for r in rows:
    if r['allnan']: continue
    v=d[r['cfg']]; l,s=locscale(v['ctx'][r['i']]); Q=B[r['cfg']][r['i']]; Qn=np.arcsinh((Q-l)/s)
    x=v['ctx'][r['i']][-2048:]; xn=np.arcsinh((x-l)/s)
    rng=(np.nanmin(xn),np.nanmax(xn))
    over=max(Qn[:,8].max()-rng[1], rng[0]-Qn[:,0].min())
    L.append((over,r['cfg'],r['i'],rng,Qn[:,0].min(),Qn[:,8].max(),r.get('ql',0)/max(r.get('absy',1),1e-9)))
L.sort(reverse=True)
for x in L[:15]: print(f'over {x[0]:.2f} {x[1]:35s} win {x[2]:3d} ctx range {x[3][0]:+.2f}..{x[3][1]:+.2f} fc q10min {x[4]:+.2f} q90max {x[5]:+.2f} win crps {x[6]:.3f}')
