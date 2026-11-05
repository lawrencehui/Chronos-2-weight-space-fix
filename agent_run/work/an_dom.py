import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
import collections
tot=collections.defaultdict(float)
for r in rows:
    if 'ql' in r: tot[r['cfg']]+=r['ql']
L=[]
for r in rows:
    if 'ql' in r and tot[r['cfg']]>0:
        sh=r['ql']/tot[r['cfg']]; n=len(d[r['cfg']]['ctx'])
        if sh>max(0.2,3/n): L.append((sh,r['cfg'],r['i'],r['allnan'],r['spread'],r['err']))
L.sort(reverse=True)
for x in L[:40]:
    c=x[1]; i=x[2]; ctx=d[c]['ctx'][i][-2048:]; nobs=np.sum(~np.isnan(ctx))
    print(f'share {x[0]:.2f} {c:36s} win {i:3d} n={len(d[c]["ctx"])} allnan {x[3]} nobs {nobs} spread80 {x[4]:.2f} err {x[5]:+.2f}')
