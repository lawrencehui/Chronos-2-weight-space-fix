import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
rows=[]
for cfg,v in d.items():
    Q=B[cfg]; y=v['label'].astype(float); pl=v['pl']
    for i,c in enumerate(v['ctx']):
        c=c[-2048:].astype(np.float64)
        if np.all(np.isnan(c)): continue
        l=np.nanmean(c); s=np.sqrt(np.nanmean((c-l)**2)); s=s if s>0 else 1e-5
        n=lambda x: np.arcsinh((x-l)/s)
        last=np.nanmean(n(c[-16:])) if np.any(~np.isnan(c[-16:])) else np.nan
        lastseas = np.nanmean(n(c[-max(v['m'],16):]))
        h=min(pl,48)
        med=np.mean(n(Q[i,:h,4])); yy=np.nanmean(n(y[i,:h]))
        spread=np.mean(n(Q[i,:h,8])-n(Q[i,:h,0]))
        rows.append((cfg,last-med,lastseas-med,yy-med,spread,pl))
import collections
R=np.array([r[1:] for r in rows],float); cf=[r[0] for r in rows]
ok=np.all(np.isfinite(R),1); R=R[ok]; cf=np.array(cf)[ok]
x=R[:,0]; e=R[:,2]
print('pooled: corr(last-med, y-med)=',np.corrcoef(x,e)[0,1], 'slope', np.polyfit(x,e,1))
x=R[:,1]; print('pooled seasonal-mean version corr', np.corrcoef(x,e)[0,1], 'slope', np.polyfit(x,e,1))
# per config slope
sl=[]
for k in dict.fromkeys(cf):
    m=cf==k
    if m.sum()<5 or np.std(R[m,0])==0: continue
    b=np.polyfit(R[m,0],R[m,2],1)[0]; sl.append((k,b,np.corrcoef(R[m,0],R[m,2])[0,1]))
sl=np.array([(s[1],s[2]) for s in sl])
print('per-config slope median',np.median(sl[:,0]),'frac positive',np.mean(sl[:,0]>0),'median corr',np.median(sl[:,1]))
# error normalized by spread
z=R[:,2]/R[:,3]
print('pooled corr with spread-normalized error', np.corrcoef(R[:,0]/R[:,3],z)[0,1])
