import pickle, numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp')
from dev_eval import load_split
TAUS=np.arange(1,10)/10
def crps(y,Q):
    ok=~np.isnan(y); y=y[ok]; Q=Q[ok]
    return float(np.mean((2*np.abs((y[:,None]-Q)*((y[:,None]<=Q)-TAUS))).sum(0)/np.abs(y).sum()))
def locscale(c):
    c=c[-2048:].astype(np.float64)
    if np.all(np.isnan(c)): return 0.,1.
    l=np.nanmean(c); s=np.sqrt(np.nanmean((c-l)**2)); return l,(s if s>0 else 1e-5)
def table():
    """per-window rows aligned with read npz order"""
    d=load_split('fit'); B=pickle.load(open('/Users/lawrencehui/Dev/MLX/wsp/astra/base_fit.pkl','rb'))
    rows=[]
    for cfg,v in d.items():
        for i,c in enumerate(v['ctx']):
            l,s=locscale(c); Q=B[cfg][i]; y=v['label'][i].astype(float)
            Qn=np.arcsinh((Q-l)/s); yn=np.arcsinh((y-l)/s)
            ok=~np.isnan(y)
            r=dict(cfg=cfg,i=i,pl=v['pl'],allnan=bool(np.all(np.isnan(c[-2048:]))),nlab=int(ok.sum()))
            if ok.any():
                r['below']=np.mean((y[ok,None]<Q[ok]),0)          # 9
                r['err']=float(np.mean(yn[ok]-Qn[ok,4]))
                r['spread']=float(np.mean(Qn[ok,8]-Qn[ok,0]))
                r['ql']=float(np.sum(2*np.abs((y[ok,None]-Q[ok])*((y[ok,None]<=Q[ok])-TAUS))))
                r['absy']=float(np.abs(y[ok]).sum())
            rows.append(r)
    return d,B,rows
