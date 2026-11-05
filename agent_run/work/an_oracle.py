import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
taus = np.arange(1,10)/10
def crps(y,Q):
    ok=~np.isnan(y); y=y[ok]; Q=Q[ok]
    return float(np.mean((2*np.abs((y[:,None]-Q)*((y[:,None]<=Q)-taus))).sum(0)/np.abs(y).sum()))
def norm(v):
    locs=[];scs=[]
    for c in v['ctx']:
        c=c[-2048:].astype(np.float64)
        l=np.nanmean(c) if np.any(~np.isnan(c)) else 0.; s=np.sqrt(np.nanmean((c-l)**2)) if np.any(~np.isnan(c)) else 1.
        s = s if s>0 else 1e-5; locs.append(l); scs.append(s)
    return np.array(locs)[:,None,None], np.array(scs)[:,None,None]
res={}
for cfg,v in d.items():
    Q=B[cfg]; y=v['label'].astype(float); l,s=norm(v)
    Qn=np.arcsinh((Q-l)/s); med=Qn[:,:,4:5]
    ok=np.array([not np.all(np.isnan(c[-2048:])) for c in v['ctx']])
    b=crps(y.reshape(-1),Q.reshape(-1,9))
    # remove all-nan windows: replace their forecast by 0
    Q2=Q.copy(); Q2[~ok]=0
    r_nan=crps(y.reshape(-1),Q2.reshape(-1,9))/b
    best_shift=min((crps(y.reshape(-1),(np.sinh(Qn+dl)*s+l).reshape(-1,9))/b,dl) for dl in np.linspace(-0.3,0.3,13))
    best_sp=min((crps(y.reshape(-1),(np.sinh(med+sc*(Qn-med))*s+l).reshape(-1,9))/b,sc) for sc in np.linspace(0.7,1.4,15))
    res[cfg]=(r_nan,best_shift,best_sp)
for i,name in enumerate(['nan-fix','best shift','best spread']):
    vals=np.array([r[i][0] if i else r[i] for r in res.values()])
    print(name,'geomean',np.exp(np.mean(np.log(vals))).round(4))
print('shift choices:',[ (k,round(r[1][1],2), round(r[1][0],3)) for k,r in res.items() if r[1][0]<0.97])
print('spread choices:',[ (k,round(r[2][1],2), round(r[2][0],3)) for k,r in res.items() if r[2][0]<0.97])
