import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split, seasonal_errors
from diagnose import stats
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
taus = np.arange(1,10)/10
def crps(y,Q):
    ok=~np.isnan(y); y=y[ok]; Q=Q[ok]
    ql = 2*np.abs((y[:,None]-Q)*((y[:,None]<=Q)-taus))
    return ql.sum(0).mean()/np.abs(y).sum()*len(y) if False else float(np.mean(ql.sum(0)/np.abs(y).sum()))
rows=[]
for cfg,v in d.items():
    Q=B[cfg]; y=v['label'].astype(float)
    # normalized space
    locs=[];scs=[]
    for c in v['ctx']:
        c=c[-2048:].astype(np.float64); c=c[~np.isnan(c)]
        l=c.mean() if len(c) else 0; s=np.sqrt(np.mean((c-l)**2)) if len(c) else 1
        s = s if s>0 else 1e-5
        locs.append(l); scs.append(s)
    locs=np.array(locs)[:,None,None]; scs=np.array(scs)[:,None,None]
    Qn=np.arcsinh((Q-locs)/scs); yn=np.arcsinh((y[:,:,None]-locs)/scs)[:,:,0]
    base=crps(y.reshape(-1),Q.reshape(-1,9))
    med=Qn[:,:,4:5]
    res={}
    for s in [0.8,0.9,1.1,1.2,1.3,1.5]:
        Qs=np.sinh(med+s*(Qn-med))*scs+locs
        res[s]=crps(y.reshape(-1),Qs.reshape(-1,9))/base
    st=stats(y.reshape(-1),Q.reshape(-1,9),np.repeat(seasonal_errors(v['ctx'],v['m']),v['pl']))
    rows.append((cfg,base,st['below_q10'],st['below_q50'],st['below_q90'],st['bias'],res))
print(f"{'cfg':42s} crps   bq10 bq50 bq90  bias  | spread scale ratios 0.8 0.9 1.1 1.2 1.3 1.5")
for r in rows:
    print(f"{r[0]:42s} {r[1]:.3f} {r[2]:.2f} {r[3]:.2f} {r[4]:.2f} {r[5]:+.2f} | "+" ".join(f"{x:.3f}" for x in r[6].values()))
for s in [0.8,0.9,1.1,1.2,1.3,1.5]:
    g=np.exp(np.mean([np.log(r[6][s]) for r in rows])); print('global spread',s,'geomean',round(g,4))
a=np.array([[r[2],r[3],r[4]] for r in rows]); print('median coverage below q10/q50/q90',np.median(a,0), 'mean',a.mean(0))
pickle.dump(rows,open('astra/work/base_rows.pkl','wb'))
