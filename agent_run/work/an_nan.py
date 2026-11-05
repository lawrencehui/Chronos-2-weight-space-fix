import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
taus = np.arange(1,10)/10
tot_cfg=0
for cfg,v in d.items():
    Q=B[cfg]; y=v['label'].astype(float)
    ql = np.nansum(2*np.abs((y[:,:,None]-Q)*((y[:,:,None]<=Q)-taus)).sum(2),1)
    allnan=[i for i,c in enumerate(v['ctx']) if np.all(np.isnan(c[-2048:]))]
    fr=[np.mean(np.isnan(c[-2048:])) for c in v['ctx']]
    mostly=[i for i,f in enumerate(fr) if 0.5<f<1]
    if allnan or mostly:
        tot_cfg+=1
        print(f"{cfg:40s} allnan {len(allnan):2d} share {ql[allnan].sum()/ql.sum():.2f} | >50%nan {len(mostly)} share {ql[mostly].sum()/ql.sum():.2f}  labels nan frac of allnan wins {np.mean(np.isnan(y[allnan])) if allnan else 0:.2f}")
print(tot_cfg)
