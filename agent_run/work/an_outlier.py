import pickle, numpy as np, sys
sys.path.insert(0,'.')
from dev_eval import load_split
d = load_split('fit'); B = pickle.load(open('astra/base_fit.pkl','rb'))
taus = np.arange(1,10)/10
for cfg in ['electricity/H/long','bitbrains_rnd/5T/long','bitbrains_rnd/5T/medium','electricity/D/short','solar/W/short']:
    v=d[cfg]; Q=B[cfg]; y=v['label'].astype(float)
    ql = np.nansum(2*np.abs((y[:,:,None]-Q)*((y[:,:,None]<=Q)-taus)).sum(2),1)  # per window
    tot=np.nansum(np.abs(y))
    o=np.argsort(-ql)
    print(cfg, 'total crps',ql.sum()/tot/9)
    for i in o[:4]:
        c=v['ctx'][i][-2048:]; 
        print(f"  win {i} share {ql[i]/ql.sum():.2f} ctx mean {np.nanmean(c):.3g} std {np.nanstd(c):.3g} last {c[-3:]} | y mean {np.nanmean(y[i]):.3g} | q10 {Q[i,:,0].mean():.3g} q50 {Q[i,:,4].mean():.3g} q90 {Q[i,:,8].mean():.3g} max q90 {Q[i,:,8].max():.3g}")
