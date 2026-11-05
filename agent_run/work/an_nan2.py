import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
for c in ['electricity/H/long','electricity/D/short','bitbrains_rnd/5T/long','bitbrains_rnd/5T/medium','electricity/H/medium','electricity/W/short','kdd_cup_2018/H/long']:
    v=d[c]
    for i,x in enumerate(v['ctx']):
        nanfrac=np.mean(np.isnan(x[-2048:]))
        if nanfrac>0.5:
            y=v['label'][i]; 
            first = np.argmax(~np.isnan(x)) if np.any(~np.isnan(x)) else -1
            lastobs = len(x)-1-np.argmax(~np.isnan(x[::-1])) if np.any(~np.isnan(x)) else -1
            print(f'{c:28s} item {v["item"][i]:4d} win {v["window"][i]} ctxlen {len(x)} nan(last2048) {nanfrac:.2f} lastobs idx {lastobs} label nonnan {np.mean(~np.isnan(y)):.2f} first nonnan label at {np.argmax(~np.isnan(y)) if np.any(~np.isnan(y)) else -1}')
