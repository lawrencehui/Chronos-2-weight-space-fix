import csv, sys, numpy as np
EMPTY={'electricity/H/long','bitbrains_rnd/5T/long','bitbrains_rnd/5T/medium','electricity/D/short'}
for f in sys.argv[1:]:
    R=list(csv.DictReader(open(f)))
    r={x['config']:float(x['crps_ratio']) for x in R}
    keys=[k for k in r if k not in EMPTY]
    lg=np.array([np.log(r[k]) for k in keys])
    ds=sorted(set(k.split('/')[0] for k in keys)); A=set(ds[::2])
    la=np.array([np.log(r[k]) for k in keys if k.split('/')[0] in A]); lb=np.array([np.log(r[k]) for k in keys if k.split('/')[0] not in A])
    print(f"{f.split('/')[-1]:40s} all {np.exp(np.mean([np.log(v) for v in r.values()])):.4f} | non-empty {np.exp(lg.mean()):.4f} better {(lg<-.005).sum()} worse {(lg>.005).sum()} | halfA {np.exp(la.mean()):.4f} halfB {np.exp(lb.mean()):.4f}")
