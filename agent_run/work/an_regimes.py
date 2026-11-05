import numpy as np, sys
sys.path.insert(0,'/Users/lawrencehui/Dev/MLX/wsp/astra/work')
from common import *
d,B,rows=table()
feats=[]
for c,v in d.items():
    for i,x in enumerate(v['ctx']):
        x=x[-2048:]; ok=~np.isnan(x)
        if ok.sum()<2: feats.append(None); continue
        l,s=locscale(x); zz=(x[ok]-l)/s
        last=np.mean(zz[-16:]); 
        feats.append(dict(maxz=np.abs(zz).max(), zf=np.mean(x[ok]==0), last=last, nobs=ok.sum(), nop=int(np.ceil(v['pl']/16)),
                          rough=np.mean(np.abs(np.diff(zz))) ))
regimes={'spiky maxz>8':lambda f:f['maxz']>8,'zeros>50%':lambda f:f['zf']>0.5,'last z>1.5':lambda f:f['last']>1.5,'last z<-1.5':lambda f:f['last']<-1.5,
         'rough>1':lambda f:f['rough']>1,'smooth<0.1':lambda f:f['rough']<0.1,'nop>=30':lambda f:f['nop']>=30,'nobs<200':lambda f:f['nobs']<200}
for name,fn in regimes.items():
    out=[]
    for kind,vals in [('shift',[-0.1,0.1]),('spread',[0.85,1.15])]:
        for val in vals:
            logs_e=[];logs_l=[];cnt=0;k=0
            for c,v in d.items():
                Y=v['label'].astype(float); Q=B[c]; W=v['window']; Q1=Q.copy(); hit=np.zeros(len(Q),bool)
                for i in range(len(Q)):
                    f=feats[k+i]
                    if f is None or not fn(f): continue
                    l,s=locscale(v['ctx'][i]); Qn=np.arcsinh((Q[i]-l)/s); med=Qn[:,4:5]
                    Qe=Qn+val if kind=='shift' else med+val*(Qn-med)
                    Q1[i]=np.sinh(Qe)*s+l; hit[i]=True
                k+=len(Q)
                if not hit.any(): continue
                cnt+=hit.sum()
                for sel,lst in [(W<W.max(),logs_e),(W==W.max(),logs_l)]:
                    y=Y[sel].reshape(-1)
                    if np.all(np.isnan(y)) or not hit[sel].any(): continue
                    lst.append(np.log(crps(y,Q1[sel].reshape(-1,9))/crps(y,Q[sel].reshape(-1,9))))
            out.append(f'{kind}{val}: early {np.exp(np.sum(logs_e)/96):.4f} last {np.exp(np.sum(logs_l)/96):.4f}')
    print(f'{name:14s} windows {cnt:4d} | '+' | '.join(out))
