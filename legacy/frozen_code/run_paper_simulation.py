import numpy as np, pandas as pd, math, json, time
from scipy.optimize import minimize
from scipy.stats import t as student_t
from pathlib import Path

SEED=20260726
OUT=Path('/mnt/data/Po_Valley_Paper_Simulation')
OUT.mkdir(exist_ok=True)


def softmax_masked(scores, mask):
    s=np.where(mask,scores,-1e9)
    s=s-np.max(s,axis=-1,keepdims=True)
    w=np.exp(s)*mask
    den=w.sum(axis=-1,keepdims=True)
    return np.divide(w,den,out=np.zeros_like(w),where=den>0)


def pair_mats(coords,z,xday,idx_tgt,idx_src):
    dc=coords[idx_tgt,None,:]-coords[idx_src][None,:,:]
    d=np.sqrt((dc*dc).sum(2))
    dz=np.abs(z[idx_tgt,None]-z[idx_src][None,:])
    dx=np.abs(xday[idx_tgt,None]-xday[idx_src][None,:])
    return d,dz,dx


def graph_predict(resid_source, coords,z,x, targets,sources, pars):
    # resid_source T x N with source residuals available
    ad,az,ax,shrink=pars
    T=resid_source.shape[0]
    pred=np.zeros((T,len(targets)))
    spread=np.zeros_like(pred)
    for t in range(T):
        d,dz,dx=pair_mats(coords,z,x[t],targets,sources)
        scores=-(ad*d+az*dz+ax*dx)
        mask=np.isfinite(resid_source[t,sources])[None,:] & (np.asarray(targets)[:,None]!=np.asarray(sources)[None,:])
        w=softmax_masked(scores,mask)
        rv=np.nan_to_num(resid_source[t,sources],nan=0.0)
        raw=w@rv
        pred[t]=shrink*raw
        spread[t]=np.sqrt(np.sum(w*(rv[None,:]-raw[:,None])**2,axis=1)+0.10)
    return pred,spread


def fit_graph(resid_cal,coords,z,xcal,sources):
    src=np.asarray(sources)
    T=resid_cal.shape[0]; ns=len(src)
    D=np.empty((T,ns,ns)); Z=np.empty_like(D); X=np.empty_like(D)
    for t in range(T):
        D[t],Z[t],X[t]=pair_mats(coords,z,xcal[t],src,src)
    mask=np.ones((T,ns,ns),dtype=bool)
    mask &= ~np.eye(ns,dtype=bool)[None,:,:]
    y=resid_cal[:,src]
    valid=np.isfinite(y); mask &= valid[:,None,:]
    rv=np.nan_to_num(y,nan=0.0)
    best=(np.inf,None)
    # Small, fixed tuning grid chosen before simulation; avoids unstable optimizer behavior.
    for ad in [2.0,4.0,6.0]:
      for az in [0.0,1.25,2.5]:
       for ax in [0.0,1.0,2.0]:
        scores=-(ad*D+az*Z+ax*X)
        w=softmax_masked(scores,mask)
        raw=np.einsum('tij,tj->ti',w,rv)
        for shrink in [0.75,1.0]:
            pred=shrink*raw
            err=(pred-y)[valid]
            mse=float(np.mean(err*err)+0.001*(az*az+ax*ax))
            if mse<best[0]: best=(mse,np.array([ad,az,ax,shrink]))
    return best[1]

def rbf_predict(resid_source,coords,targets,sources,h):
    T=resid_source.shape[0]
    dc=coords[np.asarray(targets),None,:]-coords[np.asarray(sources)][None,:,:]
    d=np.sqrt((dc*dc).sum(2))
    base=np.exp(-(d/h)**2)
    pred=np.zeros((T,len(targets))); spread=np.zeros_like(pred)
    for t in range(T):
        mask=np.isfinite(resid_source[t,sources])[None,:] & (np.asarray(targets)[:,None]!=np.asarray(sources)[None,:])
        w=base*mask; den=w.sum(1,keepdims=True); w=np.divide(w,den,out=np.zeros_like(w),where=den>0)
        rv=np.nan_to_num(resid_source[t,sources],nan=0.0)
        raw=w@rv; pred[t]=raw
        spread[t]=np.sqrt(np.sum(w*(rv[None,:]-raw[:,None])**2,axis=1)+0.10)
    return pred,spread


def fit_rbf(resid_cal,coords,xcal,sources):
    hs=[.08,.12,.18,.25,.35,.50,.75]
    src=np.asarray(sources); y=resid_cal[:,src]; valid=np.isfinite(y)
    best=None
    for h in hs:
        p,_=rbf_predict(resid_cal,coords,src,src,h)
        mse=np.mean((p[valid]-y[valid])**2)
        if best is None or mse<best[0]: best=(mse,h)
    return best[1]


def gen_rep(rng, adaptive=True, target_scale_shift=False):
    N=50; nsrc=40; Tcal=50; Ttest=70; T=Tcal+Ttest
    coords=rng.uniform(0,1,(N,2))
    z=np.sin(2*np.pi*coords[:,0])+0.7*np.cos(2*np.pi*coords[:,1])+rng.normal(0,.15,N)
    tt=np.arange(T)
    seasonal=np.sin(2*np.pi*tt/35)+0.5*np.cos(2*np.pi*tt/17)
    x=seasonal[:,None]+0.8*coords[None,:,0]-0.5*coords[None,:,1]+rng.normal(0,.35,(T,N))
    mean=18+3*z[None,:]+2*x+1.2*x*x+2*np.sin(2*np.pi*tt[:,None]/30)
    resid=np.empty((T,N))
    for t in range(T):
        dc=coords[:,None,:]-coords[None,:,:]
        d=np.sqrt((dc*dc).sum(2))
        dz=np.abs(z[:,None]-z[None,:])
        dx=np.abs(x[t,:,None]-x[t,None,:])
        if adaptive:
            K=np.exp(-4.0*d-1.2*dz-0.8*dx)
        else:
            K=np.exp(-4.0*d)
        K=0.82*K+0.18*np.eye(N)
        L=np.linalg.cholesky(K+1e-6*np.eye(N))
        g=L@rng.normal(size=N)
        nu=5
        common=np.sqrt(nu/rng.chisquare(nu))
        scale=2.2*(0.75+0.25*np.abs(x[t]))
        if target_scale_shift:
            scale[40:]*=1.35
        resid[t]=scale*g*common
    y=mean+resid
    return coords,z,x,mean,resid,y,Tcal


def station_block_factor(err,base_scale,station_axis=True,alpha=.10,finite=False):
    score=np.abs(err)/np.maximum(base_scale,0.15)
    qnom=student_t.ppf(1-alpha/2,5)
    if station_axis:
        qs=np.quantile(score,1-alpha,axis=0)
        m=len(qs)
        level=1-alpha
        if finite: level=min(1.0,math.ceil((m+1)*(1-alpha))/m)
        fac=np.quantile(qs,level,method='higher')/qnom
    else:
        fac=np.quantile(score,1-alpha)/qnom
    return fac


def crps_mc(y,mu,scale,nu=5,S=1000,seed=1):
    rng=np.random.default_rng(seed)
    z1=rng.standard_t(nu,S); z2=rng.standard_t(nu,S)
    c=.5*np.mean(np.abs(z1-z2))
    zz=(y-mu)/scale
    # flatten chunks
    out=np.empty(zz.size); flat=zz.ravel(); sf=scale.ravel()
    for a in range(0,len(flat),1000):
        b=min(a+1000,len(flat))
        out[a:b]=sf[a:b]*(np.mean(np.abs(flat[a:b,None]-z1[None,:]),axis=1)-c)
    return out.reshape(zz.shape)


def run_one(rep,adaptive,source_n,target_scale_shift=False):
    rng=np.random.default_rng(SEED+10000*int(adaptive)+100*source_n+rep+500000*int(target_scale_shift))
    coords,z,x,mean,resid,y,Tcal=gen_rep(rng,adaptive,target_scale_shift)
    allsrc=np.arange(40); targets=np.arange(40,50)
    # spatially balanced-ish random subset, same per whole period
    if source_n<40:
        # choose by farthest point seeded random
        first=rng.choice(allsrc); chosen=[first]
        while len(chosen)<source_n:
            dmin=np.min(np.sqrt(((coords[allsrc,None,:]-coords[np.array(chosen)][None,:,:])**2).sum(2)),axis=1)
            dmin[np.isin(allsrc,chosen)]=-1
            jitter=rng.uniform(0,.08,len(allsrc))
            chosen.append(int(allsrc[np.argmax(dmin+jitter)]))
        sources=np.array(sorted(chosen))
    else: sources=allsrc
    rcal=resid[:Tcal]; rtest=resid[Tcal:]
    pars=fit_graph(rcal,coords,z,x[:Tcal],sources)
    h=fit_rbf(rcal,coords,x[:Tcal],sources)
    pg_cal,sg_cal=graph_predict(rcal,coords,z,x[:Tcal],sources,sources,pars)
    pr_cal,sr_cal=rbf_predict(rcal,coords,sources,sources,h)
    pg,sg=graph_predict(rtest,coords,z,x[Tcal:],targets,sources,pars)
    pr,sr=rbf_predict(rtest,coords,targets,sources,h)
    truth=rtest[:,targets]
    # calibration from leave-one-source predictions; station block and pooled
    eg_cal=rcal[:,sources]-pg_cal; er_cal=rcal[:,sources]-pr_cal
    fg_row=station_block_factor(eg_cal,sg_cal,False); fg_block=station_block_factor(eg_cal,sg_cal,True)
    fr_row=station_block_factor(er_cal,sr_cal,False); fr_block=station_block_factor(er_cal,sr_cal,True)
    q=student_t.ppf(.95,5)
    rows=[]
    fg_finite=station_block_factor(eg_cal,sg_cal,True,finite=True)
    fr_finite=station_block_factor(er_cal,sr_cal,True,finite=True)
    for model,p,s,fr,fb,ff in [('Graph',pg,sg,fg_row,fg_block,fg_finite),('RBF',pr,sr,fr_row,fr_block,fr_finite)]:
        e=truth-p
        for cal,fac in [('row',fr),('station_block',fb),('station_block_finite',ff)]:
            scale=s*fac
            lo=p-q*scale; hi=p+q*scale
            cr=crps_mc(truth,p,scale,5,120,SEED+rep+source_n)
            rows.append({'Rep':rep,'Adaptive':adaptive,'SourceN':source_n,'ScaleShift':target_scale_shift,'Model':model,'Calibration':cal,
                         'RMSE':float(np.sqrt(np.mean(e*e))),'MAE':float(np.mean(np.abs(e))),
                         'CRPS':float(np.mean(cr)),'Cov90':float(np.mean((truth>=lo)&(truth<=hi))),
                         'Width90':float(np.mean(hi-lo)),'h_or_ad':float(h if model=='RBF' else pars[0]),
                         'az':float(0 if model=='RBF' else pars[1]),'ax':float(0 if model=='RBF' else pars[2])})
    return rows

if __name__=='__main__':
    t0=time.time(); rows=[]
    reps=50
    for adaptive in [False,True]:
        for source_n in [40,20,10]:
            for rep in range(reps):
                rows += run_one(rep,adaptive,source_n,False)
            print('done',adaptive,source_n,'elapsed',time.time()-t0,flush=True)
    # calibration shift scenarios are generated in a separate script to keep runtime bounded
    df=pd.DataFrame(rows); df.to_csv(OUT/'simulation_replicates.csv',index=False)
    summary=(df.groupby(['Adaptive','SourceN','ScaleShift','Model','Calibration'])
             .agg(RMSE=('RMSE','mean'),RMSE_SD=('RMSE','std'),MAE=('MAE','mean'),MAE_SD=('MAE','std'),CRPS=('CRPS','mean'),CRPS_SD=('CRPS','std'),Cov90=('Cov90','mean'),Cov90_SD=('Cov90','std'),Width90=('Width90','mean'),Width90_SD=('Width90','std'),az=('az','mean'),ax=('ax','mean')).reset_index())
    summary.to_csv(OUT/'simulation_summary.csv',index=False)
    # paired graph-rbf differences by scenario and calibration
    paired=[]
    for keys,g in df.groupby(['Adaptive','SourceN','ScaleShift','Calibration']):
        w=g.pivot(index='Rep',columns='Model',values=['RMSE','MAE','CRPS','Cov90','Width90'])
        for metric in ['RMSE','MAE','CRPS','Cov90','Width90']:
            d=w[metric]['Graph']-w[metric]['RBF']
            paired.append({'Adaptive':keys[0],'SourceN':keys[1],'ScaleShift':keys[2],'Calibration':keys[3],'Metric':metric,'MeanDiff':d.mean(),'Q025':d.quantile(.025),'Q975':d.quantile(.975),'PGraphLower':float(np.mean(d<0))})
    pd.DataFrame(paired).to_csv(OUT/'simulation_paired.csv',index=False)
    print(summary.to_string(index=False)); print('elapsed',time.time()-t0)
