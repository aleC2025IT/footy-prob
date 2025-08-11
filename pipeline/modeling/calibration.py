import numpy as np, bisect
def pav_isotonic(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if w is None: w=np.ones_like(y)
    else: w=np.asarray(w,float)
    o=np.argsort(x); x,y,w=x[o],y[o],w[o]
    yh=y.copy(); wh=w.copy(); i=0
    while i<len(yh)-1:
        if yh[i]>yh[i+1]:
            tot=wh[i]+wh[i+1]; avg=(yh[i]*wh[i]+yh[i+1]*wh[i+1])/tot
            yh[i]=yh[i+1]=avg; wh[i]=wh[i+1]=tot
            j=i-1
            while j>=0 and yh[j]>yh[j+1]:
                tot=wh[j]+wh[j+1]; avg=(yh[j]*wh[j]+yh[j+1]*wh[j+1])/tot
                yh[j]=yh[j+1]=avg; wh[j]=wh[j+1]=tot; j-=1
            i=max(j+1,0)
        else:
            i+=1
    return x,yh
def bin_and_fit(p,y,bins=20):
    p=np.asarray(p,float); y=np.asarray(y,float)
    o=np.argsort(p); p=p[o]; y=y[o]; n=len(p)
    if n==0: return np.array([0,1]), np.array([0,1])
    edges=np.linspace(0,n,bins+1,dtype=int)
    xs=[]; ys=[]; ws=[]
    for i in range(bins):
        a,b=edges[i],edges[i+1]
        if b<=a: continue
        xs.append(p[a:b].mean()); ys.append(y[a:b].mean()); ws.append(b-a)
    xs,ys,ws=np.asarray(xs),np.asarray(ys),np.asarray(ws)
    xf,yf=pav_isotonic(xs,ys,ws)
    xf=np.concatenate(([0.0],xf,[1.0])); yf=np.concatenate(([yf[0]],yf,[yf[-1]]))
    return xf,yf
def apply_isotonic(xs,ys,p):
    p=float(p)
    if p<=xs[0]: return float(ys[0])
    if p>=xs[-1]: return float(ys[-1])
    j=bisect.bisect_right(xs,p)-1; j=max(0, min(j,len(xs)-2))
    x0,x1=xs[j],xs[j+1]; y0,y1=ys[j],ys[j+1]
    t=0 if x1==x0 else (p-x0)/(x1-x0)
    return float(y0+t*(y1-y0))