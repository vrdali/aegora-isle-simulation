import sys,glob,os,json,pandas as pd,numpy as np
def first_mtsc(csv,crit,burn=50):
    c=pd.read_csv(csv,usecols=['timestep','resource_stock','gdp_raw']); c=c[c.timestep>burn]
    s=c.timestep[c.resource_stock<crit]
    z=(c.gdp_raw<=0).astype(int).rolling(3).sum()==3
    zt=c.timestep[z]
    v=[x for x in (s.min() if len(s) else None, zt.min() if len(zt) else None) if x is not None and x==x]
    return (min(v) if v else None, s.min() if len(s) else None)
b,out=sys.argv[1],sys.argv[2]; rows=[]
for f in sorted(glob.glob(f'{b}/runs/*/*.csv')):
    if f.endswith('_wealth.csv') or f.endswith('_audit.csv'): continue
    rid=os.path.basename(f)[:-4]; j=json.load(open(f[:-4]+'.json')); crit=j['scenario_params']['critical_min']
    t,ts=first_mtsc(f,crit); rows.append(dict(run_id=rid,gov=rid.split('_')[0],scen=rid.split('_')[1],mtsc_any=t,mtsc_stock_only=ts))
pd.DataFrame(rows).to_csv(out,index=False)
d=pd.DataFrame(rows); print(d.groupby('gov').agg(n=('run_id','count'),any=('mtsc_any',lambda s:s.notna().mean()),stock=('mtsc_stock_only',lambda s:s.notna().mean())).round(3))
