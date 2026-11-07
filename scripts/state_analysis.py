"""State-based collapse + Trader wealth from per-agent wealth logs. Usage: state_analysis.py <batch_dir> <out_csv>"""
import sys, glob, os, json
import pandas as pd, numpy as np
INIT_MEAN=10.0; THR=0.5*INIT_MEAN
def run(batch, out):
    rows=[]
    for w in sorted(glob.glob(os.path.join(batch,'runs','*','*_wealth.csv'))):
        rid=os.path.basename(w).replace('_wealth.csv','')
        gov,scen=rid.split('_')[0],rid.split('_')[1]
        df=pd.read_csv(w); df=df[df.in_indicator_scope==True]
        flags=[];last=None
        for t,g in df.groupby('timestep'):
            k=max(1,int(np.ceil(0.25*len(g)))); bq=np.sort(g.wealth.values)[:k].mean()
            flags.append(bq<THR); last=(t,g,bq)
        t,g,bq=last
        tr=g[g.agent_type=='trader'].wealth; pr=g[g.agent_type=='producer'].wealth; ex=g[g.agent_type=='extractor'].wealth
        j=json.load(open(w.replace('_wealth.csv','.json')))
        first=next((i for i,f in enumerate(flags) if f),None)
        rows.append(dict(batch=os.path.basename(batch.rstrip('/')),run_id=rid,gov=gov,scen=scen,
            frac_collapsed_state=float(np.mean(flags)),final_collapsed=int(flags[-1]),final_bottom_q_mean=bq,
            first_state_step=(None if first is None else (first+1)*100),
            trader_mean_final=tr.mean() if len(tr) else np.nan, trader_frac_le0_final=(tr<=0.01).mean() if len(tr) else np.nan,
            producer_mean_final=pr.mean() if len(pr) else np.nan, extractor_mean_final=ex.mean() if len(ex) else np.nan,
            mtsc=int(bool(j['collapse_occurred']))))
    pd.DataFrame(rows).to_csv(out,index=False); return len(rows)
if __name__=='__main__': print(run(sys.argv[1],sys.argv[2]))
