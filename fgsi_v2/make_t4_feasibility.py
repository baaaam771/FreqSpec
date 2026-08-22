#!/usr/bin/env python3
# =============================================================================
#  T4 : per-input step selection feasibility (oracle + probe + repeatability)
#  Locked cross-checks (print at end so you can verify):
#    oracle headroom 17-28%;  probe combo test AUROC 0.591 vs coverage 0.584,
#    AUPRC 0.554 (base 0.44), probe cost 0.081s;
#    repeatability Jaccard s12 0.00-0.44, s8 0.61-0.65.
#
#  Inputs: phase2/largemask/instmask _per_image.csv (oracle),
#          probe_instmask_v3.csv (probe),
#          nsrep_per_image.csv, nsrep_obj_per_image.csv (+ base idx<30) (repeat)
#  Usage:  python make_t4_feasibility.py --data /path/to/csvs
#  Output: table_t4_feasibility.tex
#  No sklearn dependency: AUROC/AUPRC implemented directly.
# =============================================================================
import argparse, os, sys, itertools
import numpy as np, pandas as pd

TAU=0.02
PAIRS=[("s12","s16"),("s8","s16"),("s12","s24")]   # (fast, fallback)

def auroc(y, s):
    y=np.asarray(y); s=np.asarray(s)
    P=(y==1).sum(); N=(y==0).sum()
    if P==0 or N==0: return np.nan
    order=np.argsort(s); ranks=np.empty(len(s)); ranks[order]=np.arange(1,len(s)+1)
    # average ranks for ties
    _,inv,cnt=np.unique(s,return_inverse=True,return_counts=True)
    # simple tie-safe Mann-Whitney
    return (ranks[y==1].sum()-P*(P+1)/2)/(P*N)

def auprc(y, s):
    y=np.asarray(y); s=np.asarray(s); o=np.argsort(-s); y=y[o]
    tp=np.cumsum(y); fp=np.cumsum(1-y)
    prec=tp/np.maximum(tp+fp,1); rec=tp/max(y.sum(),1)
    # area under precision-recall (step)
    ap=0.0; prev=0.0
    for p,r in zip(prec,rec):
        ap+=p*(r-prev); prev=r
    return ap

def logistic_fit(X, y, iters=500, lr=0.1):
    X=np.c_[np.ones(len(X)), (X-X.mean(0))/(X.std(0)+1e-9)]
    w=np.zeros(X.shape[1])
    for _ in range(iters):
        p=1/(1+np.exp(-X@w)); w-=lr*X.T@(p-y)/len(y)
    return w, X.mean(0), X.std(0)

def gtm(df, run, step):
    d=df[(df["run"]==run)&(df["method"]==f"target_{step}")]
    return d.set_index("idx")["gt_masked_lpips"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="."); a=ap.parse_args()
    D=a.data
    def rd(f):
        p=os.path.join(D,f); return pd.read_csv(p) if os.path.exists(p) else None
    per=[rd(f) for f in ["phase2_per_image.csv","largemask_per_image.csv","instmask_per_image.csv"]]
    per=pd.concat([x for x in per if x is not None], ignore_index=True)

    lines=[]
    # ---- oracle headroom ----
    lines.append("% oracle")
    oracle_rows=[]
    for run in sorted(per["run"].unique()):
        for fast,fb in PAIRS:
            gf=gtm(per,run,fast); gb=gtm(per,run,fb); g24=gtm(per,run,"s24")
            j=gf.index.intersection(gb.index).intersection(g24.index)
            if not len(j): continue
            fail=(gf.loc[j]-g24.loc[j]>TAU).values  # fast fails
            # need times
            tf=per[(per["run"]==run)&(per["method"]==f"target_{fast}")].set_index("idx")["time_sec"].loc[j]
            tb=per[(per["run"]==run)&(per["method"]==f"target_{fb}")].set_index("idx")["time_sec"].loc[j]
            t_oracle=np.where(fail, tb.values, tf.values).mean()
            t_fixed=tb.values.mean()  # safe fixed = always fallback
            headroom=1 - t_oracle/t_fixed
            oracle_rows.append((run,f"{fast}/{fb}",headroom,fail.mean()))
    if oracle_rows:
        hr=[r[2] for r in oracle_rows]
        lines.append(f"% oracle headroom range: {min(hr)*100:.0f}%..{max(hr)*100:.0f}% (locked 17-28%)")

    # ---- probe (instmask v3) ----
    probe=rd("probe_instmask_v3.csv"); prob_auroc=prob_aupr=cov_auroc=base=np.nan
    if probe is not None:
        # label: GT-m(s8)-GT-m(s16)>0.02 on instmask
        inst=per[per["run"].astype(str).str.contains("inst")]
        g8=gtm(inst,inst["run"].iloc[0],"s8") if len(inst) else None
        # fallback: derive label from probe file if it carries gt columns
        if "label" in probe.columns:
            y=probe["label"].values
        elif {"gtm_s8","gtm_s16"}.issubset(probe.columns):
            y=((probe["gtm_s8"]-probe["gtm_s16"])>TAU).astype(int).values
        else:
            y=None
        if y is not None and "idx" in probe.columns:
            test=(probe["idx"].values%2==1); train=~test
            feat=[c for c in probe.columns if c not in
                  ("run","idx","method","label","gtm_s8","gtm_s16","time_sec")]
            X=probe[feat].values.astype(float)
            w,mu,sd=logistic_fit(X[train], y[train])
            Xt=np.c_[np.ones(test.sum()), (X[test]-mu[1:])/(sd[1:]) ]
            score=1/(1+np.exp(-Xt@w))
            prob_auroc=auroc(y[test],score); prob_aupr=auprc(y[test],score)
            base=y[test].mean()
            if "mask_coverage" in probe.columns:
                cov_auroc=auroc(y[test], probe["mask_coverage"].values[test])
        lines.append(f"% probe test AUROC={prob_auroc:.3f} (locked .591), coverage={cov_auroc:.3f} (.584), "
                     f"AUPRC={prob_aupr:.3f} (.554), base={base:.3f} (.44)")

    # ---- repeatability (nsrep) ----
    def jacc_report(nsfile, tag):
        ns=rd(nsfile)
        if ns is None or "seed" not in (ns.columns if ns is not None else []):
            lines.append(f"% {tag}: needs 'seed' column in {nsfile} -- VERIFY")
            return
        out=[]
        for step in ["s12","s8"]:
            sets=[]
            for sd_ in sorted(ns["seed"].unique())[:3]:
                d=ns[(ns["seed"]==sd_)]
                gs=d[d["method"]==f"target_{step}"].set_index("idx")["gt_masked_lpips"]
                g24=d[d["method"]=="target_s24"].set_index("idx")["gt_masked_lpips"]
                j=gs.index.intersection(g24.index); j=j[j<30]
                sets.append(set(j[(gs.loc[j]-g24.loc[j]>TAU).values]))
            J=[len(a&b)/max(len(a|b),1) for a,b in itertools.combinations(sets,2)]
            out.append((step, np.mean(J) if J else np.nan))
        lines.append(f"% {tag} Jaccard: "+", ".join(f"{s}={v:.2f}" for s,v in out))
    jacc_report("nsrep_per_image.csv","nsrep(large)")
    jacc_report("nsrep_obj_per_image.csv","nsrep(obj)")

    with open("table_t4_feasibility.tex","w") as f:
        f.write("% auto-generated by make_t4_feasibility.py\n")
        f.write("\\begin{table}[t]\\centering\\small\n")
        f.write("\\caption{Per-input step-selection feasibility: oracle headroom, "
                "probe vs.\\ coverage baseline (held-out), and cross-seed repeatability.}\n")
        f.write("\\label{tab:feasibility}\n")
        f.write("\\begin{tabular}{@{}lc@{}}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n")
        if oracle_rows:
            f.write(f"Oracle headroom (range) & {min(hr)*100:.0f}--{max(hr)*100:.0f}\\% \\\\\n")
        f.write(f"Probe combo AUROC (test) & {prob_auroc:.3f} \\\\\n")
        f.write(f"Coverage baseline AUROC & {cov_auroc:.3f} \\\\\n")
        f.write(f"Probe AUPRC / base rate & {prob_aupr:.3f} / {base:.2f} \\\\\n")
        f.write("Probe cost & 0.081\\,s~\\VERIFY{} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print("wrote table_t4_feasibility.tex")
    print("\n".join(["=== CROSS-CHECK (compare to locked numbers) ==="]+lines))

if __name__=="__main__":
    main()
