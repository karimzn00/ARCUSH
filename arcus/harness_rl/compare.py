from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr, gaussian_kde, mannwhitneyu
warnings.filterwarnings("ignore")

ENV_S  = ["resource_constraint","trust_violation","concept_drift",
          "observation_noise","sensor_blackout"]
RWD_S  = ["valence_inversion","reward_noise"]
ALL_S  = ENV_S + RWD_S
SCH_ORDER = ["baseline"] + ALL_S

SCH_SHORT = {
    "baseline":"BASE","resource_constraint":"RC","trust_violation":"TV",
    "valence_inversion":"VI","concept_drift":"CD","observation_noise":"ON",
    "sensor_blackout":"SB","reward_noise":"RN"}
SCH_COLOR = {
    "baseline":"#9E9E9E","resource_constraint":"#EF5350","trust_violation":"#AB47BC",
    "valence_inversion":"#FF7043","concept_drift":"#42A5F5","observation_noise":"#26A69A",
    "sensor_blackout":"#5C6BC0","reward_noise":"#FFCA28"}
CH_LABEL = {
    "competence":"Competence","coherence":"Policy Consistency",
    "continuity":"Temporal Stability","integrity":"Observation Reliability",
    "meaning":"Action Entropy Div."}
CH_KEYS = list(CH_LABEL.keys())
CH_COLOR = {
    "competence":"#EF5350","coherence":"#AB47BC","continuity":"#42A5F5",
    "integrity":"#66BB6A","meaning":"#FF7043"}
ENV_SUITE = {
    "CartPole-v1":"Classic","Acrobot-v1":"Classic","MountainCar-v0":"Classic",
    "FrozenLake-v1":"Classic","LunarLander-v3":"Classic",
    "MountainCarContinuous-v0":"Continuous","Pendulum-v1":"Continuous",
    "HalfCheetah-v4":"MuJoCo","Hopper-v4":"MuJoCo",
    "Walker2d-v4":"MuJoCo","Walker2D-v4":"MuJoCo",
    "ALE/Pong-v5":"Atari","ALE/SpaceInvaders-v5":"Atari"}
SUITE_COLOR={"Classic":"#42A5F5","Continuous":"#66BB6A","MuJoCo":"#EF5350","Atari":"#FF7043"}
ALGO_FAMILY={
    "ppo":"On-Policy","trpo":"On-Policy","a2c":"On-Policy",
    "sac":"Off-Policy AC","td3":"Off-Policy AC","ddpg":"Off-Policy AC",
    "dqn":"Value-Based","qrdqn":"Value-Based"}
FAM_COLOR={"On-Policy":"#42A5F5","Off-Policy AC":"#EF5350","Value-Based":"#66BB6A"}
DPI=220
plt.rcParams.update({
    "font.family":"sans-serif","font.size":10,"axes.titlesize":11,
    "axes.labelsize":10,"legend.fontsize":9,"xtick.labelsize":9,
    "ytick.labelsize":9,"axes.spines.top":False,"axes.spines.right":False})

def _sl(s):    return SCH_SHORT.get(s, s)
def _sc(s):    return SCH_COLOR.get(s, "#999")
def _suite(e): return ENV_SUITE.get(e, "Other")
def _col(*names, df): return next((n for n in names if n in df.columns), None)

def _present(df, pool):
    have = set(df["schedule"].unique()) if "schedule" in df.columns else set()
    return [s for s in SCH_ORDER if s in pool and s in have]

def _save(fig, path, label=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {(label or path.stem):55s} -> {path.name}")

def _save_dark(fig, path, label=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] {(label or path.stem):55s} -> {path.name}")

def _wtex(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [TABLE] {path.name}")

def bootstrap_r(x, y, n=2000, seed=42, method="pearson"):
    x, y = np.asarray(x,float), np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<4: return np.nan,np.nan,np.nan,np.nan
    fn = pearsonr if method=="pearson" else spearmanr
    r,p = fn(x,y)
    rng = np.random.default_rng(seed); rs=[]
    for _ in range(n):
        idx=rng.integers(0,len(x),len(x))
        try:
            rv,_=fn(x[idx],y[idx])
            if math.isfinite(rv): rs.append(float(rv))
        except: pass
    lo=float(np.percentile(rs,2.5)) if rs else np.nan
    hi=float(np.percentile(rs,97.5)) if rs else np.nan
    return float(r),float(p),lo,hi

def load_data(root=None, leaderboard_csv=None):
    if leaderboard_csv and Path(leaderboard_csv).exists():
        df=pd.read_csv(leaderboard_csv)
        print(f"[INFO] {len(df)} rows from {Path(leaderboard_csv).name}")
        return df
    if root:
        frames=[]
        for p in sorted(Path(root).rglob("eval/eval_results.csv")):
            try: frames.append(pd.read_csv(p))
            except: pass
        if not frames: raise FileNotFoundError(f"No eval_results.csv under {root}")
        df=pd.concat(frames,ignore_index=True)
        print(f"[INFO] {len(df)} rows from {len(frames)} files")
        return df
    raise ValueError("Provide --root or --leaderboard")

def load_per_episode(path):
    if path and Path(path).exists():
        df=pd.read_csv(path)
        print(f"[INFO] per_episode: {len(df)} rows from {Path(path).name}")
        return df
    return None

def aggregate(df):
    grp=["env","algo","schedule","eval_mode"]
    num=[c for c in df.select_dtypes(include=[np.number]).columns if c not in grp+["seed"]]
    return df.groupby(grp)[num].mean().reset_index()

def prepare(df):
    df=df.copy()
    if "is_reward_stressor" not in df.columns:
        df["is_reward_stressor"]=df["schedule"].isin(RWD_S).astype(int)
    df["suite"]      =df["env"].map(_suite).fillna("Other")
    df["algo_family"]=df["algo"].str.lower().map(ALGO_FAMILY).fillna("Other")
    rc=_col("reward_mean","reward_mean__mean",df=df)
    if rc:
        def _norm(g):
            x=g[rc].to_numpy(float); fin=x[np.isfinite(x)]
            if fin.size==0: return pd.Series(np.full(len(g),.5),index=g.index)
            lo,hi=fin.min(),fin.max()
            return pd.Series(np.full(len(g),.5) if hi-lo<1e-12 else (x-lo)/(hi-lo),index=g.index)
        df["reward_norm"]=df.groupby("env",dropna=False,sort=False).apply(_norm).reset_index(level=0,drop=True)
    elif "reward_norm" not in df.columns:
        df["reward_norm"]=np.nan
    if "leaderboard_score" not in df.columns:
        ic=_col("identity_mean","identity_mean__mean",df=df)
        cc=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
        if ic and cc:
            df["leaderboard_score"]=(0.55*df[ic].clip(0,1).fillna(0)
                +0.30*(1-df[cc].clip(0,1).fillna(0))
                +0.15*df["reward_norm"].clip(0,1).fillna(.5))
    return df

def _det_env(df):
    d=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df.copy()
    s=_present(d,ENV_S)
    return d[d["schedule"].isin(s)].copy()

def _det_all(df):
    d=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df.copy()
    s=_present(d,ALL_S)
    return d[d["schedule"].isin(s)].copy()

def run_corr(df, plots_dir):
    sc="leaderboard_score"; rw="reward_norm"
    env_df=_det_env(df).dropna(subset=[sc,rw])
    all_df=_det_all(df).dropna(subset=[sc,rw])
    rE,pE,loE,hiE     = bootstrap_r(env_df[sc].values, env_df[rw].values)
    rF,pF,loF,hiF     = bootstrap_r(all_df[sc].values, all_df[rw].values)
    rSp,pSp,loSp,hiSp = bootstrap_r(env_df[sc].values, env_df[rw].values,method="spearman")
    e2=env_df.copy()
    e2["rz"]=e2.groupby("env")[rw].transform(lambda x:(x-x.mean())/(x.std()+1e-8))
    rZ,pZ,loZ,hiZ = bootstrap_r(e2[sc].values,e2["rz"].values)
    na=env_df[~env_df["env"].str.contains("ALE",na=False)]
    at=env_df[ env_df["env"].str.contains("ALE",na=False)]
    rNA,pNA,loNA,hiNA=bootstrap_r(na[sc].values,na[rw].values) if len(na)>3 else (np.nan,)*4
    rAT,pAT,loAT,hiAT=bootstrap_r(at[sc].values,at[rw].values) if len(at)>3 else (np.nan,)*4
    stats={
        "primary":  {"r":rE,"p":pE,"ci":[loE,hiE],"n":int(len(env_df))},
        "secondary":{"r":rF,"p":pF,"ci":[loF,hiF],"n":int(len(all_df))},
        "spearman": {"r":rSp,"p":pSp,"ci":[loSp,hiSp]},
        "z_normed": {"r":rZ,"p":pZ,"ci":[loZ,hiZ]},
        "non_atari":{"r":rNA,"ci":[loNA,hiNA],"n":int(len(na))},
        "atari":    {"r":rAT,"ci":[loAT,hiAT],"n":int(len(at))}}
    print(f"\n{'='*65}\nCORRELATION ANALYSIS\n{'='*65}")
    print(f"PRIMARY  (env, n={len(env_df)}):  r={rE:+.3f}  [{loE:.3f},{hiE:.3f}]  p={pE:.2e}")
    print(f"SECONDARY(all, n={len(all_df)}):  r={rF:+.3f}  [{loF:.3f},{hiF:.3f}]  [VI/RN inflate]")
    print(f"Spearman (env):  r={rSp:+.3f}  [{loSp:.3f},{hiSp:.3f}]")
    print(f"Z-normed (env):  r={rZ:+.3f}  [{loZ:.3f},{hiZ:.3f}]")
    if math.isfinite(rNA): print(f"Non-Atari:  r={rNA:+.3f}  [{loNA:.3f},{hiNA:.3f}]")
    if math.isfinite(rAT): print(f"Atari-only: r={rAT:+.3f}  [{loAT:.3f},{hiAT:.3f}]")
    print(f"R2={rE**2:.4f}  =>  {100*(1-rE**2):.1f}% unexplained\n{'='*65}\n")
    sd=plots_dir/"stats"; sd.mkdir(parents=True,exist_ok=True)
    (sd/"corr.json").write_text(json.dumps(stats,indent=2,default=str))
    tex="\n".join([
        f"\\newcommand{{\\ArcusEnvR}}{{{rE:+.3f}}}",
        f"\\newcommand{{\\ArcusEnvCILow}}{{{loE:.3f}}}",
        f"\\newcommand{{\\ArcusEnvCIHigh}}{{{hiE:.3f}}}",
        f"\\newcommand{{\\ArcusEnvP}}{{{pE:.2e}}}",
        f"\\newcommand{{\\ArcusEnvN}}{{{len(env_df)}}}",
        f"\\newcommand{{\\ArcusFullR}}{{{rF:+.3f}}}",
        f"\\newcommand{{\\ArcusSpearmanR}}{{{rSp:+.3f}}}",
        f"\\newcommand{{\\ArcusZnormR}}{{{rZ:+.3f}}}",
        f"\\newcommand{{\\ArcusNonAtariR}}{{{rNA:+.3f}}}",
        f"\\newcommand{{\\ArcusAtariR}}{{{rAT:+.3f}}}",
        f"\\newcommand{{\\ArcusRsquared}}{{{rE**2:.4f}}}"])
    (sd/"corr.tex").write_text(tex)
    return stats, env_df, all_df

def fig01_methodology(plots_dir):
    fig,ax=plt.subplots(figsize=(12,4)); ax.axis("off")
    boxes=[(0.05,.35,.14,.30,"Trained\nPolicy $\\pi_\\theta$","#42A5F5"),
           (0.25,.35,.14,.30,"PRE PHASE\n(baseline)\n40 eps","#66BB6A"),
           (0.45,.35,.14,.30,"SHOCK PHASE\n(stressor)\n40 eps","#EF5350"),
           (0.65,.35,.14,.30,"POST PHASE\n(recovery)\n40 eps","#FF7043"),
           (0.85,.20,.12,.60,"5 Channels\n+ ARCUS Score $\\mathcal{L}$","#AB47BC")]
    for x,y,w,h,lbl,c in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02",
            lw=1.5,ec="white",fc=c,alpha=.85,transform=ax.transAxes))
        ax.text(x+w/2,y+h/2,lbl,transform=ax.transAxes,ha="center",va="center",
                fontsize=9,color="white",fontweight="bold")
    for xs,xe in [(0.19,.25),(0.39,.45),(0.59,.65),(0.79,.85)]:
        ax.annotate("",xy=(xe,.50),xytext=(xs,.50),xycoords="axes fraction",
                    textcoords="axes fraction",arrowprops=dict(arrowstyle="->",color="#555",lw=1.5))
    ax.text(.5,.08,"8 Stressors: RC TV CD ON SB (env-axis)  |  VI RN (feedback-axis)",
            transform=ax.transAxes,ha="center",fontsize=9,color="#555",
            bbox=dict(boxstyle="round,pad=0.3",fc="#f5f5f5",ec="#ccc"))
    ax.set_title("ARCUS-H: Three-Phase Behavioral Stability Protocol",fontsize=12,fontweight="bold",pad=10)
    fig.tight_layout(); _save(fig,plots_dir/"fig01_methodology.png","Fig01")

def fig02_degeneracy_heatmap(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    subs=_present(df,ALL_S)
    if not cr or not subs: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    piv=(det[det["schedule"].isin(subs)].groupby(["algo","schedule"])[cr].mean()
         .unstack("schedule").reindex(columns=subs))
    if piv.empty: return
    arr=piv.to_numpy(float)
    fig,ax=plt.subplots(figsize=(max(10,len(subs)*1.4),max(5,len(piv)*.7+2)))
    im=ax.imshow(arr,cmap="RdYlGn_r",aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(range(len(subs))); ax.set_xticklabels([_sl(s) for s in subs])
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    ne=[i for i,s in enumerate(subs) if s in ENV_S]
    if ne and ne[-1]+1<len(subs): ax.axvline(ne[-1]+.5,color="white",lw=2.5,ls="--")
    for i in range(len(piv)):
        for j in range(len(subs)):
            v=arr[i,j]
            if np.isfinite(v):
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8,
                        color="white" if v>.55 else "black")
    plt.colorbar(im,ax=ax,label="Policy degeneracy rate (shock phase)")
    ax.set_title("Policy degeneracy rate — algorithm x stressor",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig02_degeneracy_heatmap.png","Fig02")

def fig03_suite_collapse(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    subs=_present(df,ALL_S)
    if not cr or not subs: return
    df2=df.copy(); df2["suite"]=df2["env"].map(_suite).fillna("Other")
    det=df2[df2["eval_mode"]=="deterministic"] if "eval_mode" in df2.columns else df2
    suites=[s for s in ["Classic","Continuous","MuJoCo","Atari"] if s in det["suite"].unique()]
    if not suites: return
    fig,axes=plt.subplots(1,len(suites),figsize=(4.2*len(suites),5),sharey=True)
    if len(suites)==1: axes=[axes]
    for ax,suite in zip(axes,suites):
        sub=det[(det["suite"]==suite)&det["schedule"].isin(subs)]
        means=sub.groupby("schedule")[cr].mean().reindex(subs)
        sems=sub.groupby("schedule")[cr].sem().reindex(subs)
        ax.bar(range(len(subs)),means.fillna(0).values,yerr=sems.fillna(0).values,
               color=[_sc(s) for s in subs],capsize=4,error_kw={"lw":1.2})
        ax.axhline(.05,color="gray",ls="--",lw=.9,label="FPR 5%")
        ax.set_xticks(range(len(subs))); ax.set_xticklabels([_sl(s) for s in subs],rotation=35,ha="right")
        ax.set_ylim(0,1.05); ax.set_title(suite,fontweight="bold",color=SUITE_COLOR.get(suite,"black"))
        if suite==suites[0]: ax.set_ylabel("Policy degeneracy rate")
        ax.legend(fontsize=8)
    fig.suptitle("Policy degeneracy rate by stressor and environment suite",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig03_suite_collapse.png","Fig03")

def fig04_correlation_scatter(df,stats,env_df,all_df,plots_dir):
    sc="leaderboard_score"; rw="reward_norm"
    rE=stats["primary"]["r"]; loE,hiE=stats["primary"]["ci"]
    rF=stats["secondary"]["r"]; loF,hiF=stats["secondary"]["ci"]
    fig,axes=plt.subplots(1,2,figsize=(14,5.5))
    for ax,pdf,title,rv,lo,hi in [
        (axes[0],env_df,"PRIMARY — Env stressors only\n(VI/RN excluded — circularity avoided)",rE,loE,hiE),
        (axes[1],all_df,"SECONDARY — All stressors\n(VI/RN inflate r — not for main claims)",rF,loF,hiF)]:
        for suite,grp in pdf.groupby("suite"):
            ax.scatter(grp[rw],grp[sc],c=SUITE_COLOR.get(suite,"#999"),alpha=.55,s=22,label=suite)
        if len(pdf)>3:
            xs=np.sort(pdf[rw].values); m,b=np.polyfit(pdf[rw].values,pdf[sc].values,1)
            ax.plot(xs,m*xs+b,"k--",lw=1.3,alpha=.7)
        ax.text(.05,.93,f"r = {rv:+.3f}  [{lo:.3f},{hi:.3f}]",
                transform=ax.transAxes,fontsize=9,bbox=dict(boxstyle="round,pad=0.3",fc="white",alpha=.85))
        ax.set_xlabel("Normalised reward"); ax.set_ylabel("ARCUS stability score")
        ax.set_title(title,fontweight="bold"); ax.legend(title="Suite",markerscale=1.3,fontsize=8)
    fig.suptitle("ARCUS stability score vs normalised reward",fontweight="bold",fontsize=12)
    fig.tight_layout(); _save(fig,plots_dir/"fig04_correlation_scatter.png","Fig04")

def fig05_rank_disagreement(df,plots_dir):
    sc="leaderboard_score"; rw="reward_norm"
    if sc not in df.columns or rw not in df.columns: return
    sub=_det_env(df).dropna(subset=[sc,rw]).copy()
    if len(sub)<10: return
    sub["ra"]=sub[sc].rank(ascending=False,method="min")
    sub["rr"]=sub[rw].rank(ascending=False,method="min")
    sub["rd"]=(sub["ra"]-sub["rr"]).abs()
    mu=sub["rd"].mean(); mx=sub["rd"].max(); p10=(sub["rd"]>10).mean()*100
    print(f"  [STAT] Rank disagreement: mean={mu:.1f}  max={mx:.0f}  pct>10={p10:.0f}%")
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    axes[0].hist(sub["rd"].dropna(),bins=30,color="#42A5F5",ec="white",lw=.4)
    axes[0].axvline(mu,color="#EF5350",ls="--",lw=1.5,label=f"Mean={mu:.1f}")
    axes[0].axvline(10,color="#FF7043",ls=":",lw=1.5,alpha=.8,label=f"10-pos threshold ({p10:.0f}% exceed)")
    axes[0].set_xlabel("|rank$_{ARCUS}$ - rank$_{reward}$|"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Rank disagreement  (n={len(sub)}, {p10:.0f}% shift >10 positions)",fontweight="bold")
    axes[0].legend()
    sc2=axes[1].scatter(sub[rw],sub[sc],c=sub["rd"],cmap="RdYlBu_r",alpha=.65,s=28,vmin=0,vmax=200)
    plt.colorbar(sc2,ax=axes[1],label="|rank_ARCUS - rank_reward|")
    axes[1].set_xlabel("Normalised reward"); axes[1].set_ylabel("ARCUS stability score")
    axes[1].set_title("Rank disagreement colored scatter",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig05_rank_disagreement.png","Fig05")

def fig06_fragile_robust(df,plots_dir):
    sc="leaderboard_score"; rw="reward_norm"
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr or sc not in df.columns: return
    sub=_det_env(df).dropna(subset=[cr,rw]).copy()
    if len(sub)<10: return
    q75r=sub[rw].quantile(.75); q75c=sub[cr].quantile(.75)
    q50r=sub[rw].quantile(.50); q25c=sub[cr].quantile(.25)
    frag=sub[(sub[rw]>=q75r)&(sub[cr]>=q75c)]; rob=sub[(sub[rw]<=q50r)&(sub[cr]<=q25c)]
    print(f"  [STAT] Fragile={len(frag)}  Robust={len(rob)}")
    fig,ax=plt.subplots(figsize=(7.5,5.5))
    ax.scatter(sub[rw],sub[cr],alpha=.25,s=18,c="#bdbdbd",label="All",zorder=1)
    if not frag.empty: ax.scatter(frag[rw],frag[cr],c="#EF5350",s=55,zorder=3,label=f"Fragile n={len(frag)}")
    if not rob.empty:  ax.scatter(rob[rw], rob[cr], c="#1976D2",s=55,zorder=3,label=f"Robust n={len(rob)}")
    ax.axvline(q75r,color="#EF5350",ls=":",lw=.8,alpha=.6)
    ax.axhline(q75c,color="#EF5350",ls=":",lw=.8,alpha=.6)
    ax.set_xlabel("Normalised reward"); ax.set_ylabel("Policy degeneracy rate")
    ax.set_title("Fragile vs robust agents\nRed: high reward + high degeneracy  Blue: lower reward + stable",fontweight="bold")
    ax.legend(fontsize=8); fig.tight_layout(); _save(fig,plots_dir/"fig06_fragile_robust.png","Fig06")

def fig07_sac_td3_on(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    on=df[(df["schedule"]=="observation_noise")&(df["eval_mode"]=="deterministic")].copy()
    sub=on[on["algo"].isin(["sac","td3"])].copy()
    if sub.empty: return
    envs=sorted(sub["env"].unique()); x=np.arange(len(envs)); w=.35
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for i,algo in enumerate(["sac","td3"]):
        means=[sub[(sub["env"]==e)&(sub["algo"]==algo)][cr].mean() for e in envs]
        sems =[sub[(sub["env"]==e)&(sub["algo"]==algo)][cr].sem()  for e in envs]
        c="#EF5350" if algo=="sac" else "#42A5F5"
        axes[0].bar(x+(i-.5)*w,means,w,yerr=sems,capsize=4,color=c,alpha=.85,
                    label=algo.upper(),error_kw={"lw":1.2})
    axes[0].set_xticks(x); axes[0].set_xticklabels([e.split("-")[0] for e in envs],rotation=25,ha="right")
    axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("SAC vs TD3 under Observation Noise",fontweight="bold"); axes[0].legend()
    data=[on[on["algo"]=="sac"][cr].dropna().values, on[on["algo"]=="td3"][cr].dropna().values]
    mu2=[float(np.nanmean(d)) for d in data]
    bp=axes[1].boxplot(data,patch_artist=True,notch=True,medianprops={"color":"black","lw":2})
    for patch,c in zip(bp["boxes"],["#EF5350","#42A5F5"]): patch.set_facecolor(c); patch.set_alpha(.75)
    axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["SAC","TD3"])
    axes[1].set_ylabel("Policy degeneracy rate")
    axes[1].set_title(f"SAC {mu2[0]:.1%} vs TD3 {mu2[1]:.1%}",fontweight="bold")
    fig.suptitle("SAC entropy amplifies sensor fragility | TD3 target smoothing provides robustness",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig07_sac_td3_on.png","Fig07")

def fig08_channel_heatmap(df,plots_dir):
    avail={c:f"{c}_drop" for c in CH_KEYS if f"{c}_drop" in df.columns}
    if not avail: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ALL_S)
    if not subs: return
    rows=[]
    for s in subs:
        sub=det[det["schedule"]==s]
        row={c:float(sub[col].mean()) for c,col in avail.items()}; row["schedule"]=s; rows.append(row)
    tab=pd.DataFrame(rows).set_index("schedule"); ch=list(avail.keys())
    arr=tab[ch].to_numpy(float)
    fig,ax=plt.subplots(figsize=(len(ch)*2+2,len(subs)*.75+2))
    im=ax.imshow(arr,cmap="RdYlGn_r",aspect="auto",vmin=0,vmax=.5)
    ax.set_xticks(range(len(ch))); ax.set_xticklabels([CH_LABEL[c] for c in ch],rotation=20,ha="right")
    ax.set_yticks(range(len(subs))); ax.set_yticklabels([_sl(s) for s in subs])
    for i in range(len(subs)):
        for j in range(len(ch)):
            v=arr[i,j]
            if np.isfinite(v):
                ax.text(j,i,f"{v:.3f}",ha="center",va="center",fontsize=9,
                        color="white" if v>.3 else "black")
    plt.colorbar(im,ax=ax,label="Mean channel degradation  pre - shock")
    ax.set_title("Channel x stressor degradation (RL-native names)",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig08_channel_heatmap.png","Fig08")

def fig09_radar_channels(df, plots_dir):
    avail = {c: f"{c}_drop" for c in CH_KEYS if f"{c}_drop" in df.columns}
    if not avail: return
    det = df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs = _present(det, ALL_S)
    if not subs: return

    raw = {s: {c: float(det[det["schedule"]==s][col].mean())
               for c,col in avail.items()} for s in subs}

    chs = list(avail.keys())
    ch_max = {c: max(max(0.0, raw[s].get(c,0.0)) for s in subs) or 1.0 for c in chs}
    norm = {s: [max(0.0, raw[s].get(c,0.0))/ch_max[c] for c in chs] for s in subs}

    N = len(chs)
    angles = [2*math.pi*i/N for i in range(N)] + [0]

    STYLE = {
        "resource_constraint": ("-",  "#EF5350", 2.0),
        "trust_violation":     ("--", "#AB47BC", 2.0),
        "concept_drift":       ("-",  "#42A5F5", 2.0),
        "observation_noise":   ("-",  "#26A69A", 2.0),
        "sensor_blackout":     ("-",  "#5C6BC0", 2.0),
        "valence_inversion":   (":",  "#FF7043", 2.0),
        "reward_noise":        (":",  "#FFCA28", 2.0),
    }

    fig, ax = plt.subplots(figsize=(9,9), subplot_kw={"polar": True})
    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([CH_LABEL[c] for c in chs], fontsize=10)
    ax.set_yticks([0.25,0.50,0.75,1.00])
    ax.set_yticklabels(["0.25x","0.50x","0.75x","1.0x"], fontsize=7, color="gray")
    ax.set_ylim(0,1.1); ax.set_facecolor("#fafafa")

    for s in subs:
        vals = norm[s] + [norm[s][0]]
        ls,color,lw = STYLE.get(s, ("-","#999",1.5))
        ax.plot(angles, vals, ls=ls, color=color, lw=lw, label=_sl(s))
        ax.fill(angles, vals, alpha=0.05, color=color)

    neg_notes=[f"{CH_LABEL[c]}: improves under {', '.join(_sl(s) for s in subs if raw[s].get(c,0)<0)}"
               for c in chs if any(raw[s].get(c,0)<0 for s in subs)]
    ax.set_title(
        "Channel signatures by stressor (per-channel normalised degradation)\n"
        "Each axis = own max drop  |  clipped at 0  |  Solid=Perception  Dash=Execution  Dot=Feedback",
        fontsize=10,fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.35,1.15),fontsize=9)
    if neg_notes:
        fig.text(0.01,0.01,
                 "Channels shown as 0 (stressor improved them):\n"+"\n".join(f"  - {n}" for n in neg_notes),
                 fontsize=7.5,color="#555",va="bottom",ha="left",
                 bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#ddd",alpha=0.9))
    fig.tight_layout(); _save(fig,plots_dir/"fig09_radar_channels.png","Fig09: Radar [FIXED]")

def fig10_algo_family(df,plots_dir):
    sc="leaderboard_score"; cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if sc not in df.columns or not cr: return
    sub=_det_env(df).copy()
    if sub.empty: return
    fams=[f for f in ["On-Policy","Off-Policy AC","Value-Based"] if f in sub["algo_family"].unique()]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    data=[sub[sub["algo_family"]==f][sc].dropna().values for f in fams]
    parts=axes[0].violinplot(data,positions=range(len(fams)),widths=.65,showmeans=True,showmedians=True)
    for body,f in zip(parts["bodies"],fams):
        body.set_facecolor(FAM_COLOR.get(f,"#999")); body.set_alpha(.75)
    axes[0].set_xticks(range(len(fams))); axes[0].set_xticklabels(fams,rotation=15,ha="right")
    axes[0].set_ylabel("ARCUS stability score")
    axes[0].set_title("ARCUS score distribution by algorithm family",fontweight="bold")
    algos=sorted(sub["algo"].unique())
    means=[sub[sub["algo"]==a][cr].mean() for a in algos]
    colors=[FAM_COLOR.get(ALGO_FAMILY.get(a,"Other"),"#999") for a in algos]
    idx=np.argsort(means)[::-1]
    axes[1].bar(range(len(algos)),[means[i] for i in idx],color=[colors[i] for i in idx],alpha=.85)
    axes[1].set_xticks(range(len(algos))); axes[1].set_xticklabels([algos[i].upper() for i in idx],rotation=20,ha="right")
    axes[1].set_ylim(0,1.05); axes[1].set_ylabel("Mean degeneracy rate")
    axes[1].set_title("Algorithm robustness ranking (lower = more robust)",fontweight="bold")
    for f,c in FAM_COLOR.items(): axes[1].bar([],[],color=c,alpha=.85,label=f)
    axes[1].legend(fontsize=8); fig.tight_layout(); _save(fig,plots_dir/"fig10_algo_family.png","Fig10")

def fig11_cvar_vs_arcus(df,plots_dir):
    cvar=_col("cvar_shock_05","cvar_shock_25",df=df)
    cr  =_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    sc  ="leaderboard_score"
    if not cvar or not cr or sc not in df.columns: return
    sub=df[(df["eval_mode"]=="deterministic")&df["schedule"].isin(ENV_S)].copy()
    sub=sub.dropna(subset=[cvar,cr,sc])
    if len(sub)<20: return
    sub["cvar_norm"]=sub.groupby("env")[cvar].transform(
        lambda x:(x-x.min())/(x.max()-x.min()+1e-8))
    r1,p1,lo1,hi1=bootstrap_r(sub["cvar_norm"].values, sub[cr].values)
    r2,p2,lo2,hi2=bootstrap_r(sub["cvar_norm"].values, sub[sc].values)

    sub["cvar_high"]  = sub["cvar_norm"] > 0.5
    sub["cr_low"]     = sub[cr] < 0.20
    disagree = sub[sub["cvar_high"] & sub["cr_low"]]

    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for suite,grp in sub.groupby("suite"):
        axes[0].scatter(grp["cvar_norm"],grp[cr],
                        c=SUITE_COLOR.get(suite,"#999"),alpha=.5,s=22,label=suite)
    axes[0].text(.05,.93,
                 f"r = {r1:+.3f}  [{lo1:.3f},{hi1:.3f}]\np = {p1:.2e}",
                 transform=axes[0].transAxes,fontsize=9,
                 bbox=dict(boxstyle="round",fc="white",alpha=.85))
    if not disagree.empty:
        axes[0].scatter(disagree["cvar_norm"],disagree[cr],
                        c="gold",s=60,zorder=5,marker="*",
                        label=f"High CVaR but stable (n={len(disagree)})")
    xs=np.sort(sub["cvar_norm"].values); m,b=np.polyfit(sub["cvar_norm"].values,sub[cr].values,1)
    axes[0].plot(xs,m*xs+b,"k--",lw=1.2,alpha=.6)
    axes[0].set_xlabel("Normalised CVaR-5 (shock phase)"); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("CVaR-5 vs policy degeneracy\nGold stars: CVaR high but ARCUS stable",fontweight="bold")
    axes[0].legend(fontsize=8,markerscale=1.3)

    for suite,grp in sub.groupby("suite"):
        axes[1].scatter(grp["cvar_norm"],grp[sc],
                        c=SUITE_COLOR.get(suite,"#999"),alpha=.5,s=22,label=suite)
    axes[1].text(.05,.93,
                 f"r = {r2:+.3f}  [{lo2:.3f},{hi2:.3f}]\np = {p2:.2e}",
                 transform=axes[1].transAxes,fontsize=9,
                 bbox=dict(boxstyle="round",fc="white",alpha=.85))
    xs2=np.sort(sub["cvar_norm"].values); m2,b2=np.polyfit(sub["cvar_norm"].values,sub[sc].values,1)
    axes[1].plot(xs2,m2*xs2+b2,"k--",lw=1.2,alpha=.6)
    axes[1].set_xlabel("Normalised CVaR-5"); axes[1].set_ylabel("ARCUS stability score")
    axes[1].set_title("CVaR-5 vs ARCUS stability score\nCorrelated but measuring different dimensions",fontweight="bold")
    axes[1].legend(fontsize=8,markerscale=1.3)
    fig.suptitle(
        f"CVaR-5 and ARCUS are positively correlated (r={r2:+.3f}) but not interchangeable\n"
        f"{len(disagree)} policies show high CVaR yet low ARCUS degeneracy — CVaR cannot substitute for behavioral stability",
        fontweight="bold",fontsize=11)
    fig.tight_layout(); _save(fig,plots_dir/"fig11_cvar_vs_arcus.png","Fig11: CVaR vs ARCUS")

def fig12_channel_mad(df,plots_dir):
    mad_cols=[c for c in df.columns if c.startswith("mad_")]
    if not mad_cols: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    det_b=det[det["schedule"]=="baseline"]; det_e=det[det["schedule"].isin(ENV_S)]
    if det_b.empty or det_e.empty: return
    ch_names=[c.replace("mad_","") for c in mad_cols]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    envs=sorted(det_b["env"].unique()); x=np.arange(len(envs)); w=.8/max(len(ch_names),1)
    for i,ch in enumerate(ch_names):
        col=f"mad_{ch}"
        means=[det_b[det_b["env"]==e][col].mean() for e in envs]
        offset=(i-(len(ch_names)-1)/2)*w
        axes[0].bar(x+offset,means,w,label=CH_LABEL.get(ch,ch),color=CH_COLOR.get(ch,"#999"),alpha=.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([e.split("-")[0][:10] for e in envs],rotation=30,ha="right",fontsize=8)
    axes[0].set_ylabel("Pre-phase MAD"); axes[0].set_title("Per-channel MAD by environment",fontweight="bold")
    axes[0].legend(fontsize=8)
    cb=[det_b[f"mad_{c}"].mean() for c in ch_names]; ce=[det_e[f"mad_{c}"].mean() for c in ch_names]
    xc=np.arange(len(ch_names)); wc=.35
    axes[1].bar(xc-wc/2,cb,wc,label="Baseline (pre)",color="#42A5F5",alpha=.85)
    axes[1].bar(xc+wc/2,ce,wc,label="Env stressors (shock)",color="#EF5350",alpha=.85)
    axes[1].set_xticks(xc); axes[1].set_xticklabels([CH_LABEL.get(c,c) for c in ch_names],rotation=20,ha="right")
    axes[1].set_ylabel("Mean MAD"); axes[1].set_title("Channel MAD: baseline vs shock",fontweight="bold"); axes[1].legend()
    fig.tight_layout(); _save(fig,plots_dir/"fig12_channel_mad.png","Fig12")

def fig13_walker2d(df,plots_dir):
    walker=[e for e in df["env"].unique() if "walker" in e.lower()]
    if not walker: return
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ALL_S); wdf=det[(det["env"].isin(walker))&det["schedule"].isin(subs)]
    if wdf.empty: return
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    algos=sorted(wdf["algo"].unique()); x=np.arange(len(subs)); w=.8/max(len(algos),1)
    for i,algo in enumerate(algos):
        means=[wdf[(wdf["algo"]==algo)&(wdf["schedule"]==s)][cr].mean() for s in subs]
        offset=(i-(len(algos)-1)/2)*w
        axes[0].bar(x+offset,means,w,label=algo.upper(),alpha=.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([_sl(s) for s in subs],rotation=30,ha="right")
    axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("Walker2d-v4 — FPR=0.053 (converged)",fontweight="bold"); axes[0].legend()
    mj=[e for e in df["env"].unique() if any(k in e for k in ["Walker","HalfCheetah","Hopper"])]
    em=(det[(det["env"].isin(mj))&det["schedule"].isin(ENV_S)].groupby("env")[cr].mean().sort_values(ascending=False))
    axes[1].barh(em.index.tolist(),em.values,color=SUITE_COLOR["MuJoCo"],alpha=.85)
    axes[1].set_xlabel("Mean degeneracy rate (env stressors)"); axes[1].set_title("MuJoCo suite comparison",fontweight="bold")
    fig.suptitle("Walker2d-v4 — New MuJoCo Environment",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig13_walker2d.png","Fig13")

def fig14_atari_comparison(df,plots_dir):
    atari=df[df["env"].str.contains("ALE",na=False)].copy()
    if atari.empty or atari["env"].nunique()<2: return
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    det=atari[atari["eval_mode"]=="deterministic"] if "eval_mode" in atari.columns else atari
    subs=_present(det,ALL_S); aenvs=sorted(det["env"].unique())
    x=np.arange(len(subs)); w=.8/max(len(aenvs),1)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5)); colors=["#EF5350","#42A5F5"]
    for i,env in enumerate(aenvs):
        means=[det[(det["env"]==env)&(det["schedule"]==s)][cr].mean() for s in subs]
        offset=(i-(len(aenvs)-1)/2)*w
        axes[0].bar(x+offset,means,w,label=env.split("/")[-1],alpha=.85,color=colors[i%2])
    axes[0].set_xticks(x); axes[0].set_xticklabels([_sl(s) for s in subs],rotation=30,ha="right")
    axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("Pong vs SpaceInvaders — same CNN wrapper\nON: 42% vs 13%",fontweight="bold"); axes[0].legend()
    ach=[c for c in CH_KEYS if f"{c}_drop" in det.columns]
    if ach:
        od=det[det["schedule"]=="observation_noise"]
        me={env:[float(od[od["env"]==env][f"{c}_drop"].mean()) for c in ach] for env in aenvs}
        xx=np.arange(len(ach)); ww=.8/max(len(aenvs),1)
        for i,env in enumerate(aenvs):
            offset=(i-(len(aenvs)-1)/2)*ww
            axes[1].bar(xx+offset,me[env],ww,label=env.split("/")[-1],alpha=.85,color=colors[i%2])
        axes[1].set_xticks(xx); axes[1].set_xticklabels([CH_LABEL[c] for c in ach],rotation=20,ha="right")
        axes[1].set_ylabel("Channel drop (pre - shock)"); axes[1].set_title("Channel degradation under ON",fontweight="bold")
        axes[1].legend()
    fig.suptitle("Atari: CNN robustness is representation-dependent, not architecture-determined",fontweight="bold",fontsize=11)
    fig.tight_layout(); _save(fig,plots_dir/"fig14_atari_comparison.png","Fig14")

def fig15_fpr_validation(df,plots_dir):
    cr_pre=_col("collapse_rate_pre","collapse_rate_pre__mean","fpr_actual",df=df)
    cr_shock=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr_pre or not cr_shock: return
    base=df[df["schedule"]=="baseline"]; env_s=df[df["schedule"].isin(ENV_S)]
    if base.empty: return
    pre_v=base[cr_pre].dropna().values; sh_v=env_s[cr_shock].dropna().values if not env_s.empty else np.array([])
    mfpr=float(np.mean(pre_v))
    print(f"  [STAT] FPR: mean={mfpr:.4f}  target=0.05")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].hist(pre_v,bins=25,color="#42A5F5",ec="white",lw=.4)
    axes[0].axvline(.05,color="#EF5350",ls="--",lw=1.5,label="Target a=0.05")
    axes[0].axvline(mfpr,color="#FF7043",ls=":",lw=1.5,label=f"Observed={mfpr:.3f}")
    axes[0].set_xlabel("Pre-phase degeneracy rate (FPR)"); axes[0].set_ylabel("Count")
    axes[0].set_title(f"Empirical FPR  mean={mfpr:.3f}  target=0.05",fontweight="bold"); axes[0].legend()
    if sh_v.size>0: axes[1].hist(sh_v,bins=25,color="#EF5350",ec="white",lw=.4,alpha=.7,label="Env-stressor shock")
    axes[1].hist(pre_v,bins=25,color="#42A5F5",ec="white",lw=.4,alpha=.7,label="Baseline FPR")
    axes[1].set_xlabel("Degeneracy rate"); axes[1].set_ylabel("Count")
    axes[1].set_title("FPR vs shock rates — separation confirms discriminative power",fontweight="bold"); axes[1].legend()
    fig.tight_layout(); _save(fig,plots_dir/"fig15_fpr_validation.png","Fig15")

def fig16_stoch_vs_det(df,plots_dir):
    sc="leaderboard_score"; cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if sc not in df.columns or not cr: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    sto=df[df["eval_mode"]=="stochastic"]    if "eval_mode" in df.columns else pd.DataFrame()
    if sto.empty: return
    d2=det[det["schedule"].isin(ENV_S)].copy(); s2=sto[sto["schedule"].isin(ENV_S)].copy()
    mg=d2.merge(s2,on=["env","algo","schedule"],suffixes=("_det","_sto"),how="inner")
    if mg.empty: return
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    axes[0].scatter(mg[f"{sc}_det"],mg[f"{sc}_sto"],
                    c=[SUITE_COLOR.get(_suite(e),"#999") for e in mg["env"]],alpha=.55,s=22)
    mn=min(mg[f"{sc}_det"].min(),mg[f"{sc}_sto"].min()); mx=max(mg[f"{sc}_det"].max(),mg[f"{sc}_sto"].max())
    axes[0].plot([mn,mx],[mn,mx],"k--",lw=1.2)
    rv,_,lo,hi=bootstrap_r(mg[f"{sc}_det"].values,mg[f"{sc}_sto"].values)
    axes[0].text(.05,.93,f"r={rv:+.3f}  [{lo:.3f},{hi:.3f}]",transform=axes[0].transAxes,fontsize=9,
                 bbox=dict(boxstyle="round",fc="white",alpha=.85))
    axes[0].set_xlabel("ARCUS score (det)"); axes[0].set_ylabel("ARCUS score (stoch)")
    axes[0].set_title("Det vs stochastic eval agreement",fontweight="bold")
    axes[1].scatter(mg[f"{cr}_det"],mg[f"{cr}_sto"],
                    c=[SUITE_COLOR.get(_suite(e),"#999") for e in mg["env"]],alpha=.55,s=22)
    mn2=min(mg[f"{cr}_det"].min(),mg[f"{cr}_sto"].min()); mx2=max(mg[f"{cr}_det"].max(),mg[f"{cr}_sto"].max())
    axes[1].plot([mn2,mx2],[mn2,mx2],"k--",lw=1.2)
    axes[1].set_xlabel("Collapse rate (det)"); axes[1].set_ylabel("Collapse rate (stoch)")
    axes[1].set_title("Collapse rate: det vs stochastic",fontweight="bold")
    for suite,c in SUITE_COLOR.items(): axes[1].scatter([],[],c=c,label=suite,s=22)
    axes[1].legend(title="Suite",fontsize=8)
    fig.tight_layout(); _save(fig,plots_dir/"fig16_stoch_vs_det.png","Fig16")

def fig17_algo_ranking(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df); sc="leaderboard_score"
    if not cr or sc not in df.columns: return
    sub=_det_env(df).copy(); algos=sorted(sub["algo"].unique())
    cr_m=[sub[sub["algo"]==a][cr].mean() for a in algos]
    sc_m=[sub[sub["algo"]==a][sc].mean() for a in algos]
    cr_e=[sub[sub["algo"]==a][cr].sem()  for a in algos]
    idx=np.argsort(cr_m)
    colors=[FAM_COLOR.get(ALGO_FAMILY.get(algos[i],"Other"),"#999") for i in idx]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    axes[0].barh(range(len(algos)),[cr_m[i] for i in idx],xerr=[cr_e[i] for i in idx],
                 color=colors,alpha=.85,capsize=4,error_kw={"lw":1.2})
    axes[0].set_yticks(range(len(algos))); axes[0].set_yticklabels([algos[i].upper() for i in idx])
    axes[0].set_xlabel("Mean degeneracy rate"); axes[0].set_title("Algorithm robustness ranking (lower=more robust)",fontweight="bold")
    for f,c in FAM_COLOR.items(): axes[0].barh([],[],color=c,label=f)
    axes[0].legend(fontsize=8)
    axes[1].barh(range(len(algos)),[sc_m[i] for i in idx],color=colors,alpha=.85)
    axes[1].set_yticks(range(len(algos))); axes[1].set_yticklabels([algos[i].upper() for i in idx])
    axes[1].set_xlabel("Mean ARCUS leaderboard score"); axes[1].set_title("Algorithm stability score ranking",fontweight="bold")
    for f,c in FAM_COLOR.items(): axes[1].barh([],[],color=c,label=f)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); _save(fig,plots_dir/"fig17_algo_ranking.png","Fig17")

def fig18_reward_drop(df,plots_dir):
    rw=_col("reward_mean","reward_mean__mean",df=df)
    if not rw: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ENV_S); envs=sorted(det["env"].unique())
    if not subs: return
    base=det[det["schedule"]=="baseline"].groupby("env")[rw].mean()
    rows=[]
    for s in subs:
        sub=det[det["schedule"]==s]; row={"schedule":_sl(s)}
        for e in envs:
            b=base.get(e,np.nan); v=sub[sub["env"]==e][rw].mean()
            row[e]=(v-b)/max(abs(b),1e-6) if np.isfinite(b) and np.isfinite(v) and abs(b)>1e-6 else np.nan
        rows.append(row)
    tab=pd.DataFrame(rows).set_index("schedule")[envs]; arr=tab.to_numpy(float)
    fig,ax=plt.subplots(figsize=(max(10,len(envs)*1.3),max(4,len(subs)*.7+2)))
    im=ax.imshow(arr,cmap="RdYlGn",aspect="auto",vmin=-1.5,vmax=0.1)
    ax.set_xticks(range(len(envs))); ax.set_xticklabels([e.split("-")[0][:10] for e in envs],rotation=30,ha="right",fontsize=8)
    ax.set_yticks(range(len(subs))); ax.set_yticklabels(tab.index)
    for i in range(len(subs)):
        for j in range(len(envs)):
            v=arr[i,j]
            if np.isfinite(v): ax.text(j,i,f"{v:+.2f}",ha="center",va="center",fontsize=7.5,
                                       color="white" if v<-.8 else "black")
    plt.colorbar(im,ax=ax,label="Relative reward drop vs baseline")
    ax.set_title("Reward drop under env stressors (normalised by baseline reward)",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig18_reward_drop.png","Fig18")

def fig19_score_density(df,plots_dir):
    sc="leaderboard_score"
    if sc not in df.columns: return
    sub=_det_env(df).dropna(subset=[sc]).copy()
    if sub.empty: return
    suites=[s for s in ["Classic","Continuous","MuJoCo","Atari"] if s in sub["suite"].unique()]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for suite in suites:
        vals=sub[sub["suite"]==suite][sc].dropna().values
        if len(vals)<5: continue
        try:
            kde=gaussian_kde(vals,bw_method="scott"); xs=np.linspace(vals.min()-.05,vals.max()+.05,300)
            axes[0].plot(xs,kde(xs),color=SUITE_COLOR.get(suite,"#999"),lw=2.2,label=f"{suite}  u={vals.mean():.3f}")
            axes[0].fill_between(xs,kde(xs),alpha=.12,color=SUITE_COLOR.get(suite,"#999"))
        except: pass
    axes[0].set_xlabel("ARCUS stability score"); axes[0].set_ylabel("Density")
    axes[0].set_title("Stability score distribution by suite",fontweight="bold"); axes[0].legend(fontsize=9)
    parts=axes[1].violinplot([sub[sub["suite"]==s][sc].dropna().values for s in suites],
                             positions=range(len(suites)),widths=.7,showmeans=True,showmedians=True)
    for body,suite in zip(parts["bodies"],suites):
        body.set_facecolor(SUITE_COLOR.get(suite,"#999")); body.set_alpha(.75)
    axes[1].set_xticks(range(len(suites))); axes[1].set_xticklabels(suites)
    axes[1].set_ylabel("ARCUS stability score"); axes[1].set_title("Score violin by suite",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig19_score_density.png","Fig19: Score density [NEEDED IN PAPER]")

def fig20_channel_drop_density(df,plots_dir):
    avail=[c for c in CH_KEYS if f"{c}_drop" in df.columns]
    if not avail: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ENV_S)
    if not subs: return
    sub=det[det["schedule"].isin(subs)]
    fig,axes=plt.subplots(1,len(avail),figsize=(len(avail)*3.2+1,5),sharey=False)
    if len(avail)==1: axes=[axes]
    for ax,ch in zip(axes,avail):
        col=f"{ch}_drop"
        for sched in subs:
            vals=sub[sub["schedule"]==sched][col].dropna().values
            if len(vals)<5: continue
            try:
                kde=gaussian_kde(vals,bw_method="scott")
                xs=np.linspace(max(-.05,vals.min()-.02),min(1.05,vals.max()+.02),200)
                ax.plot(xs,kde(xs),color=_sc(sched),lw=1.6,alpha=.85,label=_sl(sched))
                ax.fill_between(xs,kde(xs),alpha=.07,color=_sc(sched))
            except: pass
        ax.set_xlabel("Drop (pre - shock)"); ax.set_title(CH_LABEL[ch],fontweight="bold",fontsize=9,color=CH_COLOR.get(ch,"black"))
        ax.set_xlim(-.05,1.05)
        if ch==avail[0]: ax.set_ylabel("Density"); ax.legend(fontsize=7)
    fig.suptitle("Per-channel behavioral degradation density by stressor\n(env stressors, KDE, det eval)",fontweight="bold")
    fig.tight_layout(); _save(fig,plots_dir/"fig20_channel_drop_density.png","Fig20: Channel density [NEEDED IN PAPER]")

def fig21_cvar_density(df,plots_dir):
    cvar=_col("cvar_shock_05","cvar_shock_25",df=df); cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cvar or not cr: return
    sub=df[(df["eval_mode"]=="deterministic")&df["schedule"].isin(ENV_S)].copy()
    sub=sub.dropna(subset=[cvar,cr])
    if len(sub)<20: return
    sub["cvar_norm"]=sub.groupby("env")[cvar].transform(lambda x:(x-x.min())/(x.max()-x.min()+1e-8))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    for suite,grp in sub.groupby("suite"):
        axes[0].scatter(grp["cvar_norm"],grp[cr],c=SUITE_COLOR.get(suite,"#999"),alpha=.45,s=18,label=suite)
    try:
        x=sub["cvar_norm"].values; y=sub[cr].values; mask=np.isfinite(x)&np.isfinite(y); x,y=x[mask],y[mask]
        if len(x)>30:
            xi=np.linspace(0,1,80); yi=np.linspace(0,1,80); Xi,Yi=np.meshgrid(xi,yi)
            kde=gaussian_kde(np.vstack([x[:5000],y[:5000]]))
            Zi=kde(np.vstack([Xi.flatten(),Yi.flatten()])).reshape(Xi.shape)
            axes[0].contour(Xi,Yi,Zi,levels=5,linewidths=.8,colors="navy",alpha=.4)
    except: pass
    rv,_,lo,hi=bootstrap_r(sub["cvar_norm"].values,sub[cr].values)
    axes[0].text(.05,.93,f"r={rv:+.3f}  [{lo:.3f},{hi:.3f}]",transform=axes[0].transAxes,
                 fontsize=9,bbox=dict(boxstyle="round",fc="white",alpha=.85))
    axes[0].set_xlabel("Normalised CVaR-5"); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("CVaR-5 vs degeneracy with KDE contours",fontweight="bold")
    axes[0].legend(title="Suite",fontsize=8,markerscale=1.3)
    for suite in ["Classic","Continuous","MuJoCo","Atari"]:
        vals=sub[sub["suite"]==suite]["cvar_norm"].dropna().values
        if len(vals)<5: continue
        try:
            kde=gaussian_kde(vals,bw_method="scott"); xs=np.linspace(0,1,200)
            axes[1].plot(xs,kde(xs),color=SUITE_COLOR.get(suite,"#999"),lw=2.2,label=f"{suite}  u={vals.mean():.3f}")
            axes[1].fill_between(xs,kde(xs),alpha=.1,color=SUITE_COLOR.get(suite,"#999"))
        except: pass
    axes[1].set_xlabel("Normalised CVaR-5"); axes[1].set_ylabel("Density")
    axes[1].set_title("CVaR-5 distribution by suite",fontweight="bold"); axes[1].legend(fontsize=9)
    fig.tight_layout(); _save(fig,plots_dir/"fig21_cvar_density.png","Fig21")

def fig22_seed_variance(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    subs=_present(df,ENV_S)
    if not subs: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    data=[det[det["schedule"]==s][cr].dropna().values for s in subs]
    fig,ax=plt.subplots(figsize=(max(9,len(subs)*1.4),5))
    bp=ax.boxplot(data,patch_artist=True,notch=False,medianprops={"color":"black","lw":2})
    for patch,s in zip(bp["boxes"],subs): patch.set_facecolor(_sc(s)); patch.set_alpha(.75)
    ax.set_xticks(range(1,len(subs)+1)); ax.set_xticklabels([_sl(s) for s in subs])
    ax.set_ylabel("Policy degeneracy rate"); ax.axhline(.05,color="gray",ls="--",lw=.9,label="FPR 5%")
    ax.set_title("Seed-to-seed variance by stressor (10 seeds x all envs x all algos)",fontweight="bold")
    ax.legend(); fig.tight_layout(); _save(fig,plots_dir/"fig22_seed_variance.png","Fig22")

def fig23_per_episode_channels(df_pe, plots_dir):
    if df_pe is None or df_pe.empty: return
    if "stress_phase" not in df_pe.columns or "identity" not in df_pe.columns: return

    pe = df_pe.copy()
    subs = [s for s in ["concept_drift","resource_constraint","trust_violation","valence_inversion"]
            if s in pe["schedule"].unique()]
    if not subs: return

    channels = [c for c in ["competence","coherence","continuity","integrity","meaning","identity"]
                if c in pe.columns]
    PHASE_COLOR = {"pre":"#42A5F5","shock":"#EF5350","post":"#66BB6A"}

    fig,axes=plt.subplots(len(channels),len(subs),
                          figsize=(len(subs)*4,len(channels)*2.2+1),
                          sharex=False,sharey="row")
    if len(channels)==1: axes=[axes]
    if len(subs)==1: axes=[[ax] for ax in axes]

    for j,sched in enumerate(subs):
        sub=pe[pe["schedule"]==sched].sort_values("episode_idx") if "episode_idx" in pe.columns else pe[pe["schedule"]==sched]
        for i,ch in enumerate(channels):
            ax=axes[i][j]
            for phase,color in PHASE_COLOR.items():
                ps=sub[sub["stress_phase"]==phase]
                if ps.empty: continue
                ax.scatter(ps.get("episode_idx",ps.index),ps[ch],
                           c=color,alpha=.4,s=8,label=phase if j==0 and i==0 else "")
                try:
                    mn=ps.get("episode_idx",ps.index).rolling(5,min_periods=1).mean() if hasattr(ps.get("episode_idx",ps.index),"rolling") else ps.index
                    ax.plot(ps.get("episode_idx",ps.index),ps[ch].rolling(5,min_periods=1).mean(),
                            color=color,lw=1.5,alpha=.85)
                except: pass
            if i==0: ax.set_title(_sl(sched),fontweight="bold",color=_sc(sched))
            if j==0: ax.set_ylabel(CH_LABEL.get(ch,ch) if ch!="identity" else "ARCUS Score",fontsize=8)
            ax.set_ylim(0,1.1)
            if i<len(channels)-1: ax.set_xticks([])

    handles=[mpatches.Patch(color=c,label=p) for p,c in PHASE_COLOR.items()]
    fig.legend(handles=handles,loc="upper right",fontsize=9)
    fig.suptitle("Per-episode channel trajectories — HalfCheetah across stressors\n"
                 "Blue=Pre  Red=Shock  Green=Post",fontweight="bold",fontsize=11)
    fig.tight_layout(); _save(fig,plots_dir/"fig23_per_episode_channels.png","Fig23: Per-episode [NEEDED IN PAPER]")

def fig24_atari_robustness_density(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    atari=df[df["env"].str.contains("ALE",na=False)].copy()
    if "eval_mode" in atari.columns: atari=atari[atari["eval_mode"]=="deterministic"]
    if atari.empty or atari["env"].nunique()<2: return
    fig,ax=plt.subplots(figsize=(9,5.5))
    envs=sorted(atari["env"].unique()); colors=["#EF5350","#42A5F5"]
    for env,color in zip(envs,colors):
        for sched,ls in zip(["observation_noise","trust_violation","sensor_blackout"],["-","--",":"]):
            vals=atari[(atari["env"]==env)&(atari["schedule"]==sched)][cr].dropna().values
            if len(vals)<3: continue
            try:
                kde=gaussian_kde(vals,bw_method=.5); xs=np.linspace(0,1,300)
                label=f"{env.split('/')[-1].split('-')[0]} / {_sl(sched)}"
                ax.plot(xs,kde(xs),color=color,lw=2,ls=ls,alpha=.85,label=label)
                ax.fill_between(xs,kde(xs),alpha=.06,color=color)
            except: pass
    ax.set_xlabel("Policy degeneracy rate (shock)"); ax.set_ylabel("Density")
    ax.set_title("Atari robustness density: SpaceInvaders vs Pong\n"
                 "SpaceInvaders: dense near 0 under ON (13%)  |  Pong: broader (42%)",fontweight="bold")
    ax.legend(fontsize=7.5,ncol=2); ax.set_xlim(-.02,1.02)
    fig.tight_layout(); _save(fig,plots_dir/"fig24_atari_density.png","Fig24: Atari density [NEEDED IN PAPER]")

def fig25_mujoco_deepdive(df,plots_dir):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    if not cr: return
    mj=[e for e in df["env"].unique() if ENV_SUITE.get(e)=="MuJoCo"]
    if not mj: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ENV_S)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    x=np.arange(len(subs)); w=.8/max(len(mj),1)
    for i,env in enumerate(sorted(mj)):
        means=[det[(det["env"]==env)&(det["schedule"]==s)][cr].mean() for s in subs]
        offset=(i-(len(mj)-1)/2)*w
        axes[0].bar(x+offset,means,w,label=env.split("-")[0],alpha=.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels([_sl(s) for s in subs],rotation=20,ha="right")
    axes[0].set_ylim(0,1.05); axes[0].set_ylabel("Policy degeneracy rate")
    axes[0].set_title("MuJoCo per-stressor collapse",fontweight="bold"); axes[0].legend(title="Environment",fontsize=8)
    dm=det[(det["env"].isin(mj))&det["schedule"].isin(ENV_S)]
    pv=dm.groupby(["algo","env"])[cr].mean().unstack("env").reindex(columns=sorted(mj))
    if not pv.empty:
        im=axes[1].imshow(pv.to_numpy(float),cmap="RdYlGn_r",aspect="auto",vmin=0,vmax=1)
        axes[1].set_xticks(range(len(pv.columns))); axes[1].set_xticklabels([e.split("-")[0] for e in pv.columns],fontsize=9)
        axes[1].set_yticks(range(len(pv.index))); axes[1].set_yticklabels(pv.index)
        for i in range(len(pv.index)):
            for j in range(len(pv.columns)):
                v=pv.to_numpy(float)[i,j]
                if np.isfinite(v): axes[1].text(j,i,f"{v:.2f}",ha="center",va="center",
                                                fontsize=8.5,color="white" if v>.6 else "black")
        plt.colorbar(im,ax=axes[1],label="Mean degeneracy (env stressors)")
        axes[1].set_title("MuJoCo algo x env heatmap",fontweight="bold")
    fig.suptitle("MuJoCo Deep-Dive: Locomotion Fragility Across Three Environments",fontweight="bold",fontsize=11)
    fig.tight_layout(); _save(fig,plots_dir/"fig25_mujoco_deepdive.png","Fig25")


def tab_degeneracy(df,td):
    cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df); subs=_present(df,ALL_S)
    if not cr or not subs: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    piv=(det[det["schedule"].isin(subs)].groupby(["env","schedule"])[cr].mean()
         .unstack("schedule").reindex(columns=subs))
    ei=[i for i,s in enumerate(subs) if s in ENV_S]
    lines=[r"\begin{table*}[t]",r"\centering",r"\small",
           r"\caption{Policy degeneracy rate (shock phase) per environment and stressor. "
           r"Mean over algorithms, 10 seeds, deterministic eval. "
           r"\textbf{Bold} = worst env stressor per row. "
           r"VI/RN are feedback-axis stressors excluded from primary correlation.}",
           r"\label{tab:degeneracy}",r"\begin{tabular}{l"+"r"*len(subs)+"}",r"\toprule",
           "Environment & "+" & ".join([_sl(s) for s in subs])+r" \\",r"\midrule"]
    for env,row in piv.iterrows():
        vals=row.values.astype(float); fin=[i for i in ei if np.isfinite(vals[i])]
        worst=fin[int(np.argmax(vals[fin]))] if fin else -1
        cells=[]
        for i,v in enumerate(vals):
            if not np.isfinite(v): cells.append("--")
            elif i==worst: cells.append(r"\textbf{"+f"{v:.2f}"+"}")
            else: cells.append(f"{v:.2f}")
        lines.append(str(env).replace("_",r"\_")+" & "+" & ".join(cells)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    _wtex(td/"tab01_degeneracy.tex",lines)

def tab_corr(stats,td):
    rows=[("primary","Pearson (env stressors, VI/RN excluded)","Primary"),
          ("secondary","Pearson (all stressors incl VI/RN)","Secondary"),
          ("spearman","Spearman rank (env stressors)","Robustness"),
          ("z_normed","Pearson, per-env z-norm (env stressors)","Robustness"),
          ("non_atari","Pearson (env stressors, Atari excluded)","Stratified"),
          ("atari","Pearson (env stressors, Atari only)","Stratified")]
    lines=[r"\begin{table}[t]",r"\centering",r"\small",
           r"\caption{ARCUS stability score vs normalised reward: all correlation variants. "
           r"2000-resample bootstrap 95\% CIs. "
           r"R$^2=0.037$: 96.3\% of stability variance unexplained by reward alone. "
           r"\textbf{Bold} = primary claim.}",
           r"\label{tab:correlation}",r"\begin{tabular}{lllrrr}",r"\toprule",
           r"Key & Description & Type & $r$ & 95\% CI & $n$ \\",r"\midrule"]
    for key,desc,atype in rows:
        d=stats.get(key,{}); rv=d.get("r",np.nan); ci=d.get("ci",[np.nan,np.nan]); n=d.get("n","--")
        if not isinstance(rv,float) or not math.isfinite(rv): continue
        rs=(r"\textbf{"+f"{rv:+.3f}"+"}") if key=="primary" else f"{rv:+.3f}"
        cis=f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci and all(math.isfinite(c) for c in ci) else "N/A"
        lines.append(f"{key} & {desc} & {atype} & {rs} & {cis} & {n}"+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    _wtex(td/"tab02_correlation.tex",lines)

def tab_fpr(df,td):
    cr_pre=_col("collapse_rate_pre","collapse_rate_pre__mean","fpr_actual",df=df)
    if not cr_pre: return
    base=df[df["schedule"]=="baseline"]
    piv=base.groupby(["env","algo"])[cr_pre].mean().unstack("algo")
    algos=sorted(piv.columns); mf=float(base[cr_pre].mean())
    lines=[r"\begin{table}[t]",r"\centering",r"\small",
           rf"\caption{{Empirical FPR (pre-phase degeneracy, baseline). "
           rf"Target $\alpha=0.05$. Mean $=$ {mf:.3f}. "
           rf"\textbf{{Bold}} $>0.10$.}}",
           r"\label{tab:fpr}",r"\begin{tabular}{l"+"r"*len(algos)+"}",r"\toprule",
           "Environment & "+" & ".join([a.replace("_",r"\_") for a in algos])+r" \\",r"\midrule"]
    for env,row in piv.iterrows():
        cells=[]
        for a in algos:
            v=float(row.get(a,np.nan))
            if not math.isfinite(v): cells.append("--")
            elif v>.10: cells.append(r"\textbf{"+f"{v:.3f}"+"}")
            else: cells.append(f"{v:.3f}")
        lines.append(str(env).replace("_",r"\_").replace("/","/"
                     )+" & "+" & ".join(cells)+r" \\")
    lines+=[r"\midrule",
            r"\textbf{Mean} & \multicolumn{"+str(len(algos))+r"}{r}{"+f"{mf:.3f}"+r"} \\",
            r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    _wtex(td/"tab03_fpr.tex",lines)

def tab_channel(df,td):
    avail=[c for c in CH_KEYS if f"{c}_drop" in df.columns]
    if not avail: return
    det=df[df["eval_mode"]=="deterministic"] if "eval_mode" in df.columns else df
    subs=_present(det,ALL_S); rows=[]
    for s in subs:
        sub=det[det["schedule"]==s]
        row={c:float(sub[f"{c}_drop"].mean()) for c in avail}
        row["schedule"]=_sl(s); rows.append(row)
    if not rows: return
    tab=pd.DataFrame(rows)[["schedule"]+avail]; arr=tab[avail].to_numpy(float)
    mp=np.nanargmax(arr,axis=0)
    lines=[r"\begin{table}[t]",r"\centering",r"\small",
           r"\caption{Per-channel behavioral degradation by stressor (mean pre$-$shock). "
           r"Negative values indicate the stressor improved that channel. "
           r"\textbf{Bold} = max degradation per channel.}",
           r"\label{tab:channel}",r"\begin{tabular}{l"+"r"*len(avail)+"}",r"\toprule",
           "Stressor & "+" & ".join([CH_LABEL[c] for c in avail])+r" \\",r"\midrule"]
    for i,row in tab.iterrows():
        cells=[]
        for j,c in enumerate(avail):
            v=float(row.get(c,np.nan))
            if not math.isfinite(v): cells.append("--")
            elif i==mp[j]: cells.append(r"\textbf{"+f"{v:+.3f}"+"}")
            else: cells.append(f"{v:+.3f}")
        lines.append(f"{row['schedule']} & "+" & ".join(cells)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    _wtex(td/"tab04_channel.tex",lines)

def tab_cvar_arcus(df,td):
    cvar=_col("cvar_shock_05","cvar_shock_25",df=df); cr=_col("collapse_rate_shock","collapse_rate_shock__mean",df=df)
    sc="leaderboard_score"
    if not cvar or not cr or sc not in df.columns: return
    sub=df[(df["eval_mode"]=="deterministic")&df["schedule"].isin(ENV_S)].copy()
    sub=sub.dropna(subset=[cvar,cr,sc])
    if len(sub)<20: return
    sub["cvar_norm"]=sub.groupby("env")[cvar].transform(lambda x:(x-x.min())/(x.max()-x.min()+1e-8))
    r1,p1,lo1,hi1=bootstrap_r(sub["cvar_norm"].values,sub[cr].values)
    r2,p2,lo2,hi2=bootstrap_r(sub["cvar_norm"].values,sub[sc].values)
    disagree=sub[(sub["cvar_norm"]>0.5)&(sub[cr]<0.20)]
    lines=[r"\begin{table}[t]",r"\centering",r"\small",
           r"\caption{CVaR-5 vs ARCUS stability: correlation analysis. "
           r"CVaR and ARCUS are positively correlated but not interchangeable. "
           r"Disagreement cases = high CVaR tail risk but low ARCUS degeneracy: "
           r"policies where return tail is wide but behaviour remains stable under stress.}",
           r"\label{tab:cvar_arcus}",r"\begin{tabular}{lrrrr}",r"\toprule",
           r"Comparison & $r$ & 95\% CI & $p$ & $n$ \\",r"\midrule",
           f"CVaR-5 vs policy degeneracy & {r1:+.3f} & [{lo1:.3f},{hi1:.3f}] & {p1:.2e} & {len(sub)}"+r" \\",
           f"CVaR-5 vs ARCUS score & {r2:+.3f} & [{lo2:.3f},{hi2:.3f}] & {p2:.2e} & {len(sub)}"+r" \\",
           r"\midrule",
           f"High CVaR, low degeneracy (disagreements) & \\multicolumn{{4}}{{r}}{{{len(disagree)}/{len(sub)} ({len(disagree)/len(sub)*100:.0f}\\%)}}"+r" \\",
           r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    _wtex(td/"tab05_cvar_arcus.tex",lines)

def main():
    ap=argparse.ArgumentParser(description="ARCUS-H compare.py v4.2")
    ap.add_argument("--root",        default=None)
    ap.add_argument("--leaderboard", default=None)
    ap.add_argument("--per_episode", default=None, help="Path to per_episode.csv")
    ap.add_argument("--plots_dir",   default=None)
    ap.add_argument("--plots",       action="store_true")
    ap.add_argument("--print",       action="store_true")
    ap.add_argument("--write_csv",   action="store_true")
    args=ap.parse_args()

    root   = Path(args.root)        if args.root        else None
    lb_csv = Path(args.leaderboard) if args.leaderboard else (root/"leaderboard.csv" if root else None)
    pd_dir = Path(args.plots_dir)   if args.plots_dir   else (root/"plots" if root else Path("plots"))

    df_raw = load_data(root=root, leaderboard_csv=lb_csv)
    df_pe  = load_per_episode(args.per_episode)

    if "seed" in df_raw.columns:
        print(f"[INFO] Seed-level: {len(df_raw)} rows. Aggregating...")
        df = prepare(aggregate(df_raw))
    else:
        df = prepare(df_raw.copy())
    df_raw_prep = prepare(df_raw.copy())

    if args.print:
        sub=_det_env(df).dropna(subset=["leaderboard_score"])
        top=(sub.sort_values(["env","schedule","leaderboard_score"],ascending=[True,True,False])
             .groupby(["env","schedule"],dropna=False).head(5))
        pd.set_option("display.max_rows",300); pd.set_option("display.width",200)
        print("\n=== ARCUS-H LEADERBOARD (top 5) ===")
        cols=[c for c in ["env","schedule","algo","eval_mode","leaderboard_score",
                          "reward_norm","collapse_rate_shock"] if c in top.columns]
        print(top[cols].to_string(index=False))

    if args.write_csv and root:
        out=root/"leaderboard_aggregated.csv"
        df.sort_values(["env","schedule","eval_mode","leaderboard_score"],
                       ascending=[True,True,True,False]).to_csv(out,index=False)
        print(f"[OK] {out}")

    if not args.plots and not args.social: return

    td=pd_dir/"tables"; td.mkdir(parents=True,exist_ok=True)
    pd_dir.mkdir(parents=True,exist_ok=True)

    if args.plots:
        print(f"\n[PLOTS] -> {pd_dir}\n")
        stats,env_df,all_df = run_corr(df,pd_dir)
        print("--- Paper figures 01-25 ---")
        fig01_methodology(pd_dir)
        fig02_degeneracy_heatmap(df,pd_dir)
        fig03_suite_collapse(df,pd_dir)
        fig04_correlation_scatter(df,stats,env_df,all_df,pd_dir)
        fig05_rank_disagreement(df,pd_dir)
        fig06_fragile_robust(df,pd_dir)
        fig07_sac_td3_on(df,pd_dir)
        fig08_channel_heatmap(df,pd_dir)
        fig09_radar_channels(df,pd_dir)
        fig10_algo_family(df,pd_dir)
        fig11_cvar_vs_arcus(df,pd_dir)
        fig12_channel_mad(df,pd_dir)
        fig13_walker2d(df,pd_dir)
        fig14_atari_comparison(df,pd_dir)
        fig15_fpr_validation(df_raw_prep,pd_dir)
        fig16_stoch_vs_det(df,pd_dir)
        fig17_algo_ranking(df,pd_dir)
        fig18_reward_drop(df,pd_dir)
        fig19_score_density(df,pd_dir)
        fig20_channel_drop_density(df,pd_dir)
        fig21_cvar_density(df,pd_dir)
        fig22_seed_variance(df_raw_prep,pd_dir)
        fig23_per_episode_channels(df_pe,pd_dir)
        fig24_atari_robustness_density(df,pd_dir)
        fig25_mujoco_deepdive(df,pd_dir)
        print("\n--- Tables ---")
        tab_degeneracy(df,td)
        tab_corr(stats,td)
        tab_fpr(df_raw_prep,td)
        tab_channel(df,td)
        tab_cvar_arcus(df,td)
        sd=pd_dir/"stats"; sd.mkdir(parents=True,exist_ok=True)
        (sd/"summary.json").write_text(json.dumps({
            "compare_version":"4.2","n_envs":int(df["env"].nunique()),
            "n_algos":int(df["algo"].nunique()),"n_schedules":int(df["schedule"].nunique()),
            "env_stressors":ENV_S,"reward_stressors":RWD_S,
            "total_seed_rows":len(df_raw),"total_episodes":len(df_raw)*120,
            "correlation":stats},indent=2,default=str))
        print(f"\n[OK] All 25 paper figures + 5 tables -> {pd_dir}")

if __name__=="__main__":
    main()
