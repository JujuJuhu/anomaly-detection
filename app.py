"""
Streamlit Anomaly Detection Dashboard
Pure NumPy · Reduced False Positives (adaptive threshold + two-pass + feature engineering)
Run: pip install streamlit pandas numpy matplotlib seaborn reportlab
     python -m streamlit run app.py
"""

import json,warnings,datetime,io
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

# ═══════════════════════ PAGE CONFIG + CSS ════════════════════════════════════
st.set_page_config(page_title="Anomaly Detection Dashboard",page_icon="🔍",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0b0f1a}
[data-testid="stSidebar"]{background:#0f1520 !important;border-right:1px solid #1e2740}
section[data-testid="stSidebar"] *{color:#94a3b8 !important}
section[data-testid="stSidebar"] h2{color:#e2e8f0 !important}
h1,h2,h3,h4{color:#e2e8f0 !important}
hr{border-color:#1e2740 !important}
[data-testid="metric-container"]{background:#111827;border:1px solid #1e2740;border-radius:12px;padding:1rem 1.25rem}
[data-testid="stMetricLabel"]{color:#6b7a99 !important;font-size:.75rem !important;text-transform:uppercase;letter-spacing:.07em}
[data-testid="stMetricValue"]{color:#e2e8f0 !important;font-size:1.8rem !important;font-weight:700 !important}
[data-testid="stInfo"]{background:#0f2340 !important;border-left:3px solid #3b82f6 !important;border-radius:8px}
[data-testid="stSuccess"]{background:#0f2e1e !important;border-left:3px solid #22c55e !important;border-radius:8px}
[data-testid="stWarning"]{background:#2a1f0a !important;border-left:3px solid #f59e0b !important;border-radius:8px}
[data-testid="stError"]{background:#2a0f0f !important;border-left:3px solid #ef4444 !important;border-radius:8px}
[data-testid="stButton"]>button{background:linear-gradient(135deg,#3b82f6,#6366f1) !important;color:#fff !important;border:none !important;border-radius:8px !important;font-weight:600 !important;padding:.5rem 1.2rem !important}
[data-testid="stButton"]>button:hover{opacity:.85 !important}
[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden}
.s-card{background:#111827;border:1px solid #1e2740;border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:1.2rem}
.s-card-title{font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:#6b7a99;margin-bottom:.9rem;font-weight:600}
.rpt-wrap{background:#1a1f2e;border:1px solid #1e2740;border-radius:12px;padding:2rem}
.rpt-hdr{background:linear-gradient(135deg,#1e3a5f,#2563eb);border-radius:10px;padding:1.4rem 1.8rem;margin-bottom:1.4rem}
.rpt-title{font-size:1.5rem;font-weight:800;color:#fff;margin:0 0 .2rem}
.rpt-sub{font-size:.85rem;color:#93c5fd;margin:0}
.rpt-meta{display:flex;gap:2rem;margin-top:.9rem;flex-wrap:wrap}
.rpt-mi{font-size:.77rem;color:#bfdbfe}
.rpt-mi span{color:#fff;font-weight:600}
.sev-badge{display:inline-block;padding:.3rem 1rem;border-radius:6px;font-size:.88rem;font-weight:700;float:right;margin-top:.2rem}
.sev-CRITICAL{background:#3d0f0f;color:#ef4444;border:1px solid #ef4444}
.sev-HIGH{background:#3d2a0a;color:#f59e0b;border:1px solid #f59e0b}
.sev-MEDIUM{background:#0f2340;color:#3b82f6;border:1px solid #3b82f6}
.sev-LOW{background:#0f2e1e;color:#22c55e;border:1px solid #22c55e}
.sev-CLEAN{background:#0f2e1e;color:#22c55e;border:1px solid #22c55e}
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:1.4rem}
.stat-box{background:#111827;border:1px solid #1e2740;border-radius:10px;padding:.9rem 1rem}
.stat-top{height:3px;border-radius:2px;margin-bottom:.7rem}
.stat-lbl{font-size:.67rem;text-transform:uppercase;letter-spacing:.07em;color:#6b7a99;margin-bottom:.25rem}
.stat-val{font-size:1.5rem;font-weight:700;color:#e2e8f0}
.stat-sub{font-size:.7rem;color:#475569;margin-top:.15rem}
.sec-hdr{display:flex;align-items:center;gap:.6rem;margin:1.2rem 0 .6rem;padding-bottom:.35rem;border-bottom:2px solid #1e3a5f}
.sec-num{background:#2563eb;color:#fff;border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;font-size:.68rem;font-weight:700;flex-shrink:0}
.sec-title{font-size:.95rem;font-weight:700;color:#e2e8f0;text-transform:uppercase;letter-spacing:.05em}
.rpt-sum{background:#0f1520;border-left:4px solid #2563eb;border-radius:4px;padding:.9rem 1.1rem;color:#cbd5e1;font-size:.88rem;line-height:1.75}
.sev-box{background:#0f1520;border:1px solid #1e3a5f;border-radius:8px;padding:.75rem 1rem;margin-top:.7rem;display:flex;gap:.7rem}
.sev-text{color:#94a3b8;font-size:.86rem;line-height:1.6}
.find-list{list-style:none;padding:0;margin:0}
.find-list li{display:flex;gap:.7rem;align-items:flex-start;padding:.5rem 0;border-bottom:1px solid #1e2740}
.find-list li:last-child{border-bottom:none}
.find-num{background:#2563eb;color:#fff;border-radius:50%;width:19px;height:19px;min-width:19px;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;margin-top:.15rem}
.find-text{color:#94a3b8;font-size:.86rem;line-height:1.65}
.pat-list{list-style:none;padding:0;margin:0}
.pat-list li{display:flex;gap:.7rem;align-items:flex-start;padding:.5rem 0;border-bottom:1px solid #1e2740}
.pat-list li:last-child{border-bottom:none}
.pat-dot{width:7px;height:7px;min-width:7px;background:#f59e0b;border-radius:2px;margin-top:.45rem}
.pat-text{color:#94a3b8;font-size:.86rem;line-height:1.65}
.sc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.sc-box{background:#0f1520;border:1px solid #1e2740;border-radius:8px;padding:.7rem;text-align:center}
.sc-lbl{font-size:.65rem;color:#6b7a99;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.25rem}
.sc-val{font-size:1.1rem;font-weight:700}
.rpt-footer{background:#0f1520;border-top:1px solid #1e2740;border-radius:0 0 10px 10px;padding:.65rem 1rem;display:flex;justify-content:space-between;margin-top:1.2rem}
.rpt-footer p{font-size:.7rem;color:#3a4a6a;margin:0}
</style>
""",unsafe_allow_html=True)

# ═══════════════════════ PALETTE ═════════════════════════════════════════════
DARK_BG="#0b0f1a";CARD_BG="#111827";BORDER="#1e2740"
TEXT_PRI="#e2e8f0";TEXT_MUT="#6b7a99"
BLUE="#3b82f6";CRIMSON="#ef4444";AMBER="#f59e0b";GREEN="#22c55e"
PDF_BLUE="#2563eb";PDF_RED="#dc2626";PDF_GREEN="#16a34a";PDF_AMB="#d97706"

def _mpl_dark():
    plt.rcParams.update({"figure.facecolor":DARK_BG,"axes.facecolor":CARD_BG,
        "axes.edgecolor":BORDER,"axes.labelcolor":TEXT_MUT,"axes.titlecolor":TEXT_PRI,
        "xtick.color":TEXT_MUT,"ytick.color":TEXT_MUT,"text.color":TEXT_PRI,
        "grid.color":BORDER,"grid.linestyle":"--","grid.alpha":.5,
        "axes.titlesize":12,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
        "legend.facecolor":CARD_BG,"legend.edgecolor":BORDER,"legend.fontsize":9})
def _mpl_light():
    plt.rcParams.update({"figure.facecolor":"#ffffff","axes.facecolor":"#f8fafc",
        "axes.edgecolor":"#cbd5e1","axes.labelcolor":"#475569","axes.titlecolor":"#1e293b",
        "xtick.color":"#64748b","ytick.color":"#64748b","text.color":"#1e293b",
        "grid.color":"#e2e8f0","grid.linestyle":"--","grid.alpha":.6,
        "axes.titlesize":11,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,
        "legend.facecolor":"#ffffff","legend.edgecolor":"#cbd5e1","legend.fontsize":8})

# ═══════════════════════ PURE-NUMPY ML ═══════════════════════════════════════
def _c(n):
    if n<=1:return 0.0
    return 2*(np.log(n-1)+0.5772156649)-2*(n-1)/n

class _ITree:
    __slots__=("md","sf","sv","l","r","size","leaf")
    def __init__(s,md):s.md=md;s.sf=s.sv=s.l=s.r=None;s.size=0;s.leaf=False
    def fit(s,X,d=0):
        s.size=len(X)
        if d>=s.md or s.size<=1:s.leaf=True;return s
        f=np.random.randint(0,X.shape[1]);col=X[:,f];lo,hi=col.min(),col.max()
        if lo==hi:s.leaf=True;return s
        s.sf=f;s.sv=np.random.uniform(lo,hi);m=col<s.sv
        s.l=_ITree(s.md).fit(X[m],d+1);s.r=_ITree(s.md).fit(X[~m],d+1);return s
    def path(s,x,d=0):
        if s.leaf:return d+_c(s.size)
        return(s.l if x[s.sf]<s.sv else s.r).path(x,d+1)

class IForest:
    def __init__(s,n=200,ms=512,cont=0.05,seed=42):
        s.n=n;s.ms=ms;s.cont=cont;s.seed=seed;s.trees=[];s.thr=None;s.cn=1.0
    def fit(s,X):
        np.random.seed(s.seed);N=len(X);ss=min(s.ms,N)
        s.cn=_c(ss);md=int(np.ceil(np.log2(ss))) if ss>1 else 1
        s.trees=[_ITree(md).fit(X[np.random.choice(N,ss,replace=False)]) for _ in range(s.n)]
        sc=s._raw(X);s.thr=np.percentile(sc,100*(1-s.cont));return s
    def _raw(s,X):
        avg=np.array([np.mean([t.path(x) for t in s.trees]) for x in X])
        return -avg/s.cn if s.cn else np.zeros(len(X))
    def score(s,X):return s._raw(np.array(X,float))
    def predict(s,X):return np.where(s.score(X)>=s.thr,1,-1)

class Scaler:
    def fit_transform(s,X):
        X=np.array(X,float);s.mu=X.mean(0);s.sd=X.std(0)
        s.sd[s.sd==0]=1.0;return(X-s.mu)/s.sd

def pca2(X):
    X=np.array(X,float);X-=X.mean(0)
    var=X.var(0);mask=var>1e-10
    if mask.sum()<2:return None
    X=X[:,mask];cov=np.cov(X.T)
    if cov.ndim<2:return None
    _,vecs=np.linalg.eigh(cov)
    return X@vecs[:,::-1][:,:2]

# ═══════════════════════ FEATURE ENGINEERING (Suricata) ═══════════════════════
def engineer_features(df):
    """Add network-specific computed features before ML."""
    df = df.copy()

    # Packet ratio (asymmetry detector — scans/exfil have extreme ratios)
    if "flow_pkts_toserver" in df.columns and "flow_pkts_toclient" in df.columns:
        df["_feat_pkt_ratio"] = df["flow_pkts_toserver"] / (df["flow_pkts_toclient"] + 1)

    # Byte ratio
    if "flow_bytes_toserver" in df.columns and "flow_bytes_toclient" in df.columns:
        df["_feat_byte_ratio"] = df["flow_bytes_toserver"] / (df["flow_bytes_toclient"] + 1)

    # Bytes per packet (tunnelling/exfil detection)
    if "flow_bytes_toserver" in df.columns and "flow_pkts_toserver" in df.columns:
        df["_feat_bytes_per_pkt"] = df["flow_bytes_toserver"] / (df["flow_pkts_toserver"] + 1)

    # Connection count per source IP (burst/flood detection)
    if "src_ip" in df.columns:
        df["_feat_src_conn_count"] = df.groupby("src_ip")["src_ip"].transform("count")

    # Unique destination count per source IP (scan detection)
    if "src_ip" in df.columns and "dest_ip" in df.columns:
        df["_feat_src_unique_dests"] = df.groupby("src_ip")["dest_ip"].transform("nunique")

    # Unique destination ports per source IP (port scan detection)
    if "src_ip" in df.columns and "dest_port" in df.columns:
        df["_feat_src_unique_dports"] = df.groupby("src_ip")["dest_port"].transform("nunique")

    # Source port entropy indicator (randomised source ports = suspicious)
    if "src_ip" in df.columns and "src_port" in df.columns:
        df["_feat_src_unique_sports"] = df.groupby("src_ip")["src_port"].transform("nunique")

    # Destination IP connection count (targeted host detection)
    if "dest_ip" in df.columns:
        df["_feat_dest_conn_count"] = df.groupby("dest_ip")["dest_ip"].transform("count")

    return df

# ═══════════════════════ FEATURE SELECTION ═══════════════════════════════════
def select_features(X, min_var=0.01, max_corr=0.95):
    """Remove low-variance and highly correlated features (pure NumPy)."""
    # Drop low-variance
    var = X.var(axis=0)
    high_var = var > min_var
    X = X.loc[:, high_var]

    if X.shape[1] < 2:
        return X

    # Drop highly correlated (keep first of each correlated pair)
    arr = X.values.astype(float)
    n = arr.shape[1]
    # Correlation matrix via numpy
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    normed = (arr - arr.mean(axis=0)) / std
    corr = (normed.T @ normed) / len(arr)

    drop_idx = set()
    for i in range(n):
        if i in drop_idx:
            continue
        for j in range(i + 1, n):
            if j in drop_idx:
                continue
            if abs(corr[i, j]) > max_corr:
                drop_idx.add(j)

    keep = [i for i in range(n) if i not in drop_idx]
    X = X.iloc[:, keep]
    return X

# ═══════════════════════ TWO-PASS CONFIRMATION ═══════════════════════════════
def two_pass_confirm(scores, if_labels, z_threshold=2.0):
    """
    Pass 1: Isolation Forest flags candidates (if_labels == -1)
    Pass 2: Confirm only those whose score z-score exceeds threshold.
    Returns final labels: 1 = confirmed anomaly, 0 = normal.
    """
    mean = scores.mean()
    std = scores.std()
    if std == 0:
        return np.zeros(len(scores), dtype=int)

    z_scores = np.abs((scores - mean) / std)

    final = np.zeros(len(scores), dtype=int)
    candidates = (if_labels == -1)

    # Only confirm candidates that also exceed z-score threshold
    confirmed = candidates & (z_scores > z_threshold)
    final[confirmed] = 1

    return final, z_scores

# ═══════════════════════ ANALYSIS PIPELINE ═══════════════════════════════════
@st.cache_data(show_spinner=False)
def run_analysis(fb: bytes, cont: float, z_thr: float, feat_eng: bool, cv: int = 4):
    logs = []
    try:
        for ln in fb.decode("utf-8").splitlines():
            if ln.strip():
                logs.append(json.loads(ln))
        df = pd.DataFrame(logs)
    except Exception as e:
        return None, None, str(e), {}
    if df.empty:
        return None, None, "No rows.", {}

    steps = {}  # track what happened

    # Flatten nested dicts
    for col in list(df.columns):
        if any(isinstance(x, dict) for x in df[col].dropna()):
            try:
                mask = df[col].apply(lambda x: isinstance(x, dict))
                flat = pd.json_normalize(df.loc[mask, col]).add_prefix(f"{col}_")
                flat.index = df[mask].index
                df = df.drop(columns=[col]).join(flat)
            except:
                pass

    # Lists → str
    for col in df.columns:
        if any(isinstance(x, list) for x in df[col].dropna()):
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    # Feature engineering
    if feat_eng:
        orig_cols = len(df.columns)
        df = engineer_features(df)
        new_cols = len(df.columns) - orig_cols
        steps["feat_eng"] = f"{new_cols} computed features added"
    else:
        steps["feat_eng"] = "Disabled"

    # Separate features
    excl = {"timestamp", "src_ip", "dest_ip", "flow_id", "in_iface",
            "pkt_src", "app_proto", "proto"}
    num = [c for c in df.select_dtypes(include=np.number).columns if c not in excl]
    cat = [c for c in df.select_dtypes(include=["object", "bool"]).columns if c not in excl]

    for col in num:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    enc = pd.get_dummies(df[cat], dummy_na=True, drop_first=True, dtype=int) if cat else pd.DataFrame(index=df.index)
    scaler = Scaler()
    scaled = pd.DataFrame(scaler.fit_transform(df[num]), columns=num, index=df.index) if num else pd.DataFrame(index=df.index)

    X = pd.concat([scaled, enc], axis=1).replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    if X.empty:
        return None, None, "No usable features.", {}

    cols_before = X.shape[1]

    # Feature selection — remove noise
    X = select_features(X, min_var=0.01, max_corr=0.95)
    cols_after = X.shape[1]
    steps["feat_select"] = f"{cols_before} → {cols_after} features ({cols_before - cols_after} dropped)"

    if X.empty:
        return None, None, "No features after selection.", {}

    # ── Pass 1: Isolation Forest (200 trees, 512 subsample) ──────────────────
    model = IForest(n=200, ms=min(512, len(X)), cont=cont, seed=42)
    model.fit(X.values)

    raw_scores = model.score(X.values)
    if_labels = model.predict(X.values)  # 1=normal, -1=anomaly

    pass1_count = int((if_labels == -1).sum())
    steps["pass1"] = f"{pass1_count} candidates flagged by Isolation Forest"

    # ── Pass 2: Z-score confirmation ─────────────────────────────────────────
    final_labels, z_scores = two_pass_confirm(raw_scores, if_labels, z_thr)

    pass2_count = int(final_labels.sum())
    eliminated = pass1_count - pass2_count
    steps["pass2"] = f"{pass2_count} confirmed (z>{z_thr:.1f}), {eliminated} false positives eliminated"

    df = df.copy()
    df["anomaly_score"] = raw_scores
    df["z_score"] = z_scores
    df["if_candidate"] = (if_labels == -1).astype(int)  # pass 1 result
    df["predicted_label"] = final_labels                  # pass 2 confirmed
    df["anomaly_prediction"] = np.where(final_labels == 1, -1, 1)
    df["confidence"] = np.clip(z_scores / max(z_thr * 2, 1), 0, 1)  # 0-1 scale

    return df, X, None, steps

# ═══════════════════════ CHART GENERATORS ════════════════════════════════════
def _mk_score(df, dark=True):
    (_mpl_dark if dark else _mpl_light)()
    bg = "#0b0f1a" if dark else "#ffffff";cbg = CARD_BG if dark else "#f8fafc"
    bl = BLUE if dark else PDF_BLUE;rd = CRIMSON if dark else PDF_RED
    am = AMBER if dark else PDF_AMB;bd = BORDER if dark else "#cbd5e1"
    has_anom = df["predicted_label"].sum() > 0
    f, a = plt.subplots(figsize=(8 if dark else 5.5, 3.8 if dark else 2.8), facecolor=bg)
    a.set_facecolor(cbg)
    if has_anom:
        a.hist(df.loc[df["predicted_label"] == 0, "anomaly_score"], bins=40, color=bl, alpha=.75, label="Normal", edgecolor="none")
        a.hist(df.loc[df["predicted_label"] == 1, "anomaly_score"], bins=40, color=rd, alpha=.75, label="Anomaly", edgecolor="none")
        thr = df.loc[df["predicted_label"] == 1, "anomaly_score"].max()
        a.axvline(thr, color=am, ls="--", lw=1.5, label="Threshold")
    else:
        a.hist(df["anomaly_score"], bins=40, color=bl, alpha=.8, label="Normal (all)", edgecolor="none")
    t = ("Figure 1 — " if not dark else "") + "Score Distribution"
    a.set_title(t);a.set_xlabel("Anomaly score");a.set_ylabel("Count");a.legend();a.grid(True)
    for s in a.spines.values():s.set_edgecolor(bd)
    f.tight_layout();return f

def _mk_top15(df, dark=True):
    (_mpl_dark if dark else _mpl_light)()
    bg = "#0b0f1a" if dark else "#ffffff";cbg = CARD_BG if dark else "#f8fafc"
    bl = BLUE if dark else PDF_BLUE;rd = CRIMSON if dark else PDF_RED
    am = AMBER if dark else PDF_AMB;bd = BORDER if dark else "#cbd5e1"
    top = df.sort_values("anomaly_score").head(15)
    colors = [rd if v == 1 else bl for v in top["predicted_label"]]
    f, a = plt.subplots(figsize=(8 if dark else 5.5, 3.8 if dark else 2.8), facecolor=bg)
    a.set_facecolor(cbg)
    a.barh([str(i) for i in range(1, len(top) + 1)], top["anomaly_score"], color=colors, edgecolor="none", height=.65)
    a.axvline(0, color=am, ls="--", lw=1.2, alpha=.7)
    t = ("Figure 2 — " if not dark else "") + "Top 15 Lowest Scores"
    a.set_title(t);a.set_xlabel("Score");a.invert_yaxis()
    has = df["predicted_label"].sum() > 0
    if has:
        a.legend(handles=[mpatches.Patch(color=rd, label="Confirmed anomaly"), mpatches.Patch(color=bl, label="Normal")])
    else:
        a.legend(handles=[mpatches.Patch(color=bl, label="Normal (all)")])
    a.grid(True, axis="x")
    for s in a.spines.values():s.set_edgecolor(bd)
    f.tight_layout();return f

def _mk_pca(df, X, dark=True):
    try:
        (_mpl_dark if dark else _mpl_light)()
        bg = "#0b0f1a" if dark else "#ffffff";cbg = CARD_BG if dark else "#f8fafc"
        bl = BLUE if dark else PDF_BLUE;rd = CRIMSON if dark else PDF_RED
        bd = BORDER if dark else "#cbd5e1"
        comps = pca2(X.values)
        if comps is None:return None
        pdf = pd.DataFrame(comps, columns=["PC1", "PC2"]);pdf["lbl"] = df["predicted_label"].values
        f, a = plt.subplots(figsize=(8 if dark else 5.5, 3.8 if dark else 2.8), facecolor=bg)
        a.set_facecolor(cbg)
        n = pdf[pdf["lbl"] == 0];av = pdf[pdf["lbl"] == 1]
        has = len(av) > 0
        if len(n) > 0:a.scatter(n["PC1"], n["PC2"], s=14 if has else 18, color=bl, alpha=.5 if has else .7, label="Normal", linewidths=0)
        if has:a.scatter(av["PC1"], av["PC2"], s=45, color=rd, alpha=.85, label="Anomaly", marker="X", linewidths=0)
        t = ("Figure 3 — " if not dark else "") + "PCA Projection"
        a.set_title(t);a.set_xlabel("PC1");a.set_ylabel("PC2");a.legend();a.grid(True, alpha=.4)
        for s in a.spines.values():s.set_edgecolor(bd)
        f.tight_layout();return f
    except Exception:return None

def _mk_event(df, dark=True):
    if "event_type" not in df.columns:return None
    try:
        (_mpl_dark if dark else _mpl_light)()
        bg = "#0b0f1a" if dark else "#ffffff";cbg = CARD_BG if dark else "#f8fafc"
        bl = BLUE if dark else PDF_BLUE;rd = CRIMSON if dark else PDF_RED
        bd = BORDER if dark else "#cbd5e1"
        f, a = plt.subplots(figsize=(8 if dark else 5.5, 3.8 if dark else 2.8), facecolor=bg)
        a.set_facecolor(cbg)
        has = df["predicted_label"].sum() > 0
        if has:
            sns.countplot(x="event_type", hue="anomaly_prediction", data=df,
                          palette={1: bl, -1: rd}, ax=a, edgecolor="none")
            a.legend(handles=[mpatches.Patch(color=bl, label="Normal"), mpatches.Patch(color=rd, label="Anomaly")])
        else:
            order = df["event_type"].value_counts().index
            sns.countplot(x="event_type", data=df, color=bl, ax=a, edgecolor="none", order=order)
            a.legend(handles=[mpatches.Patch(color=bl, label="Normal (all)")])
        t = ("Figure 4 — " if not dark else "") + "Event Type Breakdown"
        a.set_title(t);a.set_xlabel("Event type");a.set_ylabel("Count")
        a.grid(True, axis="y");plt.xticks(rotation=30, ha="right")
        for s in a.spines.values():s.set_edgecolor(bd)
        f.tight_layout();return f
    except Exception:return None

# NEW: Two-pass comparison chart
def _mk_twopass(df, dark=True):
    (_mpl_dark if dark else _mpl_light)()
    bg = "#0b0f1a" if dark else "#ffffff";cbg = CARD_BG if dark else "#f8fafc"
    bl = BLUE if dark else PDF_BLUE;rd = CRIMSON if dark else PDF_RED
    yl = AMBER if dark else PDF_AMB;bd = BORDER if dark else "#cbd5e1"

    p1 = int(df["if_candidate"].sum())
    p2 = int(df["predicted_label"].sum())
    elim = p1 - p2
    total = len(df)

    f, a = plt.subplots(figsize=(8 if dark else 5.5, 3.8 if dark else 2.8), facecolor=bg)
    a.set_facecolor(cbg)

    labels = ["Pass 1:\nIsolation Forest", "Eliminated:\nFalse Positives", "Pass 2:\nConfirmed"]
    vals = [p1, elim, p2]
    colors = [yl, bl, rd]
    bars = a.bar(labels, vals, color=colors, edgecolor="none", width=0.55)

    for bar, val in zip(bars, vals):
        a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.02,
               str(val), ha="center", va="bottom", fontweight="bold",
               color=TEXT_PRI if dark else "#1e293b", fontsize=12)

    t = ("Figure 5 — " if not dark else "") + "Two-Pass Filtering"
    a.set_title(t);a.set_ylabel("Records")
    a.grid(True, axis="y", alpha=.4)
    for s in a.spines.values():s.set_edgecolor(bd)
    f.tight_layout();return f

def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig);buf.seek(0);return buf.read()

def make_pdf_charts(df, X):
    C = {}
    for k, fn in [("score", lambda: _mk_score(df, False)), ("top15", lambda: _mk_top15(df, False)),
                   ("pca", lambda: _mk_pca(df, X, False)), ("event", lambda: _mk_event(df, False)),
                   ("twopass", lambda: _mk_twopass(df, False))]:
        try:
            f = fn()
            if f:C[k] = _png(f)
        except Exception:pass
    return C

# ═══════════════════════ REPORT GENERATOR ════════════════════════════════════
def gen_report(df, X, cont, z_thr, steps):
    now = datetime.datetime.now()
    n_tot = len(df);n_a = int((df["predicted_label"] == 1).sum());n_n = n_tot - n_a
    ratio = n_a / n_tot if n_tot else 0
    scores = df["anomaly_score"]
    asc = df.loc[df["predicted_label"] == 1, "anomaly_score"]
    is_clean = (n_a == 0)
    p1_count = int(df["if_candidate"].sum())
    elim = p1_count - n_a

    if is_clean:sev, sev_col = "CLEAN", "#22c55e";sev_r = "No anomalies survived two-pass validation. All traffic is normal."
    elif ratio >= .20:sev, sev_col = "CRITICAL", "#ef4444";sev_r = f"{ratio:.1%} flagged — far above baseline."
    elif ratio >= .10:sev, sev_col = "HIGH", "#f59e0b";sev_r = f"{ratio:.1%} exceeds 10% threshold."
    elif ratio >= .05:sev, sev_col = "MEDIUM", "#3b82f6";sev_r = f"{ratio:.1%} within moderate band."
    else:sev, sev_col = "LOW", "#22c55e";sev_r = f"{ratio:.1%} below 5% baseline."

    findings = []
    if is_clean:
        findings.append(f"All {n_tot:,} records passed both Isolation Forest and z-score validation — zero confirmed anomalies.")
        if p1_count > 0:
            findings.append(f"Isolation Forest initially flagged {p1_count:,} candidates, but all {p1_count:,} were eliminated by z-score confirmation (z>{z_thr:.1f}).")
        findings.append("Traffic profile is homogeneous and consistent with normal network behaviour.")
    else:
        findings.append(f"Pass 1 (Isolation Forest) flagged {p1_count:,} candidates at contamination rate {cont:.0%}.")
        findings.append(f"Pass 2 (z-score>{z_thr:.1f}) confirmed {n_a:,} true anomalies — eliminated {elim:,} false positives ({elim / max(p1_count, 1):.0%} FP reduction).")

    ms = float(asc.min()) if not asc.empty else float(scores.min())
    findings.append(f"Lowest score: {ms:.4f} — {'significantly' if ms < -0.3 else 'moderately' if ms < -0.1 else 'slightly'} below the mean.")
    ss = float(scores.std())
    findings.append(f"Score std dev: {ss:.4f} — {'high variance' if ss > 0.15 else 'low variance'}.")

    if "event_type" in df.columns:
        src = df if is_clean else df[df["predicted_label"] == 1]
        ae = src["event_type"].value_counts()
        if not ae.empty:
            findings.append(f"Top event type: '{ae.index[0]}' ({int(ae.iloc[0])} records).")
    if "src_ip" in df.columns:
        src = df if is_clean else df[df["predicted_label"] == 1]
        ti = src["src_ip"].value_counts().head(3)
        if not ti.empty:findings.append(f"Top source IPs: {', '.join(ti.index.tolist())}.")
    if "dest_ip" in df.columns:
        src = df if is_clean else df[df["predicted_label"] == 1]
        td = src["dest_ip"].value_counts().head(3)
        if not td.empty:findings.append(f"Top dest IPs: {', '.join(td.index.tolist())}.")

    patterns = []
    tgt = asc if not asc.empty else scores
    iqr = float(tgt.quantile(.75)) - float(tgt.quantile(.25))
    if iqr < 0.05:patterns.append("Scores tightly clustered — single dominant pattern.")
    elif iqr > 0.20:patterns.append("Wide IQR — multiple distinct clusters.")
    else:patterns.append("Moderate spread — 2–3 distinguishable groups.")
    nm = float(df.loc[df["predicted_label"] == 0, "anomaly_score"].mean()) if n_n > 0 else float(scores.mean())
    am = float(asc.mean()) if not asc.empty else nm
    sep = nm - am
    if is_clean:
        patterns.append(f"Mean score: {nm:.4f} — all records cluster around the same baseline.")
    else:
        patterns.append(f"Score gap: normal ({nm:.4f}) vs anomalous ({am:.4f}) = {sep:.4f} — {'strong' if sep > 0.3 else 'moderate' if sep > 0.15 else 'weak'} separation.")
    if "proto" in df.columns:
        src = df if is_clean else df[df["predicted_label"] == 1]
        ap = src["proto"].value_counts()
        if not ap.empty:patterns.append("Protocol mix: " + ", ".join(f"{p} ({c})" for p, c in ap.head(3).items()) + ".")

    # Pipeline info
    patterns.append(f"Pipeline: {steps.get('feat_eng', 'N/A')} → {steps.get('feat_select', 'N/A')} → {steps.get('pass1', 'N/A')} → {steps.get('pass2', 'N/A')}.")

    if is_clean:
        summary = (f"Anomaly detection completed on {now.strftime('%d %B %Y at %H:%M:%S')}. "
                   f"Processed {n_tot:,} records with a two-pass pipeline (Isolation Forest + z-score confirmation). "
                   f"Zero anomalies confirmed — network appears healthy. "
                   f"{'IF flagged ' + str(p1_count) + ' candidates, all eliminated by z-score validation.' if p1_count > 0 else ''}")
    else:
        summary = (f"Anomaly detection completed on {now.strftime('%d %B %Y at %H:%M:%S')}. "
                   f"Two-pass pipeline processed {n_tot:,} records: IF flagged {p1_count:,} candidates, "
                   f"z-score confirmation retained {n_a:,} ({ratio:.2%}), eliminating {elim:,} false positives. "
                   f"Severity: {sev}.")

    tcols = [c for c in ["timestamp", "event_type", "src_ip", "dest_ip", "proto",
                          "anomaly_score", "z_score", "confidence"] if c in df.columns]
    top10 = df.sort_values("anomaly_score").head(10)[tcols].copy()
    for c in ["anomaly_score", "z_score", "confidence"]:
        if c in top10.columns:top10[c] = top10[c].round(4)

    return {"now": now, "sev": sev, "sev_col": sev_col, "sev_r": sev_r, "summary": summary,
            "findings": findings, "patterns": patterns, "top10": top10, "top10_cols": tcols,
            "n_tot": n_tot, "n_a": n_a, "n_n": n_n, "ratio": ratio, "is_clean": is_clean,
            "p1_count": p1_count, "elim": elim, "z_thr": z_thr,
            "smin": float(scores.min()), "smax": float(scores.max()),
            "smean": float(scores.mean()), "sstd": ss, "cont": cont, "steps": steps}

# ═══════════════════════ PDF BUILDER ═════════════════════════════════════════
def build_pdf(r, charts):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage, PageBreak)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
        topMargin=18*mm, bottomMargin=18*mm, title="Incident Report", author="FoxyDucky")
    W = A4[0] - 40*mm
    C = colors.HexColor
    NAVY = C("#1e3a5f");BLUE2 = C("#2563eb");MID = C("#1e293b");LIGHT = C("#f8fafc")
    SLATE = C("#475569");MUTED = C("#94a3b8");WHITE = colors.white
    RED2 = C("#dc2626");AMB2 = C("#d97706");GRN2 = C("#16a34a");BLU2 = C("#2563eb")
    BORD = C("#e2e8f0");HBORD = C("#cbd5e1")
    SEV_C = {"CRITICAL": RED2, "HIGH": AMB2, "MEDIUM": BLU2, "LOW": GRN2, "CLEAN": GRN2}
    sev_color = SEV_C.get(r["sev"], GRN2)
    def P(t, sz=9, c=MID, b=False, a=TA_LEFT, ld=None):
        return Paragraph(str(t), ParagraphStyle("x", fontSize=sz, textColor=c,
            fontName="Helvetica-Bold" if b else "Helvetica", alignment=a,
            leading=ld or sz * 1.45, spaceAfter=2))
    def HR(c=BORD, th=0.5, sb=4, sa=4):
        return HRFlowable(width="100%", thickness=th, color=c, spaceBefore=sb, spaceAfter=sa)
    def img(b, w, h):return RLImage(io.BytesIO(b), width=w, height=h)
    def sec(num, title):
        nb = Table([[P(str(num), 7, WHITE, True, TA_CENTER)]], colWidths=[15], rowHeights=[15],
            style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BLUE2), ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2), ("ROUNDEDCORNERS", [7,7,7,7])]))
        return [Table([[nb, P(title, 10, NAVY, True)]], colWidths=[20, W-20],
            style=TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("BOTTOMPADDING", (0,0), (-1,-1), 3)])), HR(BLUE2, 1)]
    story = []
    # Header
    story += [Table([[P("ANOMALY DETECTION SYSTEM", 9, WHITE, True),
        P("FOXYDUCKY · TASK 3 NOCTRA LUPRA", 8, C("#93c5fd"), a=TA_RIGHT)]], colWidths=[W*.55, W*.45],
        style=TableStyle([("BACKGROUND", (0,0), (-1,-1), NAVY), ("TOPPADDING", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9), ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10), ("ROUNDEDCORNERS", [5,5,5,5])])), Spacer(1, 8)]
    # Title
    sb = Table([[P(r["sev"], 10, WHITE, True, TA_CENTER)]], colWidths=[60], rowHeights=[20],
        style=TableStyle([("BACKGROUND", (0,0), (-1,-1), sev_color), ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4), ("ROUNDEDCORNERS", [4,4,4,4])]))
    story += [Table([[P("INCIDENT REPORT", 20, NAVY, True), sb]], colWidths=[W-70, 70],
        style=TableStyle([("VALIGN", (0,0), (-1,-1), "BOTTOM")])),
        P("Network Anomaly Detection — Two-Pass Isolation Forest", 10, SLATE), Spacer(1, 5),
        Table([[P(f"<b>Date:</b> {r['now'].strftime('%d %B %Y %H:%M:%S')}", 8, MUTED),
            P(f"<b>Contamination:</b> {r['cont']:.0%}", 8, MUTED),
            P(f"<b>Z-threshold:</b> {r['z_thr']:.1f}σ", 8, MUTED)]], colWidths=[W/3]*3,
            style=TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("TOPPADDING", (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5), ("LEFTPADDING", (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8), ("ROUNDEDCORNERS", [4,4,4,4])])),
        Spacer(1, 10), HR(BLUE2, 1.5, 2, 6)]
    # Clean notice
    if r["is_clean"]:
        story += [Table([[P("✅ CLEAN — Two-pass validation confirmed zero anomalies.", 9, C("#166534"), ld=14)]], colWidths=[W],
            style=TableStyle([("BACKGROUND", (0,0), (-1,-1), C("#dcfce7")), ("TOPPADDING", (0,0), (-1,-1), 8),
                ("BOTTOMPADDING", (0,0), (-1,-1), 8), ("LEFTPADDING", (0,0), (-1,-1), 10),
                ("RIGHTPADDING", (0,0), (-1,-1), 10), ("LINEBEFORE", (0,0), (0,-1), 4, GRN2),
                ("ROUNDEDCORNERS", [4,4,4,4])])), Spacer(1, 8)]
    # Stats
    sw = W/4-3
    stats = [("TOTAL", f"{r['n_tot']:,}", "Processed", BLU2),
             ("CONFIRMED", f"{r['n_a']:,}", f"{r['ratio']:.2%}", RED2 if not r["is_clean"] else GRN2),
             ("FP ELIMINATED", f"{r['elim']:,}", f"of {r['p1_count']:,} candidates", AMB2),
             ("STATUS", r["sev"], "Assessment", sev_color)]
    sc = []
    for lbl, val, sub, col in stats:
        bar = Table([[""]], colWidths=[sw], rowHeights=[3], style=TableStyle([("BACKGROUND", (0,0), (-1,-1), col)]))
        sc.append([bar, Spacer(1, 4), P(lbl, 6.5, MUTED, True), P(val, 16, MID, True), P(sub, 7, SLATE)])
    story += [Table([sc], colWidths=[sw]*4, style=TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,0), (-1,-1), LIGHT), ("BOX", (0,0), (0,0), 0.4, HBORD), ("BOX", (1,0), (1,0), 0.4, HBORD),
        ("BOX", (2,0), (2,0), 0.4, HBORD), ("BOX", (3,0), (3,0), 0.4, HBORD),
        ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7)])), Spacer(1, 12)]
    # Sections
    for e in sec("01", "EXECUTIVE SUMMARY"):story.append(e)
    story += [Spacer(1, 4), Table([[P(r["summary"], 9, MID, ld=15, a=TA_JUSTIFY)]], colWidths=[W],
        style=TableStyle([("BACKGROUND", (0,0), (-1,-1), C("#eff6ff")), ("TOPPADDING", (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8), ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10), ("LINEBEFORE", (0,0), (0,-1), 4, BLUE2),
            ("ROUNDEDCORNERS", [3,3,3,3])])), Spacer(1, 5),
        Table([[P(f"<b>Assessment ({r['sev']}):</b> {r['sev_r']}", 9, SLATE, ld=14)]], colWidths=[W],
        style=TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT), ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6), ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10), ("LINEBEFORE", (0,0), (0,-1), 4, sev_color),
            ("ROUNDEDCORNERS", [3,3,3,3])])), Spacer(1, 12)]
    for e in sec("02", "KEY FINDINGS"):story.append(e)
    story.append(Spacer(1, 4))
    for i, f in enumerate(r["findings"]):
        nb = Table([[P(str(i+1), 7, WHITE, True, TA_CENTER)]], colWidths=[15], rowHeights=[15],
            style=TableStyle([("BACKGROUND", (0,0), (-1,-1), BLUE2), ("TOPPADDING", (0,0), (-1,-1), 1),
                ("BOTTOMPADDING", (0,0), (-1,-1), 1), ("ROUNDEDCORNERS", [7,7,7,7])]))
        story.append(Table([[nb, P(f, 9, MID, ld=14)]], colWidths=[20, W-20],
            style=TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3), ("LINEBELOW", (0,0), (-1,-1), 0.4, BORD)])))
    story.append(Spacer(1, 12))
    for e in sec("03", "PATTERN ANALYSIS"):story.append(e)
    story.append(Spacer(1, 4))
    for p in r["patterns"]:
        dot = Table([[""]], colWidths=[7], rowHeights=[7], style=TableStyle([("BACKGROUND", (0,0), (-1,-1), AMB2), ("ROUNDEDCORNERS", [2,2,2,2])]))
        story.append(Table([[dot, P(p, 9, MID, ld=14)]], colWidths=[13, W-13],
            style=TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("TOPPADDING", (0,0), (-1,-1), 3),
                ("BOTTOMPADDING", (0,0), (-1,-1), 3), ("LINEBELOW", (0,0), (-1,-1), 0.4, BORD)])))
    story.append(Spacer(1, 12))
    for e in sec("04", "SCORE STATISTICS"):story.append(e)
    scw = W/4-3
    story += [Spacer(1, 6), Table([[[P("MIN", 7, MUTED, True, TA_CENTER), P(f"{r['smin']:.4f}", 13, RED2, True, TA_CENTER)],
        [P("MAX", 7, MUTED, True, TA_CENTER), P(f"{r['smax']:.4f}", 13, GRN2, True, TA_CENTER)],
        [P("MEAN", 7, MUTED, True, TA_CENTER), P(f"{r['smean']:.4f}", 13, BLU2, True, TA_CENTER)],
        [P("STD", 7, MUTED, True, TA_CENTER), P(f"{r['sstd']:.4f}", 13, AMB2, True, TA_CENTER)]]],
        colWidths=[scw]*4, style=TableStyle([("BACKGROUND", (0,0), (-1,-1), LIGHT),
            ("BOX", (0,0), (0,0), 0.4, HBORD), ("BOX", (1,0), (1,0), 0.4, HBORD),
            ("BOX", (2,0), (2,0), 0.4, HBORD), ("BOX", (3,0), (3,0), 0.4, HBORD),
            ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 9), ("BOTTOMPADDING", (0,0), (-1,-1), 9)])), Spacer(1, 12)]
    # Diagrams
    story.append(PageBreak())
    for e in sec("05", "DIAGRAMS"):story.append(e)
    story.append(Spacer(1, 6))
    cw = W/2-4;ch = cw*0.52
    def cpair(k1, k2):
        def cell(k):
            if k in charts:return img(charts[k], cw, ch)
            return P("(unavailable)", 8, MUTED, a=TA_CENTER)
        return Table([[cell(k1), cell(k2)]], colWidths=[cw, cw],
            style=TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("BOTTOMPADDING", (0,0), (-1,-1), 6)]))
    story += [cpair("score", "top15"), cpair("pca", "event")]
    # Two-pass chart full width
    if "twopass" in charts:
        story += [Spacer(1, 4), img(charts["twopass"], W, W * 0.45)]
    story += [Spacer(1, 10)]
    # Table
    for e in sec("06", "TOP 10 RECORDS"):story.append(e)
    story += [P("Sorted by score. Includes z-score and confidence.", 8, MUTED), Spacer(1, 6)]
    tdf = r["top10"].reset_index(drop=True);cols = list(tdf.columns);cw2 = W / len(cols)
    td = [[P(c.upper().replace("_", " "), 7, WHITE, True, TA_CENTER) for c in cols]]
    for _, row in tdf.iterrows():
        cells = [P(str(row[c])[:18] if pd.notna(row[c]) else "—", 8,
               RED2 if c == "anomaly_score" and not r["is_clean"] and pd.notna(row[c]) and float(row[c]) < -0.2 else MID) for c in cols]
        td.append(cells)
    story.append(Table(td, colWidths=[cw2]*len(cols), style=TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT, WHITE]),
        ("GRID", (0,0), (-1,-1), 0.3, HBORD), ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), ("ROUNDEDCORNERS", [4,4,4,4])])))
    story += [Spacer(1, 16), HR(NAVY, 1.5, 4, 4), Table([[P("FoxyDucky Task 3 Noctra Lupra · Two-Pass Isolation Forest", 7, MUTED),
        P(f"{r['now'].strftime('%Y-%m-%d')} · CONFIDENTIAL", 7, MUTED, a=TA_RIGHT)]], colWidths=[W*0.6, W*0.4])]
    doc.build(story);buf.seek(0);return buf.read()

# ═══════════════════════ SIDEBAR ═════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Anomaly Detection")
    st.markdown("<p style='color:#6b7a99;font-size:.82rem;margin-top:-6px'>Isolation Forest · Two-Pass · Pure NumPy</p>", unsafe_allow_html=True)
    st.markdown("---")
    uploaded = st.file_uploader("Upload JSONL / JSON file", type=["json", "jsonl"])

    st.markdown("#### Model Parameters")
    cont_rate = st.slider("Contamination rate", min_value=0.01, max_value=0.50, value=0.05, step=0.01,
                          help="Pass 1: Isolation Forest candidate threshold.")
    z_threshold = st.slider("Z-score threshold", min_value=1.0, max_value=4.0, value=2.0, step=0.1,
                            help="Pass 2: Only candidates with z-score above this are confirmed. Higher = fewer false positives.")

    st.markdown("#### Feature Engineering")
    feat_eng = st.toggle("Enable Suricata features", value=True,
                         help="Add computed features: packet/byte ratios, connection counts, unique dest IPs/ports.")

    st.markdown("---")
    st.caption(f"Config: cont={cont_rate:.0%}, z={z_threshold:.1f}σ, feat={'ON' if feat_eng else 'OFF'}")
    st.markdown("---")
    run_btn = st.button("▶  Run Detection", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("<p style='color:#3a4a6a;font-size:.72rem'>FoxyDucky · Task 3 Noctra Lupra</p>", unsafe_allow_html=True)

# ═══════════════════════ WELCOME ═════════════════════════════════════════════
if uploaded is None:
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown("""<div style='text-align:center;padding:4rem 2rem;background:#111827;
            border:1px solid #1e2740;border-radius:16px;margin-top:3rem'>
            <div style='font-size:3.5rem;margin-bottom:1rem'>🔍</div>
            <h3 style='color:#e2e8f0'>Upload a file to begin</h3>
            <p style='color:#6b7a99;font-size:.9rem;line-height:1.7'>
            Select a JSONL file, configure parameters, then click
            <strong style='color:#3b82f6'>▶ Run Detection</strong>.</p></div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════ RUN ═════════════════════════════════════════════════
if run_btn:
    with st.spinner("Running two-pass pipeline…"):
        df, X, err, steps = run_analysis(uploaded.getvalue(), cont_rate, z_threshold, feat_eng)
    if err:st.error(f"**Error:** {err}");st.stop()
    st.session_state.update({"df": df, "X": X, "cont": cont_rate, "z_thr": z_threshold, "steps": steps})
    with st.spinner("Building report & PDF…"):
        try:
            rpt = gen_report(df, X, cont_rate, z_threshold, steps)
            ch = make_pdf_charts(df, X)
            st.session_state["report"] = rpt
            st.session_state["rpt_charts"] = ch
            try:st.session_state["pdf"] = build_pdf(rpt, ch)
            except ImportError:st.session_state["pdf"] = None
            except Exception:st.session_state["pdf"] = None
        except Exception as e:
            st.warning(f"Report error: {e}")
            st.session_state["report"] = None;st.session_state["pdf"] = None;st.session_state["rpt_charts"] = {}

if "df" not in st.session_state:
    st.info("Upload a file and click **▶ Run Detection** to start.");st.stop()

df = st.session_state["df"];X = st.session_state["X"];cont = st.session_state["cont"]
z_thr = st.session_state.get("z_thr", 2.0);steps = st.session_state.get("steps", {})
n_a = int((df["predicted_label"] == 1).sum());n_n = int((df["predicted_label"] == 0).sum())
tot = len(df);rat = n_a / tot if tot else 0;is_clean = (n_a == 0)
p1 = int(df["if_candidate"].sum());elim = p1 - n_a

# ═══════════════════════ BANNER ══════════════════════════════════════════════
if is_clean:bg_c = "#0f2e1e";tx_c = "#22c55e";badge = "✅ Clean"
elif rat > 0.1:bg_c = "#2a0f0f";tx_c = "#ef4444";badge = f"{rat:.1%} anomaly rate"
else:bg_c = "#0f2e1e";tx_c = "#22c55e";badge = f"{rat:.1%} anomaly rate"

st.markdown(f"""
<div style='background:#111827;border:1px solid #1e2740;border-radius:14px;
            padding:1rem 1.5rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:14px'>
    <span style='font-size:1.5rem'>{'✅' if is_clean else '🚨'}</span>
    <div>
        <p style='margin:0;font-size:1.05rem;font-weight:700;color:#e2e8f0'>Analysis complete — two-pass pipeline</p>
        <p style='margin:0;font-size:.82rem;color:#6b7a99'>{tot:,} records · cont={cont:.0%} · z={z_thr:.1f}σ</p>
    </div>
    <div style='margin-left:auto;background:{bg_c};border-radius:8px;
                padding:.3rem .9rem;color:{tx_c};font-weight:700;font-size:.88rem'>{badge}</div>
</div>""", unsafe_allow_html=True)

if is_clean:
    if p1 > 0:
        st.success(f"✅ **No anomalies confirmed.** IF flagged {p1:,} candidates but all {p1:,} were eliminated by z-score validation (z>{z_thr:.1f}σ).")
    else:
        st.success("✅ **No anomalies detected.** All traffic is normal — network appears healthy.")
elif elim > 0:
    st.info(f"🔬 **Two-pass filtering:** IF flagged {p1:,} candidates → z-score confirmed {n_a:,} → **{elim:,} false positives eliminated** ({elim/max(p1,1):.0%} reduction)")

# Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total records", f"{tot:,}")
c2.metric("Confirmed anomalies", f"{n_a:,}", delta=f"{rat:.1%}" if not is_clean else "0", delta_color="inverse" if not is_clean else "off")
c3.metric("FP eliminated", f"{elim:,}", delta=f"of {p1:,} candidates" if p1 > 0 else "none", delta_color="off")
c4.metric("Normal 🟢", f"{n_n:,}", delta=f"{1-rat:.1%}")

# Pipeline steps expander
with st.expander("⚙️ Pipeline details", expanded=False):
    for k, v in steps.items():
        st.markdown(f"**{k}:** {v}")

st.markdown("---")

# ═══════════════════════ CHARTS ══════════════════════════════════════════════
st.markdown("<div class='s-card'><div class='s-card-title'>📊 Visualisations</div>", unsafe_allow_html=True)

# Row 1: Score + Top 15
r1l, r1r = st.columns(2)
with r1l:
    try:f = _mk_score(df, True);st.pyplot(f);plt.close(f)
    except Exception as e:st.warning(f"Score chart: {e}")
with r1r:
    try:f = _mk_top15(df, True);st.pyplot(f);plt.close(f)
    except Exception as e:st.warning(f"Top 15: {e}")

# Row 2: PCA + Event
r2l, r2r = st.columns(2)
with r2l:
    try:
        f = _mk_pca(df, X, True)
        if f:st.pyplot(f);plt.close(f)
        else:st.info("PCA unavailable — insufficient feature variance.")
    except Exception as e:st.warning(f"PCA: {e}")
with r2r:
    try:
        f = _mk_event(df, True)
        if f:st.pyplot(f);plt.close(f)
        else:st.info("No `event_type` column.")
    except Exception as e:st.warning(f"Event chart: {e}")

# Row 3: Two-pass comparison (full width)
try:
    f = _mk_twopass(df, True);st.pyplot(f);plt.close(f)
except Exception as e:
    st.warning(f"Two-pass chart: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════ TOP 20 TABLE ════════════════════════════════════════
try:
    st.markdown("<div class='s-card'><div class='s-card-title'>📋 Top 20 records by score</div>", unsafe_allow_html=True)
    dcols = [c for c in ["timestamp", "event_type", "src_ip", "dest_ip", "proto",
                          "anomaly_score", "z_score", "confidence", "if_candidate", "predicted_label"] if c in df.columns]
    tbl = df.sort_values("anomaly_score").head(20)[dcols].copy()
    if "predicted_label" in tbl.columns:
        tbl["predicted_label"] = tbl["predicted_label"].map({1: "🔴 Anomaly", 0: "🟢 Normal"})
    if "if_candidate" in tbl.columns:
        tbl["if_candidate"] = tbl["if_candidate"].map({1: "⚠️ Yes", 0: "—"})
    for c in ["anomaly_score", "z_score", "confidence"]:
        if c in tbl.columns:tbl[c] = tbl[c].round(4)
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
except Exception as e:
    st.warning(f"Table error: {e}")

# ═══════════════════════ REPORT (AUTO-SHOWN) ═════════════════════════════════
st.markdown("---")
st.markdown("<div class='s-card'><div class='s-card-title'>📄 Incident Report</div>", unsafe_allow_html=True)

rpt = st.session_state.get("report")
rpt_charts = st.session_state.get("rpt_charts", {})

if not rpt:
    st.info("Click **▶ Run Detection** to generate the full report.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

rc1, rc2 = st.columns([1, 5])
with rc1:
    if st.button("🔄 Regenerate", key="regen", use_container_width=True):
        with st.spinner("Regenerating…"):
            try:
                rpt = gen_report(df, X, cont, z_thr, steps)
                rpt_charts = make_pdf_charts(df, X)
                st.session_state["report"] = rpt;st.session_state["rpt_charts"] = rpt_charts
                try:st.session_state["pdf"] = build_pdf(rpt, rpt_charts)
                except:st.session_state["pdf"] = None
            except Exception as e:st.error(f"Error: {e}")
        st.rerun()
with rc2:
    st.caption("Report auto-generated after detection. Click Regenerate to refresh.")

sev = rpt["sev"]
icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🔵", "LOW": "🟢", "CLEAN": "✅"}

st.markdown(f"""
<div class="rpt-wrap">
  <div class="rpt-hdr">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
      <div><p class="rpt-title">INCIDENT REPORT</p>
        <p class="rpt-sub">Two-Pass Anomaly Detection — Isolation Forest + Z-Score Confirmation</p></div>
      <span class="rpt-badge sev-{sev}">{icons.get(sev, '')} {sev}</span>
    </div>
    <div class="rpt-meta">
      <div class="rpt-mi">Generated: <span>{rpt['now'].strftime('%d %B %Y %H:%M:%S')}</span></div>
      <div class="rpt-mi">Contamination: <span>{rpt['cont']:.0%}</span></div>
      <div class="rpt-mi">Z-threshold: <span>{rpt['z_thr']:.1f}σ</span></div>
    </div>
  </div>
  <div class="stat-row">
    <div class="stat-box"><div class="stat-top" style="background:#3b82f6"></div>
      <div class="stat-lbl">Total Records</div><div class="stat-val">{rpt['n_tot']:,}</div>
      <div class="stat-sub">Processed</div></div>
    <div class="stat-box"><div class="stat-top" style="background:{'#22c55e' if rpt['is_clean'] else '#ef4444'}"></div>
      <div class="stat-lbl">Confirmed</div><div class="stat-val">{rpt['n_a']:,}</div>
      <div class="stat-sub">{rpt['ratio']:.2%} of total</div></div>
    <div class="stat-box"><div class="stat-top" style="background:#f59e0b"></div>
      <div class="stat-lbl">FP Eliminated</div><div class="stat-val">{rpt['elim']:,}</div>
      <div class="stat-sub">of {rpt['p1_count']:,} candidates</div></div>
    <div class="stat-box"><div class="stat-top" style="background:{rpt['sev_col']}"></div>
      <div class="stat-lbl">Status</div><div class="stat-val">{sev}</div>
      <div class="stat-sub">Assessment</div></div>
  </div>
  <div class="sec-hdr"><div class="sec-num">01</div><div class="sec-title">Executive Summary</div></div>
  <div class="rpt-sum">{rpt['summary']}</div>
  <div class="sev-box"><div class="sev-text"><b>Assessment ({sev}):</b> {rpt['sev_r']}</div></div>
  <div class="sec-hdr"><div class="sec-num">02</div><div class="sec-title">Key Findings</div></div>
  <ul class="find-list">{''.join(f'<li><div class="find-num">{i+1}</div><div class="find-text">{f}</div></li>' for i, f in enumerate(rpt['findings']))}</ul>
  <div class="sec-hdr"><div class="sec-num">03</div><div class="sec-title">Pattern Analysis</div></div>
  <ul class="pat-list">{''.join(f'<li><div class="pat-dot"></div><div class="pat-text">{p}</div></li>' for p in rpt['patterns'])}</ul>
  <div class="sec-hdr"><div class="sec-num">04</div><div class="sec-title">Score Statistics</div></div>
  <div class="sc-grid">
    <div class="sc-box"><div class="sc-lbl">Minimum</div><div class="sc-val" style="color:#ef4444">{rpt['smin']:.4f}</div></div>
    <div class="sc-box"><div class="sc-lbl">Maximum</div><div class="sc-val" style="color:#22c55e">{rpt['smax']:.4f}</div></div>
    <div class="sc-box"><div class="sc-lbl">Mean</div><div class="sc-val" style="color:#3b82f6">{rpt['smean']:.4f}</div></div>
    <div class="sc-box"><div class="sc-lbl">Std Dev</div><div class="sc-val" style="color:#f59e0b">{rpt['sstd']:.4f}</div></div>
  </div>
  <div class="sec-hdr"><div class="sec-num">05</div><div class="sec-title">Diagrams</div></div>
</div>""", unsafe_allow_html=True)

# Charts
ch1, ch2 = st.columns(2)
with ch1:
    if "score" in rpt_charts:st.image(rpt_charts["score"], use_container_width=True)
    else:st.info("Score chart unavailable.")
with ch2:
    if "top15" in rpt_charts:st.image(rpt_charts["top15"], use_container_width=True)
    else:st.info("Top 15 unavailable.")
ch3, ch4 = st.columns(2)
with ch3:
    if "pca" in rpt_charts:st.image(rpt_charts["pca"], use_container_width=True)
    else:st.info("PCA unavailable.")
with ch4:
    if "event" in rpt_charts:st.image(rpt_charts["event"], use_container_width=True)
    else:st.info("Event type unavailable.")
if "twopass" in rpt_charts:
    st.image(rpt_charts["twopass"], use_container_width=True)

st.markdown("<div class='sec-hdr'><div class='sec-num'>06</div><div class='sec-title'>Top 10 Records</div></div>", unsafe_allow_html=True)
st.dataframe(rpt["top10"], use_container_width=True, hide_index=True)

st.markdown(f"""<div class="rpt-footer"><p>FoxyDucky Task 3 Noctra Lupra · Two-Pass Isolation Forest</p>
  <p>Generated: {rpt['now'].strftime('%Y-%m-%d')} · CONFIDENTIAL</p></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if st.session_state.get("pdf"):
    st.download_button("⬇️ Download PDF", data=st.session_state["pdf"],
        file_name=f"incident_report_{rpt['now'].strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf", key="dl_pdf")
else:
    st.info("Install ReportLab: `pip install reportlab`")

st.markdown("</div>", unsafe_allow_html=True)