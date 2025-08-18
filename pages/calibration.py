# Streamlit page: General Court Calibration (Fixed Params, Model vs Case CSVs)
# ------------------------------------------------------------------------
# This page reads your **exported per-case CSVs** (no uploads), uses
# **completely fixed parameters** (hardcoded below — no UI), computes the
# model's predicted P(X) **per juror count M** via your OracleModel's
# expected utilities + QRE, converts that to the **expected majority share**
# E[max(K/M, 1-K/M)] with K~Bin(M, q), and compares against the General Court
# data by M. We show:
#   • General Court data summaries (by M)
#   • Model predictions (expected majority share by M)
#   • A per-M tally table (observed vs predicted) with per-M MAE
#   • Global MAE
#   • A calibration plot (predicted majority share vs observed)

from pathlib import Path
from typing import Dict, Iterable
import sys, os

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from scipy.stats import binom
from numpy import log
from scipy.special import gammaln

# Ensure parent directory (project root) is importable for local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to use the reusable loader from evaluation.py; fall back if missing
try:
    from evaluation import load_cases  # -> DataFrame: case_id, M, k_majority, k_minority, is_tie, x_share_majority
except Exception:
    load_cases = None

# Import your OracleModel for expected utilities
try:
    from model import OracleModel
except Exception as e:
    OracleModel = None

st.set_page_config(page_title="General Court Calibration (Fixed Params)", layout="wide")
st.markdown("# [General Court](https://klerosboard.com/1/courts/0) — Calibration & MAE (Fixed Parameters)")
st.caption("Clicking on General Court will take you to Kleros' general court data")

# -----------------------------
# 0) FIXED PARAMETERS (EDIT HERE)
# -----------------------------
# No UI. These values are constant for the page run.
FIXED = {
    "payoff_type": "redistributive",  # "basic" | "redistributive" | "symbiotic"
    "epsilon": 0.00,
    "p": 33.40,
    "d": 99.49,
    "noise": 0.00,          # set 0.0 so q depends purely on expected utilities (deterministic)
    "attack": False,
}

with st.sidebar:

    st.sidebar.markdown(r"$p+\varepsilon$ attacks are disabled in order to calibrate the model with the real-world data")

    st.header("Fixed parameters")
    log_lambda = st.sidebar.slider(r"log$_{10}$ QRE Sensitivity ($\lambda$)",
                               -3.0, # log10(0.001)
                               0.5, # log10(10)
                               value=-1.73, 
                               step=0.01,
                               help=r"Higher $\lambda$ means jurors are more sensitive to payoff differences (closer to rational). Lower values add noise and irrationality.")
    st.sidebar.latex(rf"\lambda = {10**log_lambda:.3g}")
    x_mean     = st.slider(r"Expected Share of Votes for $X$ ($x$)", 0.5, 1.0, value=0.83, step=0.01,
                                  help=r"This sets the focal point for $x$: the juror's expected proportion of other jurors voting $\text{X}$ (the coherent vote).")
    
    st.sidebar.markdown(r"$p$ = 33.40")
    st.sidebar.markdown(r"$d$ = 99.49")
    st.sidebar.markdown("noise = 0.00")
    st.sidebar.markdown("Payoff Mechanism = Redistributive")

lambda_qre = 10 ** log_lambda

# Combine fixed params with interactive ones
PARAMS = {**FIXED, "lambda_qre": float(lambda_qre), "x_mean": float(x_mean)}

st.markdown(r"$\lambda$ and $x$ can be changed to see their effect on the MAE (mean absolute error)")

# -----------------------------
# 1) Resolve CSV folder
# -----------------------------
# -----------------------------
# 1) Resolve CSV folder
# -----------------------------
# -----------------------------
# 1) Resolve CSV folder
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_DIR = PROJECT_ROOT / "Kleros general court json data" / "CVS general court results"

@st.cache_data(show_spinner=False)
def _resolve_csv_dir(p_str: str) -> Path:
    p = Path(p_str).expanduser()
    if p.exists():
        return p
    p2 = (PROJECT_ROOT / p_str).resolve()
    if p2.exists():
        return p2
    p3 = (PROJECT_ROOT / "Kleros general court json data" / p_str).resolve()
    if p3.exists():
        return p3
    last = Path(p_str).name
    for candidate in PROJECT_ROOT.rglob(last):
        if candidate.is_dir() and any(candidate.glob("case_*.csv")):
            return candidate
    return p  # caller will validate

# Always use the default (no UI)
csv_dir = _resolve_csv_dir(str(DEFAULT_CSV_DIR))

# Validate
if not csv_dir.exists():
    st.error("Folder not found.")
    st.stop()

case_files = sorted(csv_dir.glob("case_*.csv"))
if not case_files:
    st.error("No case_*.csv files found in this folder. Double-check the path.")
    st.stop()

# -----------------------------
# 2) Load cases (prefer evaluation.load_cases, fallback to direct parse)
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_cases(folder: Path) -> pd.DataFrame:
    if load_cases is not None:
        try:
            df = load_cases(folder)
            if isinstance(df, pd.DataFrame) and not df.empty and "M" in df.columns:
                return df
        except Exception:
            pass
    # Fallback: parse per-case CSVs directly
    rows = []
    for p in sorted(folder.glob("case_*.csv")):
        try:
            df = pd.read_csv(p, index_col=0)
            vX = int(df.loc["X", "Vote Count"]) if "X" in df.index else 0
            vY = int(df.loc["Y", "Vote Count"]) if "Y" in df.index else 0
            M  = int(df.loc["X", "Total Jurors"]) if "X" in df.index else (vX + vY)
            if M <= 0:
                continue
            k_majority = max(vX, vY)
            k_minority = min(vX, vY)
            is_tie = (vX == vY)
            
            rows.append({
                "case_id": p.stem.replace("case_", ""),
                "file": p.name,
                "M": M,
                "k_majority": k_majority,
                "k_minority": k_minority,
                "is_tie": is_tie,
                "x_share_majority": k_majority / M,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

cases = _load_cases(csv_dir)
if cases.empty or "M" not in cases.columns:
    st.error("No usable cases found or missing 'M' in parsed CSVs.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    st.caption(f"Loaded {len(cases)} cases across {cases['M'].nunique()} unique juror counts.")
with right:
    st.metric("Median jurors (M)", int(cases["M"].median()))

with st.expander("General Court — sample (first 20)", expanded=False):
    display_cols = {
        "case_id": "case id",
        "file": "file",
        "M": "number of jurors (M)",
        "k_majority": "number of X votes",
        "k_minority": "number of Y votes",
        "is_tie": "tie round",
        "x_share_majority": "ratio of X votes",
    }
    st.dataframe(
        cases.rename(columns=display_cols).head(20),
        use_container_width=True
    )

# Distribution of cases by juror count M
st.subheader("General Court — cases by juror count (M)")
byM_counts = cases.groupby("M", as_index=False).size().rename(columns={"size": "n_cases"})
bar_cases = alt.Chart(byM_counts).mark_bar().encode(
    x=alt.X("M:O", title="Jurors (M)"),
    y=alt.Y("n_cases:Q", title="# Cases")
).properties(height=260)
st.altair_chart(bar_cases, use_container_width=True)

st.subheader("Error metrics")

# Overall (case-level) MAE — X/Y-agnostic
st.markdown("**MAE (mean absolute error)** = the average per-case difference between the model's vote probability and the observed winner's share")
st.latex(r"""
\text{MAE} \;=\; \frac{1}{N}\sum_{i=1}^{N}
\min\!\Big(\,|q_i - s_i|,\; |q_i - (1 - s_i)|\,\Big)
""")

# Per-M MAE
st.markdown("**MAE by juror count (M)** = the average error **only over cases that used M jurors**.")
st.latex(r"""
\text{MAE}(\text{M}) \;=\; \frac{1}{\text{N}_\text{M}}\sum_{i:\,\text{M}_i=\text{M}}
\min\!\Big(\,|q_i - s_i|,\; |q_i - (1 - s_i)|\,\Big)
""")

st.markdown(r"$\text{N}_\text{M}$ = number of cases with exactly $\text{M}$ jurors")

st.markdown(
    "- $q_i$: model's **juror vote probability** for case $i$\n"
    "- $s_i$: **observed winner's share** (majority fraction) for case $i$\n"
    "- Smaller error is taken because the labels X/Y are arbitrary"
)

st.markdown("**Negative Log-Likelihood (NLL)** — the average negative log of the model-assigned probability of the **observed** vote counts")
st.latex(r"""
\text{NLL} \;=\; -\frac{1}{N}\sum_{i=1}^{N}
\log\!\Big[
\binom{n_i}{m_i} q_i^{\,m_i} (1-q_i)^{\,n_i-m_i}
\;+\;
\binom{n_i}{n_i-m_i} q_i^{\,n_i-m_i} (1-q_i)^{\,m_i}
\Big]
""")

st.markdown(
    "- $n_i$: **observed** panel size (number of jurors) in case $i$  \n"
    "- $m_i$: **observed** number of votes for the winning side (majority count) in case $i$  \n"
    "- $q_i$: model's **juror vote probability** for case $i$\n"
    "- There are two binomial terms so swapping between X and Y votes doesn't change the score  \n"
)

st.markdown(
        "**Lower is better**: each case contributes `-log(probability_that_the_model_assigns_to_the_observed_counts)`. \n"
    "If the model places **high probability** on what actually happened, NLL is small; "
    "if it places **low probability**, NLL is large. \n" 
    "A perfect model would have **NLL = 0**."
)

# ---------------------------------------------------
# 3) Deterministic P(X | M) from OracleModel expected utilities + QRE
# ---------------------------------------------------

def qre_prob_X_for_M(M: int, params: Dict) -> float:
    """Compute the juror's QRE probability of voting X for a panel of size M.
    Uses OracleModel._expected_payoffs (expected utilities) and applies the
    logit QRE choice with λ. We set noise=0.0 in this step to keep it deterministic
    (noise is a separate parameter in FIXED but does not enter this expectation).
    """
    if OracleModel is None:
        # Fallback: use x_mean if model not importable
        return float(np.clip(params.get("x_mean", 0.72), 1e-9, 1 - 1e-9))

    model = OracleModel(
        num_jurors=int(M),
        noise=0.0,  # deterministic expected utilities for QRE
        lambda_qre=float(params["lambda_qre"]),
        p=float(params["p"]),
        d=float(params["d"]),
        epsilon=float(params["epsilon"]),
        payoff_type=str(params["payoff_type"]),
        attack=bool(params["attack"]),
        x_mean=float(params["x_mean"]),
    )
    # Any juror index is equivalent by symmetry here
    uX, uY = model._expected_payoffs(0)
    lam = float(params["lambda_qre"]) if float(params["lambda_qre"]) >= 0 else 0.0
    # Softmax / logit QRE
    try:
        eX = np.exp(lam * uX)
        eY = np.exp(lam * uY)
        q = float(eX / (eX + eY))
    except OverflowError:
        q = 1.0 if (lam * uX) > (lam * uY) else 0.0
    return float(np.clip(q, 1e-9, 1 - 1e-9))

@st.cache_data(show_spinner=False)
def compute_q_map(M_values: Iterable[int], lambda_qre: float, x_mean: float, other: Dict) -> Dict[int, float]:
    params = {**other, "lambda_qre": float(lambda_qre), "x_mean": float(x_mean)}
    return {int(M): qre_prob_X_for_M(int(M), params) for M in sorted(set(int(m) for m in M_values))}

unique_M = sorted(cases["M"].unique())
q_map = compute_q_map(unique_M, lambda_qre, x_mean, FIXED)

# --- Label-invariant q-vs-share metrics (to match parameter_estimation) ---
cases["q_pred"] = cases["M"].map(q_map).astype(float)           # juror vote probability
cases["s_obs"]  = cases["k_majority"] / cases["M"]              # observed share for 'X' as exported

def two_sided_binom_nll(m: np.ndarray, n: np.ndarray, q: np.ndarray) -> float:
    q = np.clip(q, 1e-12, 1 - 1e-12)
    m = m.astype(float); n = n.astype(float)
    logpmf1 = (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)
               + m*log(q) + (n - m)*log(1 - q))
    m2 = n - m
    logpmf2 = (gammaln(n + 1) - gammaln(m2 + 1) - gammaln(n - m2 + 1)
               + m2*log(q) + (n - m2)*log(1 - q))
    # logsumexp over the two labelings
    loglik = np.logaddexp(logpmf1, logpmf2)
    return float(-np.mean(loglik))

n = cases["M"].to_numpy()
m = cases["k_majority"].to_numpy()    # works because we use two-sided form
q = cases["q_pred"].to_numpy()
nll_sym = two_sided_binom_nll(m, n, q)

# label-invariant MAE: distance to the closer of s or 1-s
cases["mae_sym_q"] = np.minimum(
    np.abs(cases["q_pred"] - cases["s_obs"]),
    np.abs(cases["q_pred"] - (1.0 - cases["s_obs"]))
)

mae_sym_q_overall = cases["mae_sym_q"].mean()

# ---------------------------------------------------
# 4) Predicted majority share per case: E[max(K/M, 1-K/M)] with K~Bin(M, q)
# ---------------------------------------------------

def expected_majority_share(M: int, q: float) -> float:
    k = np.arange(0, M + 1)
    pmf = binom.pmf(k, M, q)
    maj_share = np.maximum(k, M - k) / M
    return float(np.dot(pmf, maj_share))

@st.cache_data(show_spinner=False)
def compute_pred_majority_share_map(M_values: Iterable[int], qmap: Dict[int, float]) -> Dict[int, float]:
    out = {}
    for M in sorted(set(int(m) for m in M_values)):
        q = float(qmap[int(M)])
        out[int(M)] = expected_majority_share(int(M), q)
    return out

pred_share_map = compute_pred_majority_share_map(unique_M, q_map)

# Attach predictions to every case (by M)
cases["pred_majority_share"] = cases["M"].map(pred_share_map).astype(float)
# Observed majority share already in cases['x_share_majority']
cases["abs_err"] = (cases["pred_majority_share"] - cases["x_share_majority"]).abs()

st.subheader("Per-M comparison: observed vs predicted (tally)")
perM = (cases
    .groupby("M", as_index=False)
    .agg(
        n_cases=("case_id", "count"),
        obs_share_mean=("x_share_majority", "mean"),
        pred_share=("pred_majority_share", "mean"),
        mae_share=("abs_err", "mean"),
    )
)

perM["line_gap"] = (perM["obs_share_mean"] - perM["pred_share"]).abs()

# Line-gap MAE (macro = unweighted over M) + weighted by number of cases
line_gap_macro = perM["line_gap"].mean()
line_gap_weighted = (perM["line_gap"] * perM["n_cases"]).sum() / perM["n_cases"].sum()

# ------------------------------------
# 5) Global MAE + per-M comparison (tally)
# ------------------------------------
colA, colB, colC, colD, colE = st.columns(5)
with colA:
   st.metric("MAE (vote prob vs observed share)", f"{mae_sym_q_overall:.3f}")  # matches parameter_estimation.py
with colB:
    st.metric("MAE per number of jurors (M)", f"{line_gap_macro:.3f}")          # when the two lines diverge
with colC:
    st.metric("NLL", f"{nll_sym:.3f}")
with colD:
    st.metric("Cases", f"{len(cases)}")
with colE:
    st.metric("unique M", f"{cases['M'].nunique()}")

# contributions to the global (per-case) MAEs

perM[["mae_share","line_gap"]] = perM[["mae_share","line_gap"]].round(3)
perM = perM.sort_values("M")
st.dataframe(
    perM[["M", "n_cases", "obs_share_mean", "pred_share", "line_gap"]]
        .sort_values("M"),
    use_container_width=True
)
# Dual lines per M: observed vs predicted
lines_df = perM.melt(
    id_vars=["M"],                              # <- drop n_cases / mae_share / line_gap
    value_vars=["obs_share_mean", "pred_share"],
    var_name="series",
    value_name="value",
)

line_chart = alt.Chart(lines_df).mark_line(point=True).encode(
    x=alt.X(
        "M:Q",
        title="Jurors (M)",
        axis=alt.Axis(labelAngle=0, tickMinStep=1)
    ),
    y=alt.Y(
        "value:Q",
        title="Majority share",
        scale=alt.Scale(domain=[0.5, 1.0], nice=False)
    ),
    color=alt.Color(
        "series:N",
        title="Series",
        scale=alt.Scale(domain=["obs_share_mean", "pred_share"],
                        range=["#1f77b4", "#ff7f0e"])
    ),
    tooltip=[
        alt.Tooltip("M:Q", title="M"),
        alt.Tooltip("series:N", title="Series"),
        alt.Tooltip("value:Q", title="Majority share", format=".3f"),
    ],
).properties(height=300)

st.latex(r"\Large \textbf{Observed vs.\ Predicted Majority Share by Juror Count }\,M")

st.altair_chart(line_chart, use_container_width=True)

# --------------------------------------------------------
# 6) Calibration: predicted majority share vs observed majority share
# --------------------------------------------------------
st.latex(r"{\Large \textbf{Calibration: Predicted vs.\ Observed Majority Share}}")

st.markdown("Each bubble aggregates cases into bins by the model's **predicted** majority share (x-axis).")

st.markdown("The bubble's y-value is the **observed** mean majority share in that bin, "
    "and the bubble size is the **number of cases** in the bin.")
    
st.markdown("Points **on** the $y=x$ line mean perfect calibration. Points **above** the line mean "
    "the model **underpredicts** majority strength (underfitting); points **below** the line mean it "
    "**overpredicts** (overfitting).")

@st.cache_data(show_spinner=False)
def calibration_bins(df: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    # Bin by PREDICTED share in [0.5, 1.0]
    preds = df["pred_majority_share"].to_numpy()
    preds = np.clip(preds, 0.5, 1.0)
    obs   = df["x_share_majority"].to_numpy()

    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx  = np.digitize(preds, bins) - 1

    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            rows.append({"bin": b, "p_mean": np.nan, "y_mean": np.nan, "count": 0})
            continue
        rows.append({
            "bin": b,
            "p_mean": float(np.nanmean(preds[mask])),  # predicted mean (X-axis)
            "y_mean": float(np.nanmean(obs[mask])),    # observed mean (Y-axis)
            "count": int(np.sum(mask)),
        })
    return pd.DataFrame(rows)

bin_df = calibration_bins(cases, n_bins=20).dropna()

# Identity line (perfect calibration) WITH axis titles on the first layer
perfect = pd.DataFrame(
    {"x": [0.5, 1.0], "y": [0.5, 1.0], "series": ["Perfect calibration (y = x)"] * 2}
)

line = alt.Chart(perfect).mark_line().encode(
    x=alt.X("x:Q",
            title="Predicted majority share (model)",
            scale=alt.Scale(domain=[0.5, 1.0], nice=False)),
    y=alt.Y("y:Q",
            title="Observed majority share (court)",
            scale=alt.Scale(domain=[0.5, 1.0], nice=False)),
    color=alt.Color("series:N", title="Legend")
)

# Binned averages
pts = alt.Chart(bin_df.assign(series="Binned averages")).mark_point().encode(
    x=alt.X("p_mean:Q",
            title="Predicted majority share (model)",
            scale=alt.Scale(domain=[0.5, 1.0], nice=False)),
    y=alt.Y("y_mean:Q",
            title="Observed majority share (court)",
            scale=alt.Scale(domain=[0.5, 1.0], nice=False)),
    size=alt.Size("count:Q", legend=None),
    color=alt.Color("series:N", title="Legend"),
    tooltip=[
        alt.Tooltip("p_mean:Q", title="Predicted mean", format=".3f"),
        alt.Tooltip("y_mean:Q", title="Observed mean",  format=".3f"),
        alt.Tooltip("count:Q",  title="# cases"),
    ],
).properties(height=380)

st.altair_chart(line + pts, use_container_width=True)