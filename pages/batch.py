# page: Batch Processing for further Analysis

# pages/batch.py

"""
this page represents the batch processing

the user can select parameters to fix and free parameters with a range and grid size
"""

import streamlit as st
import numpy as np
import pandas as pd
import itertools
from stqdm import stqdm
from multiprocessing import cpu_count
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# for large file uploads
# now file is no longer directly uploaded into RAM
# much faster data processing

from collections import defaultdict

ALT_USECOLS = [
    "Number of Jurors", "num_jurors",    # one of these will exist
    "lambda_qre", "x_mean", "avg_qre_prob_X",
    "Majority", "attack", "epsilon", "Tie"
]
DTYPES = {
    "num_jurors": "int16",
    "lambda_qre": "float32",
    "x_mean": "float32",
    "avg_qre_prob_X": "float32",
    "Majority": "category",
    "attack": "boolean",
    "epsilon": "float32",
    "Tie":"boolean",
}

# this file is in a folder called pages for streamlit to detect it as a new page
# so functions need to have the file path written

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from batch_runner import run_batch_parallel  

@st.cache_data(show_spinner=True)
def stream_aggregates(_file, chunksize=1_000_000):
    """
    Single-pass, low-memory aggregation for very large batch CSVs.
    Returns the exact tables your plots already expect.
    """
    wanted = set(ALT_USECOLS)

    # Running tallies keyed by (M, λ, x))
    tot = defaultdict(lambda: {"n":0, "xwins":0, "ties":0, "sum_q":0.0, "qcnt":0})
    # For ε plots: keyed by (M, λ, x, ε)
    eps_tot = defaultdict(lambda: {"n":0, "xwins":0, "sum_q":0.0, "qcnt":0})

    # Reader (try fast pyarrow engine; fall back to pandas)
    try:
        reader = pd.read_csv(_file, engine="pyarrow",
                             usecols=lambda c: c in wanted,
                             chunksize=chunksize)
    except Exception:
        reader = pd.read_csv(_file,
                             usecols=lambda c: c in wanted,
                             chunksize=chunksize)

    for chunk in reader:
        # Normalize juror column name
        if "num_jurors" not in chunk.columns and "Number of Jurors" in chunk.columns:
            chunk = chunk.rename(columns={"Number of Jurors": "num_jurors"})

        # Light dtypes
        for col, dt in DTYPES.items():
            if col in chunk.columns:
                try:
                    chunk[col] = chunk[col].astype(dt)
                except Exception:
                    pass

        chunk["is_X"]   = (chunk["Majority"] == "X").astype("int8")

        if "Tie" in chunk.columns:
            chunk["is_Tie"] = chunk["Tie"].astype("int8")
        else:
            chunk["is_Tie"] = (chunk["Majority"] == "Tie").astype("int8")

        # ---- Aggregate (M, λ, x)
        gb = chunk.groupby(["num_jurors", "lambda_qre", "x_mean"], observed=True)
        cnt   = gb["is_X"].count()
        xsum  = gb["is_X"].sum()
        ties  = gb["is_Tie"].sum()
        qsum  = gb["avg_qre_prob_X"].sum()
        qcnt = gb["avg_qre_prob_X"].count()

        for (M, lam, x), n in cnt.items():
            key = (int(M), float(lam), float(x))
            t = tot[key]
            t["n"]     += int(n)
            t["xwins"] += int(xsum.loc[(M, lam, x)])
            t["ties"]  += int(ties.loc[(M, lam, x)])
            t["sum_q"] += float(qsum.loc[(M, lam, x)])
            t["qcnt"]  += int(qcnt.loc[(M, lam, x)])

        # ---- ε aggregates (only attack==True if attack column exists)
        if "epsilon" in chunk.columns:
            sub = chunk
            if "attack" in chunk.columns:
                atk_mask = chunk["attack"].astype(str).str.lower().isin(["1","true","t","yes","y"])
                sub = chunk[atk_mask]

            if not sub.empty:
                gb2 = sub.groupby(["num_jurors", "lambda_qre", "x_mean", "epsilon"], observed=True)
                cnt2  = gb2["is_X"].count()
                xsum2 = gb2["is_X"].sum()
                qsum2 = gb2["avg_qre_prob_X"].sum()
                qcnt2 = gb2["avg_qre_prob_X"].count()
                for (M, lam, x, eps), n in cnt2.items():
                    key = (int(M), float(lam), float(x), float(eps))
                    s = eps_tot[key]
                    s["n"]     += int(n)
                    s["xwins"] += int(xsum2.loc[(M, lam, x, eps)])
                    s["sum_q"] += float(qsum2.loc[(M, lam, x, eps)])
                    s["qcnt"]  += int(qcnt2.loc[(M, lam, x, eps)])

    # ---- Build DataFrames for your charts
    grid_win = pd.DataFrame(
        [(M, lam, x, d["xwins"]/d["n"]) for (M, lam, x), d in tot.items()],
        columns=["num_jurors","lambda_qre","x_mean","X_win_rate"],
    )
    
    grid3 = pd.DataFrame(
        [(M, lam, x, (d["sum_q"]/d["qcnt"] if d["qcnt"] else np.nan))
         for (M, lam, x), d in tot.items()],
        columns=["num_jurors","lambda_qre","x_mean","pX"],
    )

    eps_px = pd.DataFrame(
        [(M, lam, x, eps, (d["sum_q"] / d["qcnt"] if d["qcnt"] else np.nan))
         for (M, lam, x, eps), d in eps_tot.items()],
        columns=["num_jurors", "lambda_qre", "x_mean", "epsilon", "P_X"]
    )

    eps_win = pd.DataFrame(
        [(M, lam, x, eps, d["xwins"] / d["n"])
         for (M, lam, x, eps), d in eps_tot.items()],
        columns=["num_jurors", "lambda_qre", "x_mean", "epsilon", "X_win_rate"]
    )
    
    tie_rate = pd.DataFrame(
        [(M, lam, x, d["ties"]/d["n"]) for (M, lam, x), d in tot.items()],
        columns=["num_jurors","lambda_qre","x_mean","tie_rate"],
    )
    
    # Outcome mix by x (across λ), per M
    #   -> we’ll melt to X/Y proportions for the stacked-area plot
    tmp = {}
    for (M, lam, x), d in tot.items():
        tmp.setdefault((M, x), {"n":0, "xwins":0})
        tmp[(M, x)]["n"]     += d["n"]
        tmp[(M, x)]["xwins"] += d["xwins"]
    outcome_by_x = pd.DataFrame(
        [(M, x, v["xwins"]/v["n"]) for (M, x), v in tmp.items()],
        columns=["num_jurors","x_mean","X_rate"],
    ).assign(Y_rate=lambda df: 1 - df["X_rate"]).melt(
        id_vars=["num_jurors","x_mean"],
        value_vars=["X_rate","Y_rate"],
        var_name="Majority", value_name="proportion"
    ).replace({"X_rate":"X", "Y_rate":"Y"})

    return grid_win, grid3, tie_rate, outcome_by_x, eps_px, eps_win

def nearest_value(available: np.ndarray, target: float) -> float:
    idx = np.abs(available - target).argmin()
    return float(available[idx])

st.set_page_config(page_title="Batch Processor")
st.title("Batch Simulation Processor")
st.sidebar.header("Batch Configuration")
st.sidebar.markdown("Select ranges or fixed values:")

# ========== Parameter Selection UI ==========

def range_input(label, fixed, min_val, max_val, default, step, key, force_float=False):
    use_float = force_float or any(isinstance(v, float) for v in [min_val, max_val, default, step])

    if fixed:
        return [st.sidebar.slider(label, min_val, max_val, default, step=step, key=f"{key}_fixed")]

    min_v = st.sidebar.number_input(
        f"Min {label}",
        float(min_val) if use_float else int(min_val),
        float(max_val) if use_float else int(max_val),
        float(min_val) if use_float else int(min_val),
        step=float(step) if use_float else int(step),
        key=f"{key}_min"
    )

    max_v = st.sidebar.number_input(
        f"Max {label}",
        float(min_val) if use_float else int(min_val),
        float(max_val) if use_float else int(max_val),
        float(max_val) if use_float else int(max_val),
        step=float(step) if use_float else int(step),
        key=f"{key}_max"
    )

    inc = st.sidebar.number_input(
        f"Increment size for {label}",
        float(step) if use_float else int(step),
        float(max_val - min_val) if use_float else int(max_val - min_val),
        float(step) if use_float else int(step),
        step=float(step) if use_float else int(step),
        key=f"{key}_step"
    )

    return list(np.arange(min_v, max_v + inc, inc)) if use_float else list(range(int(min_v), int(max_v) + 1, int(inc)))

# Juror number (M)
fixed_jurors = st.sidebar.checkbox(r"Fix number of jurors ($M$)", value=True)
juror_range = range_input(r"Number of Jurors ($M$)", fixed_jurors, 1, 31, 9, 1, r"$M$", force_float=False)

# Base reward (p)
fixed_p = st.sidebar.checkbox(r"Fix base reward ($p$)", value=True)
p_range = range_input(r"base reward ($p$)", fixed_p, 0.0, 100.0, 33.40, 0.1, r"$p$")

# Deposit (d)
fixed_d = st.sidebar.checkbox(r"Fix deposit ($d$)", value=True)
d_range = range_input(r"deposit ($d$)", fixed_d, 0.0, 100.0, 99.49, 0.1, r"$d$")

# Lambda (QRE)
fixed_lambda = st.sidebar.checkbox(r"Fix $\lambda$ (QRE sensitivity)", value=True)
log_lambda_label = r"log$_{10}(\lambda)$"

if fixed_lambda:
    lambda_val = st.sidebar.slider(log_lambda_label, -3.0, 0.5, value=0.0, step=0.01, key="log_lambda_fixed")
    lambda_range = [10 ** lambda_val]

    lam = 10 ** lambda_val
    st.sidebar.latex(rf"\lambda = {lam:.3g}")
else:
    min_log = st.sidebar.number_input("Min log$_{10}(\lambda)$", -3.0, 0.5, -3.0, step=0.01, key="log_lambda_min")
    max_log = st.sidebar.number_input("Max log$_{10}(\lambda)$", -3.0, 0.5, 0.5, step=0.01, key="log_lambda_max")
    log_step = st.sidebar.number_input("Step size (log$_{10}(\lambda)$)", 0.01, 1.0, 0.1, step=0.01, key="log_lambda_step")

    lambda_range = [10 ** val for val in np.arange(min_log, max_log + log_step, log_step)]

    lam_min, lam_max = 10 ** min_log, 10 ** max_log
    step_factor = 10 ** log_step
    d_min = lam_min * (step_factor - 1.0)
    d_max = lam_max * (step_factor - 1.0)

    st.sidebar.latex(
        rf"\lambda \in \left[{lam_min:.3g},\, {lam_max:.3g}\right]"
    )

    st.sidebar.markdown("**Log-step grid:** each tick multiplies the next $\lambda$ value by")

    st.sidebar.latex(
        rf"{{10^{{{log_step:.3g}}} \approx \times {step_factor:.3g}}}"
    )

# Noise
fixed_noise = st.sidebar.checkbox("Fix noise", value=True)
noise_range = range_input("noise", fixed_noise, 0.0, 1.0, 0.1, 0.01, "noise")

# x_mean
fixed_x = st.sidebar.checkbox(r"Fix $x$ (expected coherence)", value=True)
x_range = range_input(r"$x$", fixed_x, 0.0, 1.0, 0.5, 0.01, r"$x$")

# payoff type
payoff_type = st.sidebar.selectbox("Payoff Type", ["Basic", "Redistributive", "Symbiotic"])

# attack_mode
attack_mode = st.sidebar.checkbox(r"Enable p+$\varepsilon$ Attack", value=False)

# epsilon
if attack_mode:
    fixed_epsilon = st.sidebar.checkbox(r"Fix $\varepsilon$ (Bribe amount)", value=True)
    epsilon_range = range_input(r"$\varepsilon$", fixed_epsilon, 0.0, 100.0, 0.0, 0.1, r"$\varepsilon$")

# number of simulations
num_simulations = st.sidebar.number_input("Simulations per combination", 10, 1000, 100, step=10)
st.sidebar.warning("**Warning:** Large grids with many parameters and small increments may take a long time. \n"
                   "\n The time remaining is generally not accurate as it runs the lower number of jurors first \n"
                   "\n A batch size of 50 000 x 100 simulations is recommended")

epsilon_vals = epsilon_range if attack_mode else [0.0] # makes values 0 if attack_mode is disabled

# ========== Button and Simulation Trigger ==========

if st.sidebar.button("Run Batch Simulation"):
    process=7 # for 8 cpu cores otherwise use more or less
    param_list = []
    for M, p, d, lam, n, x, eps in itertools.product(juror_range, p_range, d_range, lambda_range, noise_range, x_range, epsilon_vals):
        param_list.append({
            "num_jurors": M,
            "p": p,
            "d": d,
            "lambda_qre": lam,
            "noise": n,
            "x_mean": x,
            "payoff_type": payoff_type,
            "attack": attack_mode,
            "epsilon": eps,
            "num_simulations": num_simulations
        })

    total_batches = len(param_list)
    st.info(f"Running {total_batches} batches x {num_simulations} simulations...")

    chunksize = 750  # must match what's used in batch_runner.py
    approx_chunks = (total_batches + chunksize - 1) // chunksize
    st.info(f"Using {process} processes with approx. {approx_chunks} chunks")
    
    # Parallel call

    output_file = "batch_results.csv"
    run_batch_parallel(param_list, processes=process, output_file=output_file)

    # Load only head for preview
    df_preview = pd.read_csv(output_file, nrows=10)
    st.success("Simulations complete")
    st.dataframe(df_preview)

    # Download button
    with open(output_file, "rb") as f:
        st.download_button("Download CSV", f, file_name=output_file, mime="text/csv")

# ========== CSV Upload to analyses results ==========

st.markdown("### Upload CSV File to Analyse Results")
uploaded_file = st.file_uploader("Upload a CSV file from batch results")
st.warning("**Warning:** It is recommended to upload files below 1GB for faster processing")
if uploaded_file:
    st.info("Streaming & aggregating (no full CSV load to avoid overstraining RAM)")
    grid_win, grid3, tie_rate, outcome_by_x, eps_px, eps_win = stream_aggregates(uploaded_file)
    st.success("CSV loaded successfully")
    # show the head of the *raw CSV*
    uploaded_file.seek(0)  # ensure we're at the start
    try:
        preview = pd.read_csv(uploaded_file, nrows=10, engine="pyarrow")
    except Exception:
        uploaded_file.seek(0)
        preview = pd.read_csv(uploaded_file, nrows=10)
    st.dataframe(preview)

    uploaded_file.seek(0)

    # ---------- TOP: 3D P(X) vs λ vs x (select a single M) ----------

    # --- choose source for the 3D P(X) plots ---
    source3 = grid3.copy()  # default (no attack / no epsilon slice)
    eps_sel = None          # <-- track selected ε for titles

    st.markdown("### 3D: QRE P(X) by λ and x")

    m_opts = sorted(grid3["num_jurors"].dropna().unique().tolist())
    sel_M = st.selectbox("Jurors (M)", m_opts, index=0)

    if not eps_px.empty:
        eps_unique = np.sort(pd.to_numeric(eps_px["epsilon"], errors="coerce").dropna().unique())

        # bounds + step (robust if there's only one ε)
        eps_min = float(eps_unique.min())
        eps_max = float(eps_unique.max())
        eps_step = float(np.diff(eps_unique).min()) if eps_unique.size > 1 else 0.01
        eps_default = 0.0 if np.any(np.isclose(eps_unique, 0.0)) else float(np.median(eps_unique))

        eps_pick = st.slider(
            r"$\varepsilon$ value",
            eps_min,
            eps_max,
            float(np.clip(eps_default, eps_min, eps_max)),
            step=eps_step,
            format="%g",
            key="eps_slice_slider",
            disabled=(eps_min == eps_max),   # shows but greys out if only one ε
            help=r"Select which $\varepsilon$ slice to show in the 3D P(X) plots",
        )
        eps_sel = nearest_value(eps_unique, float(eps_pick))

        if eps_min == eps_max:
            st.caption(rf"Only one $\varepsilon$ in file ({eps_sel:.3g}); slider disabled.")
        else:
            st.caption(rf"Showing slice at $\varepsilon={eps_sel:.3g}$.")

        if eps_sel is not None:
            eps_slice = eps_px[np.isclose(eps_px["epsilon"], eps_sel)]
            if not eps_slice.empty:
                source3 = (
                    eps_slice.rename(columns={"P_X": "pX"})
                             [["num_jurors", "lambda_qre", "x_mean", "pX"]]
                )


    grid3_M = source3[source3["num_jurors"] == sel_M].sort_values(["lambda_qre", "x_mean"])

    # 3D scatter plot
    title_scatter = rf"\text{{QRE }} P(X) \text{{ by }} \lambda \text{{ and }} x \text{{ (M}}={sel_M}\text{{)}}"
    if eps_sel is not None:
        title_scatter += rf"\ \ (\varepsilon={eps_sel:.3g})"
    st.latex(title_scatter)
    
    fig_scatter = px.scatter_3d(
        grid3_M, x="x_mean", y="lambda_qre", z="pX",
        color="pX", color_continuous_scale="Viridis",
        labels={"lambda_qre": "λ", "x_mean": "x", "pX": "P(X)"},
    )
    fig_scatter.update_traces(marker=dict(size=3))
    
    fig_scatter.update_layout(
        height=750, 
        scene=dict(
            xaxis=dict(title="x", autorange="reversed", range=[0, 1])
        ),
        scene_camera=dict(
            projection=dict(type="orthographic") 
        ),
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 3D surface (only if grid is rectangular)
    lam_vals = np.sort(grid3_M["lambda_qre"].unique())
    x_vals   = np.sort(grid3_M["x_mean"].unique())
    pivot    = grid3_M.pivot(index="x_mean", columns="lambda_qre", values="pX")

    is_rect  = (pivot.shape == (len(x_vals), len(lam_vals))) and pivot.notna().all().all()
    if is_rect:

        surf_title = rf"\text{{Surface: QRE }} P(X) \text{{ over }} (\lambda,\ x) \text{{ (M}}={sel_M}\text{{)}}"
        if eps_sel is not None:
            surf_title += rf"\ \ (\varepsilon={eps_sel:.3g})"
        st.latex(surf_title)

        fig_surface = go.Figure(data=[go.Surface(
            x=x_vals, y=lam_vals, z=pivot.values.T,
            colorscale="Viridis", showscale=True
        )])
        fig_surface.update_layout(
            height=750,
            scene=dict(
                xaxis=dict(title="x", autorange="reversed", range=[0, 1]),
                yaxis=dict(title="λ"),  # reuse min_lam/max_lam from above
                zaxis=dict(title="P(X)", range=[0, 1]),
                ),
            scene_camera=dict(
                projection=dict(type="orthographic")
                ),
        )

        st.plotly_chart(fig_surface, use_container_width=True)

    # ---------- Phase boundary: λ*(x) where X_win_rate ≥ τ ----------

    st.markdown(r"""
    #### Phase boundary $\lambda^*(x)$

    This plot shows, for each coherence prior $x$ (expected share of peers voting $\text{X}$), the **minimum QRE sensitivity** needed for $\text{X}$ to win often enough.
 
    **Definition.**
    """)

    st.latex(r"""\lambda^*(x)=\min\{\lambda:\,\widehat{P}(\mathrm{majority}=\text{X} \mid x,\lambda)\ge \tau\}""")

    st.markdown(r"""
    **How to read the curve**
    - **Above** the curve ($\lambda \ge \lambda^*(x)$): $\text{X}$ is the majority in at least a $\tau$ fraction of runs.  
    - **Below** the curve: $\text{X}$ does not meet the $\tau$ bar (the bribed $\text{Y}$ tends to prevail).  
    - Increasing $\tau$ moves the curve **up** (a stricter success criterion).  
    - Changing the panel size $\text{M}$ shifts the boundary because the majority threshold and vote aggregation change with $\text{M}$.
    """)

    tau = st.slider(r"Threshold $\tau$ for '$\text{X}$ wins' (rate ≥ $\tau$)", 0.5, 0.95, value=0.5, step=0.05)
    gM  = grid_win[grid_win["num_jurors"] == sel_M].sort_values(["x_mean", "lambda_qre"])

    # Ensure numeric dtypes (streamed CSVs can give strings)
    for col in ("x_mean", "lambda_qre", "X_win_rate"):
        gM[col] = pd.to_numeric(gM[col], errors="coerce")
    gM = gM.dropna(subset=["x_mean", "lambda_qre", "X_win_rate"])

    # For each x, pick the smallest λ whose win rate meets τ (order-independent)
    curve = (
        gM[gM["X_win_rate"] >= tau]
          .groupby("x_mean", as_index=False)["lambda_qre"]
          .min()
          .rename(columns={"lambda_qre": "lambda_star"})
          .sort_values("x_mean")
    )

    if curve.empty:
        st.info(r"No $\lambda$ satisfies the threshold $\tau$ for this $\text{M}$; try lowering $\tau$ or choosing another $\text{M}$ value.")
    else:
        st.latex(rf"\text{{Critical }} \lambda^*(x) \text{{ for }} \tau = {tau:.2f} \text{{ (M}}={sel_M}\text{{)}}")

        st.altair_chart(
            alt.Chart(curve).mark_line(point=True).encode(
                x=alt.X("x_mean:Q", title="x"),
                y=alt.Y("lambda_star:Q", title="λ*"),
                tooltip=[alt.Tooltip("x_mean:Q", format=".3f"),
                         alt.Tooltip("lambda_star:Q", format=".3g")]
            ).properties(height=320),
            use_container_width=True
        )

    # ---------- Average QRE P(X) vs λ (colored by x) ----------
    st.latex(rf"\text{{Average QRE }} P(X) \text{{ by }} \lambda \text{{ (M}}={sel_M}\text{{)}}")

    st.altair_chart(
        alt.Chart(grid3_M).mark_line().encode(
            x=alt.X("lambda_qre:Q", title="λ"),
            y=alt.Y("pX:Q", title="P(X)"),
            color=alt.Color("x_mean:O", title="x"),
            tooltip=["lambda_qre", "x_mean", alt.Tooltip("pX:Q", format=".3f")]
        ).properties(height=320),
        use_container_width=True
    )

    # ---------- Outcome mix by x (stacked area) ----------
    outcome_dist = outcome_by_x[outcome_by_x["num_jurors"] == sel_M]

    st.latex(rf"\text{{Distribution of Majority Outcome by }} x \text{{ (M}}={sel_M}\text{{)}}")

    st.altair_chart(
        alt.Chart(outcome_dist).mark_area().encode(
            x=alt.X("x_mean:Q", title="x"),
            y=alt.Y("proportion:Q", stack="normalize", title="Proportion"),
            color=alt.Color("Majority:N", title="Majority Outcome"),
            tooltip=["x_mean", "Majority", alt.Tooltip("proportion:Q", format=".2f")]
        ).properties(height=300),
        use_container_width=True
    )
    
    # ---------- ε (bribe) analysis — only if attack is present in data ----------
    if not eps_px.empty:
        st.markdown("### Effect of bribe $\\varepsilon$")

        dfA_px  = eps_px[eps_px["num_jurors"] == sel_M]
        dfA_win = eps_win[eps_win["num_jurors"] == sel_M]

        left, right = st.columns(2)
        with left:
            x_unique = np.sort(dfA_px["x_mean"].dropna().unique())
            # smallest gap as slider step (fallback if only one value)
            x_step = float(np.diff(x_unique).min()) if len(x_unique) > 1 else 0.01

            x_pick = st.slider(
                "Fix coherence x",
                float(x_unique.min()),
                float(x_unique.max()),
                float(np.median(x_unique)),
                step=x_step,
                key="eps_x_slider",
            )
            # snap to the nearest available x in the data
            sel_x_for_eps = nearest_value(x_unique, x_pick)

        with right:
            st.caption(fr"Jurors ($\text{{M}}$) fixed at **{sel_M}** for these $\varepsilon$-plots.")

        lam_all = np.sort(dfA_px["lambda_qre"].dropna().unique())
        default_lams = [lam_all[0], lam_all[len(lam_all)//2], lam_all[-1]] if len(lam_all) >= 3 else lam_all.tolist()

        chosen_lams = st.multiselect(
            "Show λ curves",
            lam_all.tolist(),
            default=default_lams,
            format_func=lambda v: f"{v:.3g}",
            key="eps_lambda_curves",
        )

        # 2D: P(X) vs ε
        lines_px = (
            dfA_px[(dfA_px["x_mean"] == sel_x_for_eps) & (dfA_px["lambda_qre"].isin(chosen_lams))]
            .sort_values(["lambda_qre", "epsilon"])
        )

        st.latex(rf"P(X) \text{{ vs }} \varepsilon \text{{ at M}} ={sel_M},\ x={sel_x_for_eps:.3g}")

        st.altair_chart(
            alt.Chart(lines_px).mark_line(point=True).encode(
                x=alt.X("epsilon:Q", title="ε (bribe amount)"),
                y=alt.Y("P_X:Q", title="P(X)"),
                color=alt.Color("lambda_qre:N", title="λ", sort=chosen_lams, scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("epsilon:Q", format=".3f"),
                         alt.Tooltip("lambda_qre:N", title="λ", format=".3g"),
                         alt.Tooltip("P_X:Q", title="P(X)", format=".3f")],
            ).properties(height=320),
            use_container_width=True,
        )

        # 2D: Majority(X) rate vs ε
        lines_win = (
            dfA_win[(dfA_win["x_mean"] == sel_x_for_eps) & (dfA_win["lambda_qre"].isin(chosen_lams))]
            .sort_values(["lambda_qre", "epsilon"])
        )

        st.latex(rf"P(\text{{Majority}}=\text{{X}}) \text{{ vs }} \varepsilon \text{{ at M}} ={sel_M},\ x={sel_x_for_eps:.3g}")

        st.altair_chart(
            alt.Chart(lines_win).mark_line(point=True).encode(
                x=alt.X("epsilon:Q", title="ε (bribe amount)"),
                y=alt.Y("X_win_rate:Q", title="P(Majority = X)"),
                color=alt.Color("lambda_qre:N", title="λ", sort=chosen_lams, scale=alt.Scale(scheme="greens")),
                tooltip=[alt.Tooltip("epsilon:Q", format=".3f"),
                         alt.Tooltip("lambda_qre:N", title="λ", format=".3g"),
                         alt.Tooltip("X_win_rate:Q", title="P(Maj=X)", format=".3f")],
            ).properties(height=300),
            use_container_width=True,
        )
 
        # 3D surfaces
        st.markdown("#### 3D: Surfaces involving $\\varepsilon$")
        surface_mode = st.radio("Surface axes", ["ε vs λ (fix x)", "ε vs x (fix λ)"], horizontal=True, key="eps_surface_mode")

        if surface_mode == "ε vs λ (fix x)":
            x_unique = np.sort(dfA_px["x_mean"].dropna().unique())
            x_step   = float(np.diff(x_unique).min()) if len(x_unique) > 1 else 0.01

            x_pick = st.slider("Fix x (coherence)", float(x_unique.min()), float(x_unique.max()),
                               float(np.median(x_unique)), step=x_step)
            x_for_surface = nearest_value(x_unique, x_pick)
            grid_eps = dfA_px[dfA_px["x_mean"] == x_for_surface][["epsilon","lambda_qre","P_X"]]
            eps_vals = np.sort(grid_eps["epsilon"].unique()); lam_vals = np.sort(grid_eps["lambda_qre"].unique())
            pivot = grid_eps.pivot(index="epsilon", columns="lambda_qre", values="P_X")
            if pivot.shape == (len(eps_vals), len(lam_vals)) and pivot.notna().all().all():

                st.latex(rf"\text{{Surface: }} P(X) \text{{ over }} (\varepsilon,\ \lambda) \text{{ (M}}={sel_M},\ x={x_for_surface:.3g}\text{{)}}")

                fig = go.Figure(data=[go.Surface(x=lam_vals, y=eps_vals, z=pivot.values,
                                                 colorscale="Viridis", cmin=0, cmax=1, showscale=True)])
                fig.update_layout(
                    height=750,
                    scene=dict(
                        xaxis=dict(title="λ"),
                        yaxis=dict(title="ε"),
                        zaxis=dict(title="P(X)", range=[0, 1]),
                    ),
                    scene_camera=dict(
                        projection=dict(type="orthographic")
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Surface needs a full rectangular grid in (ε, λ) for the chosen M and x.")
        else:
            lam_unique = np.sort(dfA_px["lambda_qre"].dropna().unique())
            lam_step   = float(np.diff(lam_unique).min()) if len(lam_unique) > 1 else float(lam_unique[0] or 1e-3)

            lam_pick = st.slider("Fix λ", float(lam_unique.min()), float(lam_unique.max()),
                                 float(np.median(lam_unique)), step=lam_step, format="%g")
            lam_for_surface = nearest_value(lam_unique, lam_pick)
            grid_eps = dfA_px[dfA_px["lambda_qre"] == lam_for_surface][["epsilon","x_mean","P_X"]]
            eps_vals = np.sort(grid_eps["epsilon"].unique()); x_vals = np.sort(grid_eps["x_mean"].unique())
            pivot = grid_eps.pivot(index="epsilon", columns="x_mean", values="P_X")
            if pivot.shape == (len(eps_vals), len(x_vals)) and pivot.notna().all().all():

                st.latex(rf"\text{{Surface: }} P(X) \text{{ over }} (\varepsilon,\ x) \text{{ (M}}={sel_M},\ \lambda={lam_for_surface:.3g}\text{{)}}")

                fig = go.Figure(data=[go.Surface(x=x_vals, y=eps_vals, z=pivot.values,
                                                 colorscale="Viridis", showscale=True)])
                fig.update_layout(
                    height=750,
                    scene=dict(
                        xaxis=dict(title="x", autorange="reversed", range=[0, 1]),
                        yaxis=dict(title="ε"),
                        zaxis=dict(title="P(X)", range=[0, 1]),
                    ),
                    scene_camera=dict(
                        projection=dict(type="orthographic")
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(r"Surface needs a full rectangular grid in $(\varepsilon, x)$ for the chosen $\text{M}$ and $\lambda$.")
