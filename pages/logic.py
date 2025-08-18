"""
Place this file as `pages/logic.py` (or run as a standalone Streamlit page).
It displays the Schelling-point oracle model logic, equations, and simulation flow.
No parameters are taken; this page is purely explanatory.
"""

import streamlit as st

st.set_page_config(page_title="Model Logic", layout="wide")

# ======= Small CSS polish (optional) =======
st.markdown(
    """
    <style>
    .callout {padding: 0.9rem 1.1rem; border-left: 4px solid #999; background: rgba(0,0,0,0.03);
              border-radius: 0.5rem; margin: 1rem 0;}
    .equation {font-size: 1.05rem;}
    .subtle {color: #666;}
    .mono {font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Model Logic")
st.caption(r"Schelling-point oracle with a targeted $p+\varepsilon$ attack (QRE agents)")

st.markdown(
    """
The goal of this page is to document the **equations used** by the simulation so
that results elsewhere in the app can be interpreted unambiguously.
This page **does not** expose any parameters or controls.
    """
)

st.header("1) Model Overview")
st.markdown(
    r"""
We model a single-round Schelling oracle with a panel of $\text{M}$ jurors who each vote
for one of two options, $\text{X}$ or $\text{Y}$. The majority outcome wins (ties are broken in
favour of $\text{X}$). A targeted **$p+\varepsilon$ attack** offers an extra bonus
$\varepsilon$ to jurors who vote for a chosen target (here $\text{Y}$), making
$\text{Y}$ more attractive.

Each juror is an agent using a **quantal response** (logit) rule governed by a
single sensitivity parameter $\lambda$. Agents are homogeneous in their decision
function. Optionally, Gaussian **perception noise** $\sigma$ is added to the
computed utilities before applying the logit.
    """
)

st.subheader("Key parameters")
st.markdown(
    r"""
- **Jury size** $\text{M}$: number of jurors; a majority of $\text{T} = \lfloor \text{M}/2 \rfloor + 1$ is required.
- **Base reward** $p$: payoff to a juror whose vote matches the final outcome.
- **Deposit** $d$: stake lost by a juror whose vote is not in the final outcome.
- **Attack bonus** $\varepsilon$: bonus promised to $\text{Y}$-voters under the attack.
- **QRE sensitivity** $\lambda$: higher means choices concentrate on higher expected utility.
- **Perception noise** $\sigma$: standard deviation of $\mathcal{N}(0,\sigma)$ added to utilities.
- **Expected peer $\text{X}$-vote fraction** $x \in [0,1]$: belief that any other juror votes $\text{X}$ with probability $x$.
    """
)

st.header("2) Payoff Mechanisms")
st.markdown(
    "The simulation supports three payoff schemes. The juror payoff is a function of the vote, the final outcome, and (for the last two schemes) the number of co-winners.")

basic, redist, symb = st.tabs(["Basic (Def. 2.1)", "Redistributive (Def. 2.2)", "Symbiotic (Def. 2.3)"])

with basic:
    st.markdown(
        r"""
**Basic / Simple Schelling game.**
If you vote with the outcome you gain $+p$; otherwise you lose $-d$.

Per-juror payoffs (no attack) in the form $u(\text{vote, outcome})$:
$$
\begin{aligned}
&u(X, X) = +p, && u(X, Y) = -d,\\
&u(Y, X) = -d, && u(Y, Y) = +p
\end{aligned}
$$

Under an active **$p+\varepsilon$ attack**, $u(Y,X)$ becomes: 
$$
\boxed{\;u(Y, X) = p + \varepsilon\;}
$$
while all other entries remain as above. (This captures the idea that voting $\text{Y}$
never performs worse than honest voting.)
        """
    )

with redist:
    st.markdown(
        r"""
**Redistributive game.** Losers' deposits are pooled and split equally among winners.
Let $k$ be the number of **other** jurors who vote $\text{X}$ (so there are $N=M-1$ others).
If you vote $\text{X}$ then the $\text{X}$-winner count is $k+1$ and the $\text{X}$-loser count is
$M-(k+1)$. Symmetrically for $\text{Y}$.

Per-juror payoffs (no attack) in the form $u(\text{vote, outcome})$:
$$
\begin{aligned}
&u(X, X\mid k) = \frac{(M-(k+1))d+Mp}{k+1}, && u(X, Y\mid k) = -d,\\
&u(Y, X\mid k) = -d, && u(Y, Y\mid k) = \frac{kd+Mp}{M-k}
\end{aligned}
$$

Under an active **$p+\varepsilon$ attack**, $u(Y,X)$ becomes: 
$$
\boxed{\;u(Y, X\mid k) = \frac{(M-(k+1))d+Mp}{k+1} + \varepsilon\;}
$$
while all other entries remain as above. (This captures the idea that voting $\text{Y}$
never performs worse than honest voting.)
        """
    )

with symb:
    st.markdown(
        r"""
**Symbiotic game.** Rewards scale with the size of the winning coalition.
If $k$ other jurors vote $\text{X}$ and $\text{X}$ wins, each $\text{X}$-winner gets a fraction of $p$ proportional to the winning share.

Per-juror payoffs (no attack) in the form $u(\text{vote, outcome})$:
$$
\begin{aligned}
&u(X, X\mid k) = \frac{p(k+1)}{M}, && u(X, Y\mid k) = -d,\\
&u(Y, X\mid k) = -d, && u(Y, Y\mid k) = \frac{p(M-k)}{M}
\end{aligned}
$$

Under an active **$p+\varepsilon$ attack**, $u(Y,X)$ becomes: 
$$
\boxed{\;u(Y, X\mid k) = \frac{p(k+1)}{M} + \varepsilon\;}
$$
while all other entries remain as above. (This captures the idea that voting $\text{Y}$
never performs worse than honest voting.)
        """
    )

st.header("3) Expected Utilities")
st.markdown(
    r"""
Let $N=M-1$ be the number of *other* jurors, and $T=\lfloor M/2 \rfloor + 1$ the majority threshold.
Each other juror votes $\text{X}$ independently with probability $x$ (the agent's belief).

For the **Basic** mechanism, it is convenient to work with outcome probabilities directly:
$$
\begin{aligned}
&P(\text{$\text{X}$ wins}\mid \text{vote }X) = \sum_{k=T-1}^{N} \binom{N}{k} x^k (1-x)^{N-k},\\
&P(\text{$\text{X}$ wins}\mid \text{vote }Y) = \sum_{k=T}^{N} \binom{N}{k} x^k (1-x)^{N-k}.
\end{aligned}
$$
Then the expected utilities are
$$
\begin{aligned}
U_X &= P(\text{$\text{X}$ wins}\mid X)\,u(X,X) + \bigl(1-P(\text{$\text{X}$ wins}\mid X)\bigr)\,u(X,Y),\\
U_Y &= P(\text{$\text{X}$ wins}\mid Y)\,u(Y,X) + \bigl(1-P(\text{$\text{X}$ wins}\mid Y)\bigr)\,u(Y,Y).
\end{aligned}
$$

For **Redistributive** and **Symbiotic** mechanisms, payoffs depend on the *exact* split of other votes.
Let $k\in\{0,\dots,N\}$ denote the number of other $\text{X}$ votes. The expected utilities are:
$$
\begin{aligned}
U_X &= \sum_{k=0}^{N} \underbrace{\binom{N}{k} x^k (1-x)^{N-k}}_{\Pr(k \text{ others vote } X)}\; u\bigl(X,\;\text{outcome from }k+1\text{ $\text{X}$ votes}\mid k\bigr),\\
U_Y &= \sum_{k=0}^{N} \binom{N}{k} x^k (1-x)^{N-k}\; u\bigl(Y,\;\text{outcome from }k\text{ $\text{X}$ votes}\mid k\bigr).
\end{aligned}
$$
    """
)

st.header("4) Quantal Response (Logit) Choice")
st.markdown(
    r"""
Given expected utilities $(U_X, U_Y)$, the juror votes probabilistically using a logit rule with sensitivity $\lambda$:
$$
P(\text{vote }X) = \frac{\exp(\lambda U_X)}{\exp(\lambda U_X)+\exp(\lambda U_Y)},\qquad
P(\text{vote }Y) = 1 - P(\text{vote }X).
$$

Optional perception noise $\sigma$ is added **before** applying the logit:
$$
U'_X = U_X + \mathcal{N}(0,\sigma),\qquad U'_Y = U_Y + \mathcal{N}(0,\sigma),
$$
then use $(U'_X, U'_Y)$ in the logit formula above.
    """
)

st.header("5) Simulation Procedure (One Round)")
st.markdown(
    "The simulator executes a single dispute round as follows (repeated many times for statistics):"
)

st.markdown(
    r"""
1. **Compute utilities & vote probabilities** for each juror using the formulas above (and add noise if $\sigma>0$).
2. **Sample votes** independently: each juror chooses $\text{X}$ with probability $P(\text{vote }X)$.
3. **Tally votes**: if $N_X \ge N_Y$ then the outcome is $\text{X}$ (tie-break to $\text{X}$), else $\text{Y}$.
4. **Assign payoffs** using the selected payoff mechanism (and attack rule if active).
5. **Record results**: $(N_X, N_Y)$, outcome, average payoffs by vote, and diagnostics.
    """
)

st.subheader("Reference pseudocode")
st.code(
    """
# Parameters: (M, p, d, epsilon, lambda, sigma, x, payoff_type, attack_flag)
N = M - 1
T = (M // 2) + 1
votes_X = votes_Y = 0

for i in range(M):
    # 1) Expected utilities for actions
    U_X, U_Y = compute_expected_utilities(M, p, d, epsilon, x, payoff_type, attack_flag)

    # Optional perception noise
    if sigma > 0:
        U_X += Normal(0, sigma)
        U_Y += Normal(0, sigma)

    # Quantal response probability
    p_i = exp(lambda * U_X) / (exp(lambda * U_X) + exp(lambda * U_Y))

    # 2) Sample a vote
    if rand() < p_i:
        votes_X += 1
    else:
        votes_Y += 1

# 3) Majority outcome (ties -> X)
outcome = "X" if votes_X >= votes_Y else "Y"

# 4) Payoffs depend on mechanism and full vote counts
#    (Basic uses fixed terminal payoffs; Redistributive/Symbiotic use formulas of k)
for i in range(M):
    payoff_i = u(vote_i, outcome, votes_X, votes_Y, mechanism=payoff_type, attack=attack_flag)

# 5) Record summary statistics
    """,
    language="python",
)

st.divider()

st.markdown(
    """
    <style>
    /* Style only the next top-level block after the anchor */
    #notes_anchor + div [data-testid="stVerticalBlock"]{
        background: #f3f4f6;          /* grey fill */
        border: 2px solid #666;       /* dark grey border */
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    </style>
    <div id="notes_anchor"></div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Notes:**")
    st.markdown(
        r"""
- *Tie-break to $\text{X}$:* $\text{X}$ is treated as the default Schelling answer.
  This only affects the boundary where $N_X = N_Y$.

- *Attack rule:* We implement the guarantee as:
        """
    )
    # 3) Center the equation via columns
    c1, c2, c3 = st.columns([1, 8, 1])
    with c2:
        st.latex(r"u(Y, X \mid k) = u(X, X \mid k) + \varepsilon")

    st.markdown(
        r"""
    In words: a $\text{Y}$-voter who "loses" is paid as if they had voted correctly plus $\varepsilon$.
    In the Basic scheme this reduces to $p+\varepsilon$.

- *Heterogeneity:* Jurors are identically parameterised. Each vote is independent across jurors.
        """
    )