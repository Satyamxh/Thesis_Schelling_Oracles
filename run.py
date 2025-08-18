# page: Schelling Point Oracle

"""
To run this in terminal use

streamlit run "run.py"

the first page corresponds to the simulation

the second page corresponds to batch processing to generate a large CVS file for
further analysis of the parameters
"""

import streamlit as st
st.set_page_config(page_title="Simulate")

import streamlit as st
import pandas as pd
import altair as alt

from model import OracleModel
from payoff_mechanisms import (compute_payoff_basic_attack, compute_payoff_basic_no_attack, 
                               compute_payoff_redistributive_attack, compute_payoff_redistributive_no_attack, 
                               compute_payoff_symbiotic_attack, compute_payoff_symbiotic_no_attack)

# to add paragraph in help (formatting) - streamlit does not allow markdown and this is a way to bypass that restriction

help_attack = """
Tick if you want to enable a $p+\\varepsilon$ attack. 

$p+\\varepsilon$ attacks are done via smart contracts, they are publicly visible and target all jurors.

This simply means the juror's payoff matrix adapts to fit the attack.
"""

help_payoff_mech = """
Choose how jurors are rewarded depending on how their vote aligns with the outcome.

**Basic**: Jurors who vote with the majority get a fixed reward; others lose their deposit.

**Redistributive**: Losers' deposits are redistributed among winners. Payoff depends on how many others voted the same way.

**Symbiotic**: Rewards increase with coordination. The more jurors vote the same way, the greater the reward — fostering consensus.
"""

help_noise = """
This models uncertainty in the juror's perception of expected payoffs for each option.

A higher value increases the likelihood that jurors misjudge which option maximises their payoff.

This helps to simulate cognitive bias or limited understanding. This affects strategic (rational) voting behaviour.
"""

# help_x_guess = """
# This models uncertainty in the juror's belief about how many other jurors will vote for X.

# A conservative estimate of 50$\%$ of the total number of jurors is selected for $x$.

# This parameter sets the level of variation. A higher value means more variation in the juror's internal estimate of $x$.

# This helps to human decision-making.
# """

st.title("Schelling Oracle Simulation")

# Sidebar controls for model parameters
st.sidebar.header("Simulation Parameters")
num_jurors = st.sidebar.slider("Number of Jurors", min_value=1, max_value=100, value=10, step=1,
                               help="Specifies the number of jurors voting.")
log_lambda = st.sidebar.slider(r"log$_{10}$ QRE Sensitivity ($\lambda$)",
                               -3.0, # log10(0.001)
                               0.5, # log10(10)
                               value=-1.73, 
                               step=0.01,
                               help=r"Higher $\lambda$ means jurors are more sensitive to payoff differences (closer to rational). Lower values add noise and irrationality.")
lambda_qre = 10 ** log_lambda
st.sidebar.latex(rf"\lambda = {lambda_qre:.3g}")
noise = st.sidebar.slider("Perception Noise (Payoff Uncertainty)", 0.0, 1.0, value=0.1, step=0.01,
                          help=help_noise)
deposit = st.sidebar.slider("Deposit ($d$)", 0.0, 100.0, value=99.49, step=0.1,
                            help="Specifies the initial deposit paid by the juror ($d$ in payoff matrix).")
base_reward_frac = st.sidebar.slider("Base Reward ($p$)", 0.0, 100.0, value=33.40, step=0.1,
                                     help="Specifies the reward for voting with the majority ($p$ in payoff matrix).")
x_mean = st.sidebar.slider("Expected Share of Votes for $X$ ($x$)", 0.0, 1.0, value=0.83, step=0.01,
                                  help=r"This sets the focal point for $x$: the juror's expected proportion of other jurors voting $\text{X}$ (the coherent vote).")
payoff_options = ["Basic", "Redistributive", "Symbiotic"]
payoff_mode = st.sidebar.selectbox(
    "Payoff Mechanism",
    payoff_options,
    index=1,
    help=help_payoff_mech,
)
attack_mode = st.sidebar.checkbox(r"Enable p+$\varepsilon$ Attack", value=False,
                                  help=help_attack)
epsilon_bonus = st.sidebar.slider(r"Epsilon (Bribe amount $\varepsilon$)", 0.0, 100.0, value=0.0, step=0.1, disabled=(not attack_mode),
                                  help=r"Specifies Bribe amount ($\varepsilon$ in payoff matrix).")
num_rounds = st.sidebar.number_input("Number of Simulation Rounds", min_value=1, max_value=10000, value=100, step=1,
                                     help="Specifies number of simulations to run.")

# Initialize the Oracle model with selected parameters
model = OracleModel(num_jurors=num_jurors,
                    lambda_qre=lambda_qre,
                    noise=noise,
                    p=base_reward_frac,
                    d=deposit,
                    epsilon=epsilon_bonus,
                    payoff_type=payoff_mode,
                    attack=attack_mode,
                    x_mean=x_mean,
                    #x_guess_noise=x_guess_noise
                    )

progress_bar = st.progress(0)
status_text = st.empty()

# Run the simulation for the specified number of rounds
results = model.run_simulations(int(num_rounds), progress_bar=progress_bar, status_text=status_text)

progress_bar.empty()
status_text.empty()

# use model history for normal rounds
history_X = results.get("history_X", [])
history_Y = results.get("history_Y", [])
avg_payoff_X = results.get("avg_payoff_X", [])
avg_payoff_Y = results.get("avg_payoff_Y", [])

# Prepare DataFrame for plotting and CSV download
index_label = "Round"
rounds_index = list(range(1, len(history_X) + 1))

df_overlay = None
rounds_index = list(range(1, len(history_X) + 1))

data_dict = {

    # input paramters
    
    "Round": rounds_index,
    "Number of Jurors": [num_jurors] * len(history_X),
    "base reward (p)": [results["p"]] * len(history_X),
    "deposit (d)": [results["d"]] * len(history_X),
    "noise": [results["noise"]] * len(history_X),
    "lambda_qre": [results["lambda_qre"]] * len(history_X),
    "x_mean": [results["x_mean"]] * len(history_X),
    # "x_guess_noise": [results["x_guess_noise"]] * len(history_X) if "x_guess_noise" in results else [0.0] * len(history_X),
    "payoff_type": [results["payoff_type"]] * len(history_X),
    
    # output parameters

    "X_votes": history_X,
    "Y_votes": history_Y,
    "avg_payoff_X": avg_payoff_X,
    "avg_payoff_Y": avg_payoff_Y,
    "qre_prob_X": results.get("qre_prob_X_list", [None] * len(history_X)),
    "utility_X": results.get("utility_X_list", [None] * len(history_X)),
    "utility_Y": results.get("utility_Y_list", [None] * len(history_X)),

    # standard deviations

    "std_votes_X": [results["std_votes_X"]] * len(history_X),
    "std_votes_Y": [results["std_votes_Y"]] * len(history_X),
    "std_payoff_X": [results["std_payoff_X"]] * len(history_X),
    "std_payoff_Y": [results["std_payoff_Y"]] * len(history_X),
    "avg_qre_prob_X": [results["avg_qre_prob_X"]] * len(history_X),

}

# if attack mode add epsilon value
if attack_mode:
    data_dict["epsilon"] = [results["epsilon"]] * len(history_X)

# add no-attack vote columns if attack mode
if attack_mode and "history_X_no_attack" in results:
    data_dict["X_votes_no_attack"] = results["history_X_no_attack"]
    data_dict["Y_votes_no_attack"] = results["history_Y_no_attack"]

df = pd.DataFrame(data_dict)

# determine the majority and whether attack succeeded
df["Majority"] = df.apply(
    lambda row: "Tie" if row["X_votes"] == row["Y_votes"]
    else ("X" if row["X_votes"] > row["Y_votes"] else "Y"),
    axis=1
)
df["Tie"] = (df["X_votes"] == df["Y_votes"]).astype(int)
if attack_mode:
    df["AttackSucceeded"] = df["Majority"].apply(lambda m: 1 if m == "Y" else 0)
else:
    df["AttackSucceeded"] = 0

# Payoff matrix visualisation
st.subheader("Payoff Mechanism Matrix")

# Prepare table content based on selected payoff type and attack mode
if payoff_mode == "Basic":
    st.markdown("#### Basic Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$p$",
            r"$-d$" if not attack_mode else r"$p+\varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$p$"
        ]
    }
    variables = [
        "- **$p$**: Base reward",
        "- **$d$**: Deposit amount",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

elif payoff_mode == "Redistributive":
    st.markdown("#### Redistributive Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$\frac{(M - x - 1)d + Mp}{x + 1}$",
            r"$-d$" if not attack_mode else r"$\frac{(M - x - 1)d + Mp}{x + 1} + \varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$\frac{xd + Mp}{M - x}$"
        ]
    }
    variables = [
        "- **$p$**: Base reward multiplier",
        "- **$d$**: Deposit amount",
        "- **$x$**: Number of jurors who voted for X (other than user)",
        "- **$M$**: Total number of jurors",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

elif payoff_mode == "Symbiotic":
    st.markdown("#### Symbiotic Mechanism" + (" with Attack" if attack_mode else ""))
    data = {
        "X wins": [
            r"$\frac{p(x + 1)}{M}$", 
            r"$-d$" if not attack_mode else r"$\frac{p(x + 1)}{M} + \varepsilon$"],
        "Y wins": [
            r"$-d$",
            r"$\frac{p(M - x)}{M}$"
        ]
    }
    variables = [
        "- **$p$**: Base reward multiplier",
        "- **$d$**: Deposit amount",
        "- **$x$**: Number of jurors who voted for X (other than user)",
        "- **$M$**: Total number of jurors",
    ]
    if attack_mode:
        variables.append(r"- **$\varepsilon$**: Bribe amount")

# Display the table and variable explanations

df_details = pd.DataFrame(data, index=["User votes X", "User votes Y"])
st.table(df_details)

st.markdown("### Variables")
for var in variables:
    st.markdown(var)

# st.markdown("---") - adds a line to seperate parts cleanly
st.markdown("### Expected vs Actual Numeric Payoff Matrix")
st.markdown(r"""<span style='color:green'>Green = Based on juror's expected $\mathit{x}$</span><br><span style='color:red'>Red = Based on actual $\mathit{x}$  from simulation</span>""", unsafe_allow_html=True)

# Compute x values
M = num_jurors
p = base_reward_frac
d = deposit
epsilon = epsilon_bonus if attack_mode else 0

# show expected 'x' value based on x_mean vs actual 'x' value calculated post simulation
x_expected = (x_mean * (M - 1)) # number of jurors other than the user voting 'X'
x_actual = df["X_votes"].mean() * (num_jurors - 1) / num_jurors # average number of other jurors voting 'X'

# 'x' value based on x_mean vs actual 'x' value as a percentage
x_expected_pct = 100 * x_expected / (num_jurors - 1)
x_actual_pct = 100 * x_actual / (num_jurors - 1)

# compute payoffs for each case (VOTE, OUTCOME): (X,X), (X,Y), (Y,X), (Y,Y)
if payoff_mode == "Basic":
    payoff_func = compute_payoff_basic_attack if attack_mode else compute_payoff_basic_no_attack
    payoff_args = (p, d, epsilon) if attack_mode else (p, d)
    def wrapper(vote, outcome):
        return payoff_func(vote, outcome, *payoff_args)

elif payoff_mode == "Redistributive":
    payoff_func = compute_payoff_redistributive_attack if attack_mode else compute_payoff_redistributive_no_attack
    def wrapper(vote, outcome, x_val):
        return payoff_func(vote, outcome, x_val, M, p, d, epsilon) if attack_mode else payoff_func(vote, outcome, x_val, M, p, d)

elif payoff_mode == "Symbiotic":
    payoff_func = compute_payoff_symbiotic_attack if attack_mode else compute_payoff_symbiotic_no_attack
    def wrapper(vote, outcome, x_val):
        return payoff_func(vote, outcome, x_val, M, p, d, epsilon) if attack_mode else payoff_func(vote, outcome, x_val, M, p, d)

# defining cases: (VOTE, OUTCOME)
entries = [("X", "X"), ("Y", "X"), ("X", "Y"), ("Y", "Y")]

expected_vals = []
actual_vals = []

for vote, outcome in entries:
    if payoff_mode == "Basic":
        val = wrapper(vote, outcome)
        expected_vals.append(val)
        actual_vals.append(val)
    else:
        expected_vals.append(wrapper(vote, outcome, x_expected))
        actual_vals.append(wrapper(vote, outcome, x_actual))

# HTML matrix
import streamlit.components.v1 as components

html_matrix = f"""
<style>
    .matrix-table {{
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 16px;
        width: 60%;
    }}
    .matrix-table th, .matrix-table td {{
        border: 1px solid black;
        padding: 10px;
        text-align: center;
        vertical-align: middle;
        line-height: 1.5;
    }}
    .expected {{
        color: green;
        display: block;
    }}
    .actual {{
        color: red;
        display: block;
    }}
</style>

<table class="matrix-table">
    <tr>
        <th>Vote \\ Outcome</th>
        <th>X</th>
        <th>Y</th>
    </tr>
    <tr>
        <td><b>X</b></td>
        <td><span class='expected'>{expected_vals[0]:.2f}</span><span class='actual'>{actual_vals[0]:.2f}</span></td>
        <td><span class='expected'>{expected_vals[2]:.2f}</span><span class='actual'>{actual_vals[2]:.2f}</span></td>
    </tr>
    <tr>
        <td><b>Y</b></td>
        <td><span class='expected'>{expected_vals[1]:.2f}</span><span class='actual'>{actual_vals[1]:.2f}</span></td>
        <td><span class='expected'>{expected_vals[3]:.2f}</span><span class='actual'>{actual_vals[3]:.2f}</span></td>
    </tr>
</table>
"""

components.html(html_matrix, height=200)

# Display x values below the table
st.markdown(r"""
<span style='color:green'><b>Expected</b> $\mathit{x}$ : """  + f"{x_expected:.2f} ({x_expected_pct:.1f}%)" + r"""</span><br>
<span style='color:red'><b>Actual</b> $\mathit{x}$ : """  + f"{x_actual:.2f} ({x_actual_pct:.1f}%)" + r"""</span>
""", unsafe_allow_html=True)

# Display simulation results
st.subheader("Simulation Results")
if num_rounds == 1:
    # Single round: show outcome and votes
    outcome_counts = results["outcome_counts"]
    outcome = "X" if outcome_counts["X"] == 1 else "Y"
    votes_for_X = int(results.get("average_votes_X", 0))
    votes_for_Y = int(results.get("average_votes_Y", 0))
    st.write(f"Outcome of this round: **{outcome}**")
    st.write(f"Votes — X: {votes_for_X}, Y: {votes_for_Y}")
    if "Tie" in df.columns:
        st.write(f"The result was a tie")
    if attack_mode:
        if outcome == "Y":
            st.write("Attack Outcome: **Succeeded** (Target outcome Y achieved)")
        else:
            st.write("Attack Outcome: **Failed** (Target outcome Y not achieved)")
else:
    # Multiple rounds: show aggregated statistics
    total_runs = num_rounds 
    wins_X = (df["Majority"] == "X").sum()
    wins_Y = (df["Majority"] == "Y").sum()
    wins_Tie = (df["Majority"] == "Tie").sum()
    pct_X = (wins_X / total_runs) * 100
    pct_Y = (wins_Y / total_runs) * 100
    pct_Tie = (wins_Tie / total_runs) * 100
    st.write(f"Out of **{total_runs}** simulation rounds:")
    st.write(f"- Outcome **X** won **{wins_X}** times ({pct_X:.1f}%)")
    st.write(f"- Outcome **Y** won **{wins_Y}** times ({pct_Y:.1f}%)")
    if "Tie" in df.columns:
        st.write(f"- Number of **tie rounds**: {wins_Tie} ({pct_Tie:.1f}%)")
    if attack_mode:
        success_rate = results.get("attack_success_rate", 0)
        st.write(f"Attack Success Rate (Rate of Y wins as compared to no attack): **{success_rate:.1f}%**")
    avg_X = results.get("average_votes_X", None)
    avg_Y = results.get("average_votes_Y", None)
    
    if avg_X is not None and avg_Y is not None:
        st.write(f"Average votes per round — X: **{avg_X:.2f}**, Y: **{avg_Y:.2f}**")

# Line chart of vote counts across rounds (only shown if multiple rounds)
if len(df) > 1:
    st.subheader("Voting Dynamics Across Rounds")

    df_long = df.copy()

    # Create long-format data with Tie markers included
    melted = df_long.melt(id_vars=[index_label], value_vars=["X_votes", "Y_votes"],
                          var_name="Vote Type", value_name="Count")

    # Add Tie points as a separate category
    if "Tie" in df.columns and df["Tie"].sum() > 0:
        tie_df = df[df["Tie"] == 1].copy()
        tie_df["Vote Type"] = "Tie"
        tie_df["Count"] = tie_df["X_votes"]  # or Y_votes (they are equal in tie)
        tie_df = tie_df[[index_label, "Vote Type", "Count"]]
        melted = pd.concat([melted, tie_df], ignore_index=True)

    # Plot line chart for X and Y votes
    base_chart = alt.Chart(melted[melted["Vote Type"] != "Tie"]).mark_line().encode(
        x=alt.X(f"{index_label}:Q", title=index_label),
        y=alt.Y("Count:Q", title="Number of Votes"),
        color=alt.Color("Vote Type:N", title="Vote Option",
            scale=alt.Scale(domain=["X_votes", "Y_votes", "Tie"],
                            range=["steelblue", "red", "gold"]),
            legend=alt.Legend(labelExpr="""{
                'X_votes': 'Votes for X',
                'Y_votes': 'Votes for Y',
                'Tie': 'Tied Votes'
            }[datum.label]"""))
    ).properties(
        width=800,
        height=400,
    )

    # Add dots for ties
    tie_dots = alt.Chart(melted[melted["Vote Type"] == "Tie"]).mark_point(
        size=80,
        shape="circle"
    ).encode(
        x=alt.X(f"{index_label}:Q"),
        y=alt.Y("Count:Q"),
        color=alt.Color("Vote Type:N", scale=alt.Scale(domain=["Tie"], range=["gold"]))
    )

    st.latex(r"\textbf{Number of votes for } \mathrm{X}/\mathrm{Y}")

    st.altair_chart(base_chart + tie_dots, use_container_width=False)

# Average payoff

# Nullify payoff where tie occurs
df.loc[df["Tie"] == 1, ["avg_payoff_X", "avg_payoff_Y"]] = None

# Tie lines (dashed vertical rules where tie == 1)
tie_df = df[df["Tie"] == 1][[index_label]].copy()
tie_df["Vote Type"] = "Tie round"
tie_df["Average Payoff"] = 0  # Dummy value for y-axis binding

tie_lines = alt.Chart(tie_df).mark_rule(
    strokeDash=[4, 4],
    stroke="gold",
    strokeWidth=2
).encode(
    x=alt.X(f"{index_label}:Q"),
    color=alt.Color("Vote Type:N",
        scale=alt.Scale(domain=["avg_payoff_X", "avg_payoff_Y", "Tie round"],
                        range=["steelblue", "red", "gold"]),
        legend=alt.Legend(title="Vote Option",
            labelExpr="""{'avg_payoff_X': 'Payoff for voting X',
                          'avg_payoff_Y': 'Payoff for voting Y',
                          'Tie round': 'Tie round'}[datum.label]""")
    )
)

# Base payoff lines (as before)
base_payoff = alt.Chart(df).transform_fold(
    ["avg_payoff_X", "avg_payoff_Y"],
    as_=["Vote Type", "Average Payoff"]
).mark_line().encode(
    x=alt.X(f"{index_label}:Q", title=index_label),
    y=alt.Y("Average Payoff:Q", title="Payoff"),
    color=alt.Color("Vote Type:N",
        scale=alt.Scale(domain=["avg_payoff_X", "avg_payoff_Y", "Tie round"],
                        range=["steelblue", "red", "gold"]),
        legend=alt.Legend(title="Vote Option",
            labelExpr="""{'avg_payoff_X': 'Payoff for voting X',
                          'avg_payoff_Y': 'Payoff for voting Y',
                          'Tie round': 'Tie round'}[datum.label]""")
    )
).properties(width=800, height=400)

st.latex(r"\textbf{Average Payoffs for voting } \mathrm{X}/\mathrm{Y}")

# Combine and render
st.altair_chart(base_payoff + tie_lines, use_container_width=False)


# CSV download for all results (voting dynamics and average payoff)

csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Simulation Results as a CSV file",
    data=csv_data,
    file_name="Simulation_Results.csv",
    mime="text/csv"
)