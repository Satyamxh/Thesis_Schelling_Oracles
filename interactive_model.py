import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import plotly.graph_objects as go
from oracle_model import run_baseline_simulation, run_attack_simulation, was_attack_successful

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Kleros Dispute Resolution Simulation"),

    html.Label("Number of Jurors"),
    dcc.Slider(id='num_jurors', min=3, max=50, step=1, value=10, marks={i: str(i) for i in range(3, 51, 5)}),

    html.Label("Total Jurors"),
    dcc.Slider(id='total_jurors', min=10, max=500, step=10, value=100, marks={i: str(i) for i in range(10, 501, 50)}),

    html.Label("Bribe Amount (ε)"),
    dcc.Slider(id='bribe_amount', min=0, max=5, step=0.1, value=1, marks={i: str(i) for i in range(6)}),

    html.Label("Bribe Acceptance Probability"),
    dcc.Slider(id='bribe_acceptance_prob', min=0, max=1, step=0.05, value=0.3, marks={i/10: str(i/10) for i in range(0, 11)}),

    html.Label("Honesty Level"),
    dcc.Slider(id='honesty_level', min=0, max=1, step=0.05, value=0.8, marks={i/10: str(i/10) for i in range(0, 11)}),

    html.Label("Rationality"),
    dcc.Slider(id='rationality', min=0, max=1, step=0.05, value=0.7, marks={i/10: str(i/10) for i in range(0, 11)}),

    html.Label("Bribed Juror Ratio"),
    dcc.Slider(id='bribed_juror_ratio', min=0, max=1, step=0.05, value=0.2, marks={i/10: str(i/10) for i in range(0, 11)}),

    html.Label("Number of Simulations"),
    dcc.Slider(id='num_simulations', min=100, max=10000, step=100, value=5000, marks={i: str(i) for i in range(100, 10001, 1000)}),

    html.Button('Run Simulation', id='run_simulation', n_clicks=0),
    html.Button('Run Validation (Attack Disabled)', id='run_validation', n_clicks=0),

    html.H3("Average Jurors Converted by Bribery:"),
    html.Div(id='attack_success_rate'),

    dcc.Graph(id='attack_results_plot'),
])

@app.callback(
    [Output('attack_success_rate', 'children'),
     Output('attack_results_plot', 'figure'),
     Output('honesty_level', 'value'),
     Output('rationality', 'value')],
    [Input('run_simulation', 'n_clicks'),
     Input('run_validation', 'n_clicks')],
    [State('num_simulations', 'value'),
     State('num_jurors', 'value'),
     State('total_jurors', 'value'),
     State('bribe_amount', 'value'),
     State('bribe_acceptance_prob', 'value'),
     State('honesty_level', 'value'),
     State('rationality', 'value'),
     State('bribed_juror_ratio', 'value')]
)
def run_simulation(n_clicks, validation_n_clicks, num_simulations, num_jurors, total_jurors, 
                   bribe_amount, bribe_acceptance_prob, honesty_level, rationality, bribed_juror_ratio):
    """Runs Monte Carlo simulations and measures how many jurors changed votes due to bribery."""
    
    validation_mode = validation_n_clicks > n_clicks  

    # If validation mode is activated, override attack parameters and slider values
    if validation_mode:
        effective_honesty = 1.0
        effective_rationality = 0.0
    else:
        effective_honesty = honesty_level
        effective_rationality = rationality

    # Define beliefs once, so baseline and attack simulations use the same set of beliefs
    beliefs = np.random.choice(["X", "Y"], num_jurors).tolist()
    
    baseline_votes = run_baseline_simulation(num_jurors, total_jurors, beliefs, effective_honesty, effective_rationality)
    
    total_converted = []
    for _ in range(num_simulations):
        np.random.seed(42)  # Consistent randomization
        # Using the same beliefs for each simulation run:
        attack_votes = run_attack_simulation(num_jurors, total_jurors, beliefs, effective_honesty, effective_rationality,
                                             bribe_amount, bribe_acceptance_prob, bribed_juror_ratio)
        converted_jurors = was_attack_successful(baseline_votes, attack_votes)
        total_converted.append(converted_jurors)
        
    avg_converted = np.mean(total_converted)

    if validation_mode:
        assert avg_converted == 0.0, f"Validation mode failed! Expected 0.0 but got {avg_converted}"
    
    attack_success_text = f"Average Jurors Converted by Bribery: {avg_converted:.2f}"
    
    fig = go.Figure(data=[go.Bar(
        x=['Votes Unchanged', 'Votes Changed by Bribery'],
        y=[num_jurors - avg_converted, avg_converted],
        marker=dict(color=['blue', 'red'])
    )])
    fig.update_layout(title="Bribery Impact on Jurors", xaxis_title="Outcome", yaxis_title="Average Jurors Affected")
    
    # Also update the slider values to show the fixed values in validation mode
    return attack_success_text, fig, effective_honesty, effective_rationality

if __name__ == "__main__":
    app.run(debug=True)