from model import OracleModel
import pandas as pd
import math
from multiprocessing import Pool, cpu_count
from stqdm import stqdm
import csv
import os

def run_simulation(params):

    local_params = params.copy()
    num_simulations = local_params.pop("num_simulations")  # Remove before passing to OracleModel

    model = OracleModel(**local_params)
    results = model.run_simulations(num_simulations)
    
    # Extract history lists
    history_X = results.get("history_X", [])
    history_Y = results.get("history_Y", [])
    avg_payoff_X = results.get("avg_payoff_X", [])
    avg_payoff_Y = results.get("avg_payoff_Y", [])
    utility_X = results.get("utility_X_list", [None] * len(history_X))
    utility_Y = results.get("utility_Y_list", [None] * len(history_Y))
    qre_probs = results.get("qre_prob_X_list", [None] * len(history_X))

    rounds = range(1, len(history_X) + 1)
    rows = []

    for i, r in enumerate(rounds):
        row = {
            "Round": r,
            "Number of Jurors": params["num_jurors"],
            "base reward (p)": params["p"],
            "deposit (d)": params["d"],
            "noise": params["noise"],
            "lambda_qre": params["lambda_qre"],
            "x_mean": params["x_mean"],
            "payoff_type": params["payoff_type"],
            "attack": params["attack"],
            "epsilon": params["epsilon"] if params["attack"] else 0.0,

            "X_votes": history_X[i],
            "Y_votes": history_Y[i],
            "avg_payoff_X": avg_payoff_X[i] if i < len(avg_payoff_X) else None,
            "avg_payoff_Y": avg_payoff_Y[i] if i < len(avg_payoff_Y) else None,
            "utility_X": utility_X[i],
            "utility_Y": utility_Y[i],
            "qre_prob_X": qre_probs[i],

            "std_votes_X": results.get("std_votes_X", 0.0),
            "std_votes_Y": results.get("std_votes_Y", 0.0),
            "std_payoff_X": results.get("std_payoff_X", 0.0),
            "std_payoff_Y": results.get("std_payoff_Y", 0.0),
            "avg_qre_prob_X": results.get("avg_qre_prob_X", 0.0),
        }

        # Majority + Tie logic
        row["Majority"] = "Tie" if history_X[i] == history_Y[i] else ("X" if history_X[i] > history_Y[i] else "Y")
        row["Tie"] = 1 if history_X[i] == history_Y[i] else 0

        if params["attack"]:
            row["AttackSucceeded"] = 1 if row["Majority"] == "Y" else 0
            if "history_X_no_attack" in results:
                row["X_votes_no_attack"] = results["history_X_no_attack"][i]
                row["Y_votes_no_attack"] = results["history_Y_no_attack"][i]
        else:
            row["AttackSucceeded"] = 0

        rows.append(row)

    return rows  # list of dicts (to be flattened later)

def run_batch_over_params(param_list_chunk):
    all_rows = []
    for params in param_list_chunk:
        all_rows.extend(run_simulation(params))
    return all_rows

def chunk_list(lst, n):
    """Split list `lst` into chunks of size n"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_batch_parallel(param_list, processes, output_file="batch_results.csv"):
    chunksize = 750  # conservative size
    param_chunks = list(chunk_list(param_list, chunksize))
    total_chunks = len(param_chunks)

    # Remove previous output if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    header_written = False

    with Pool(processes=processes) as pool:
        results = pool.imap_unordered(run_batch_over_params, param_chunks)
        for chunk in stqdm(results, total=total_chunks):
            df_chunk = pd.DataFrame(chunk)
            if not header_written:
                df_chunk.to_csv(output_file, index=False, mode="w", quoting=csv.QUOTE_NONNUMERIC)
                header_written = True
            else:
                df_chunk.to_csv(output_file, index=False, header=False, mode="a", quoting=csv.QUOTE_NONNUMERIC)

    return pd.read_csv(output_file)