import numpy as np

########################################
# 1. Basic Payoff Model (Definition 2.1)
########################################
def compute_payoff_basic_no_attack(vote, outcome, p, d):
    if outcome == "X":
        return p if vote == "X" else -d
    else:
        return p if vote == "Y" else -d

def compute_payoff_basic_attack(vote, outcome, p, d, epsilon):
    if outcome == "X":
        return p + (epsilon if vote == "Y" else 0)
    else:
        return p if vote == "Y" else -d

#################################################
# 2. Redistributive Payoff Model (Definition 2.2)
#################################################
def compute_payoff_redistributive_no_attack(vote, outcome, x, M, p, d):
    if outcome == "X":
        # If outcome X wins: voters for X split losers' deposits (if any)
        return ((M - x - 1) * d + M * p) / (x + 1) if vote == "X" else -d
    else:
        # If outcome Y wins
        return (x * d + M * p) / (M - x) if vote == "Y" else -d

def compute_payoff_redistributive_attack(vote, outcome, x, M, p, d, epsilon):
    if outcome == "X":
        # If X wins under attack: Y voters get epsilon bonus
        return ((M - x - 1) * d + M * p) / (x + 1) + (epsilon if vote == "Y" else 0)
    else:
        return (x * d + M * p) / (M - x) if vote == "Y" else -d

################################################
# 3. Symbiotic Payoff Model (Definition 2.3)
################################################
def compute_payoff_symbiotic_no_attack(vote, outcome, x, M, p, d):
    if outcome == "X":
        # If X wins: each X voter gets external reward proportional to X count
        return (p * (x + 1)) / M if vote == "X" else -d
    else:
        # If Y wins: each Y voter gets external reward proportional to Y count
        return (p * (M - x)) / M if vote == "Y" else -d

def compute_payoff_symbiotic_attack(vote, outcome, x, M, p, d, epsilon):
    if outcome == "X":
        # If X wins under attack: Y voters get epsilon bonus on top of external reward
        return (p * (x + 1)) / M + (epsilon if vote == "Y" else 0)
    else:
        return (p * (M - x)) / M if vote == "Y" else -d

######################################################
# 4. A single function to compute the average payoff
######################################################
def compute_average_payoff(model,
                           basic_no_attack=None,
                           basic_attack=None,
                           redis_no_attack=None,
                           redis_attack=None,
                           sym_no_attack=None,
                           sym_attack=None):
    # Determine outcome from model's recorded votes
    outcome = "X" if model.votes.get("X", 0) >= model.votes.get("Y", 0) else "Y"
    M = model.num_jurors
    p = model.p
    d = model.d
    epsilon = model.bribe_amount if model.bribe_amount > 0 else 0

    total_payoff = 0
    for juror in model.selected_jurors:
        vote = juror.vote
        x_other = model.votes.get("X", 0) - (1 if vote == "X" else 0)
        if basic_no_attack is not None:
            pay = basic_no_attack(vote, outcome, p, d)
        elif basic_attack is not None:
            pay = basic_attack(vote, outcome, p, d, epsilon)
        elif redis_no_attack is not None:
            pay = redis_no_attack(vote, outcome, x_other, M, p, d)
        elif redis_attack is not None:
            pay = redis_attack(vote, outcome, x_other, M, p, d, epsilon)
        elif sym_no_attack is not None:
            pay = sym_no_attack(vote, outcome, x_other, M, p, d)
        elif sym_attack is not None:
            pay = sym_attack(vote, outcome, x_other, M, p, d, epsilon)
        else:
            pay = 0
        total_payoff += pay
    return total_payoff / M
