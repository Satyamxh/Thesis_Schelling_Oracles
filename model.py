import random
import math
import numpy as np
from scipy.stats import binom
from typing import List, Tuple
from agents import Juror
from payoff_mechanisms import (compute_payoff_basic_attack, compute_payoff_basic_no_attack, 
                               compute_payoff_redistributive_attack, compute_payoff_redistributive_no_attack, 
                               compute_payoff_symbiotic_attack, compute_payoff_symbiotic_no_attack)

class OracleModel:
    """
    Agent-based model of the Schelling point oracle. Simulates a single dispute resolution round 
    with a panel of jurors voting on outcome "X" vs "Y", including payoff mechanism and optional attack.
    """
    def __init__(self, num_jurors: int, noise: float, lambda_qre: float, p: float, d: float, epsilon: float, 
                 payoff_type: str, attack: bool, x_mean: float):
        # Store model parameters
        self.num_jurors = num_jurors              # Number of jurors in the panel
        self.noise = noise                        # Noise standard deviation for payoff estimation (juror's imperfect perception of the payoff by adding a noise - simulate human decision - making)
        self.lambda_qre = lambda_qre
        self.p = p                                # Base reward (p)
        self.d = d                                # Deposit (stake)
        self.epsilon = epsilon                    # Bonus payoff (epsilon) given by attacker for bribery
        self.payoff_type = payoff_type            # Payoff mechanism: "Basic", "Redistributive", or "Symbiotic"
        self.attack = attack                      # Whether an attack (p+epsilon attack) is enabled
        self.x_mean = x_mean                      # focal point for estimating 'x' for symbiotic and redistributive mechanisms (set up at 50%) 
        # self.x_guess_noise = x_guess_noise        # Noise for estimating 'x' for symbiotic and redistributive mechanisms

        # Initialise the jurors
        self.jurors: List[Juror] = [Juror(lambda_qre=lambda_qre, noise=noise) for _ in range(num_jurors)]
        # Treat all jurors as the selected panel for voting
        self.selected_jurors: List[Juror] = self.jurors

        # Appending results for CVS file
        self.history_X = []
        self.history_Y = []

    def _expected_payoffs(self, juror_index: int) -> Tuple[float, float]:
        other_count = self.num_jurors - 1
        majority_needed = math.ceil(self.num_jurors / 2) 

        exp_payoff_X = 0.0
        exp_payoff_Y = 0.0
        
        # Assume: If I vote X, everyone else votes Y → outcome is Y
        # Assume: If I vote Y, everyone else votes X → outcome is X
        outcome_X = "Y"
        outcome_Y = "X"

        if self.payoff_type.lower() == "basic":
            # belief that any *other* juror votes X
            P = self.x_mean
            others = other_count
            M = majority_needed

            # P(X wins | I vote X): need k+1 ≥ M ⇒ k ≥ M−1
            p_win_ifX = 1 - binom.cdf(M-2, others, P)
            # P(X wins | I vote Y): need k ≥ M ⇒ k ≥ M
            p_win_ifY = 1 - binom.cdf(M-1, others, P)

            # terminal payoffs
            if self.attack:
                uX_x = compute_payoff_basic_attack("X","X", self.p, self.d, self.epsilon)
                uX_y = compute_payoff_basic_attack("X","Y", self.p, self.d, self.epsilon)
                uY_x = compute_payoff_basic_attack("Y","X", self.p, self.d, self.epsilon)
                uY_y = compute_payoff_basic_attack("Y","Y", self.p, self.d, self.epsilon)
            else:
                uX_x = compute_payoff_basic_no_attack("X","X", self.p, self.d)
                uX_y = compute_payoff_basic_no_attack("X","Y", self.p, self.d)
                uY_x = compute_payoff_basic_no_attack("Y","X", self.p, self.d)
                uY_y = compute_payoff_basic_no_attack("Y","Y", self.p, self.d)

            exp_payoff_X = p_win_ifX*uX_x + (1-p_win_ifX)*uX_y
            exp_payoff_Y = p_win_ifY*uY_x + (1-p_win_ifY)*uY_y

            return exp_payoff_X, exp_payoff_Y

        else:
            # x_mean-based redistributive/symbiotic logic

            # Redistributive/Symbiotic expected utility with belief P = x_mean
            P = self.x_mean

            k_values = np.arange(0, other_count + 1)
            prob_k = binom.pmf(k_values, other_count, P)

            for k, prob in zip(k_values, prob_k):
                # If I vote X, total X-votes = k + 1
                votes_X_ifX = k + 1
                outcome_ifX = "X" if votes_X_ifX >= majority_needed else "Y"

                # If I vote Y, total X-votes = k
                votes_X_ifY = k
                outcome_ifY = "X" if votes_X_ifY >= majority_needed else "Y"

                if self.attack:
                    if self.payoff_type.lower() == "redistributive":
                        payoff_X = compute_payoff_redistributive_attack("X", outcome_ifX, votes_X_ifX, self.num_jurors, self.p, self.d, self.epsilon)
                        payoff_Y = compute_payoff_redistributive_attack("Y", outcome_ifY, votes_X_ifY, self.num_jurors, self.p, self.d, self.epsilon)
                    elif self.payoff_type.lower() == "symbiotic":
                        payoff_X = compute_payoff_symbiotic_attack("X", outcome_ifX, votes_X_ifX, self.num_jurors, self.p, self.d, self.epsilon)
                        payoff_Y = compute_payoff_symbiotic_attack("Y", outcome_ifY, votes_X_ifY, self.num_jurors, self.p, self.d, self.epsilon)
                    else:
                        payoff_X = payoff_Y = 0.0
                else:
                    if self.payoff_type.lower() == "redistributive":
                        payoff_X = compute_payoff_redistributive_no_attack("X", outcome_ifX, votes_X_ifX, self.num_jurors, self.p, self.d)
                        payoff_Y = compute_payoff_redistributive_no_attack("Y", outcome_ifY, votes_X_ifY, self.num_jurors, self.p, self.d)
                    elif self.payoff_type.lower() == "symbiotic":
                        payoff_X = compute_payoff_symbiotic_no_attack("X", outcome_ifX, votes_X_ifX, self.num_jurors, self.p, self.d)
                        payoff_Y = compute_payoff_symbiotic_no_attack("Y", outcome_ifY, votes_X_ifY, self.num_jurors, self.p, self.d)
                    else:
                        payoff_X = payoff_Y = 0.0

                
                exp_payoff_X += prob * payoff_X
                exp_payoff_Y += prob * payoff_Y        
            
            return exp_payoff_X, exp_payoff_Y

    def simulate_once(self) -> Tuple[str, int, int, float, float]:
        """
        Run a single simulation round: assign juror beliefs, apply bribery if attack is enabled, 
        collect votes from all jurors, and determine the outcome.
        
        Returns:
            outcome (str): "X" or "Y" (winning outcome of this round)
            votes_for_X (int): Number of jurors who voted "X"
            votes_for_Y (int): Number of jurors who voted "Y"
        """

        votes = []
        exp_utilities_X = []   # for QRE analysis
        exp_utilities_Y = []   # for QRE analysis

        # 1. Assign random beliefs to jurors ("X" belief with probability p, else "Y")
        for juror in self.jurors:
            juror.belief = "X" if random.random() < self.p else "Y"
        
        # 2. Voting decisions
        for i, juror in enumerate(self.jurors):
            expX, expY = self._expected_payoffs(i)
            exp_utilities_X.append(expX)
            exp_utilities_Y.append(expY)

            #the following is to print expected payoffs
            # print(f"Epsilon: {self.epsilon} | Exp Payoffs: X={expX:.3f}, Y={expY:.3f}")
            
            utility_X = expX + random.gauss(0, self.noise)
            utility_Y = expY + random.gauss(0, self.noise)
            vote = juror.decide_vote(utility_X, utility_Y)
            votes.append(vote)

        # 4. Count votes
        votes_for_X = votes.count("X")
        votes_for_Y = votes.count("Y")

        # 5. Determine winning outcome (tie-break goes to "A")
        outcome = "X" if votes_for_X >= votes_for_Y else "Y"

        # Store votes in a dictionary for potential payoff analysis
        self.votes = {"X": votes_for_X, "Y": votes_for_Y}

        x_payoffs = []
        y_payoffs = []

        for i, juror in enumerate(self.jurors):
            vote = votes[i]
            k = votes.count("X") - (1 if vote == "X" else 0) # x is the number of votes (other than that of USR) for X
            
            if self.attack:
                if self.payoff_type.lower() == "basic":
                    payoff = compute_payoff_basic_attack(vote, outcome, self.p, self.d, self.epsilon)
                elif self.payoff_type.lower() == "redistributive":
                    payoff = compute_payoff_redistributive_attack(vote, outcome, k, self.num_jurors, self.p, self.d, self.epsilon)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff = compute_payoff_symbiotic_attack(vote, outcome, k, self.num_jurors, self.p, self.d, self.epsilon)
                else:
                    payoff = 0.0
            else:
                if self.payoff_type.lower() == "basic":
                    payoff = compute_payoff_basic_no_attack(vote, outcome, self.p, self.d)
                elif self.payoff_type.lower() == "redistributive":
                    payoff = compute_payoff_redistributive_no_attack(vote, outcome, k, self.num_jurors, self.p, self.d)
                elif self.payoff_type.lower() == "symbiotic":
                    payoff = compute_payoff_symbiotic_no_attack(vote, outcome, k, self.num_jurors, self.p, self.d)
                else:
                    payoff = 0.0
            
            if vote == "X":
                x_payoffs.append(payoff)
            else:
                y_payoffs.append(payoff)
        
         # Compute average QRE vote probability and utilities for this run
        self.qre_vote_probs = []

        for ux, uy in zip(exp_utilities_X, exp_utilities_Y):
            noisy_ux = ux + random.gauss(0, self.noise)
            noisy_uy = uy + random.gauss(0, self.noise)

          #  if not math.isfinite(noisy_ux) or not math.isfinite(noisy_uy):
           #     print(f"[Warning] Invalid utility values: ux={noisy_ux}, uy={noisy_uy}")

            exp_ux = math.exp(self.lambda_qre * noisy_ux)
            exp_uy = math.exp(self.lambda_qre * noisy_uy)
            prob_X = exp_ux / (exp_ux + exp_uy)
            self.qre_vote_probs.append(prob_X)
        
        avg_qre_prob_X = np.mean(self.qre_vote_probs)
        
        avg_utility_X = np.mean(x_payoffs) if x_payoffs else 0.0
        avg_utility_Y = np.mean(y_payoffs) if y_payoffs else 0.0
        
        # if no one voted for X then x_payoffs is set to 0 and same for Y
        return outcome, votes_for_X, votes_for_Y, avg_utility_X, avg_utility_Y, avg_qre_prob_X

    def run_simulations(self, num_simulations: int, progress_bar=None, status_text=None) -> dict:
        """
        Run the simulation for a given number of rounds and aggregate the results.
        Returns a dict with keys:
           - "total_runs", "outcome_counts", "attack_success_rate",
           - "average_votes_X", "average_votes_Y",
           - "history_X", "history_Y",
           - (if attack=True) "history_Y_attack", "history_Y_no_attack", "attack_effect_percent".
        """
        # Clear history lists for fresh results
        self.history_X.clear()
        self.history_Y.clear()

        self.qre_prob_X_list = []         # average QRE probability for vote = X per round
        self.utility_X_list = []          # average perceived utility for X per round
        self.utility_Y_list = []          # average perceived utility for Y per round

        vote_std_X = []                   # standard deviation of votes for X
        vote_std_Y = []                   # standard deviation of votes for Y
        payoff_std_X = []                 # standard deviation of payoffs for X
        payoff_std_Y = []                 # standard deviation of payoffs for Y

        self.qre_vote_probs_all = []      # vote probability per round 

        if self.attack:
            # Initialize lists to collect Y-votes for attacked vs. no-attack runs
            self.history_Y_attack = []
            self.history_Y_no_attack = []
            attack_effect_percent = []

        outcomes = {"X": 0, "Y": 0}
        votes_X_array = np.zeros(num_simulations, dtype=np.uint16)
        votes_Y_array = np.zeros(num_simulations, dtype=np.uint16)
        payoff_X_array = np.zeros(num_simulations, dtype=np.float32)
        payoff_Y_array = np.zeros(num_simulations, dtype=np.float32)

        if self.attack:
            # Arrays to store per-round Y-votes for attacked and baseline runs
            votes_X_attack_array = np.zeros(num_simulations, dtype=np.uint16)
            votes_Y_attack_array = np.zeros(num_simulations, dtype=np.uint16)
            votes_X_no_attack_array = np.zeros(num_simulations, dtype=np.uint16)
            votes_Y_no_attack_array = np.zeros(num_simulations, dtype=np.uint16)

        for i in range(num_simulations):
            # 1. Run the simulation with the attack enabled
            outcome, votes_X, votes_Y, avg_X_payoff, avg_Y_payoff, avg_qre_prob_X = self.simulate_once()

            if self.attack:
                # Record Y-votes from the attacked simulation
                votes_Y_attack_array[i] = votes_Y

                # 2. Run a second simulation with the same parameters but attack disabled
                self.attack = False
                _, votes_X_no, votes_Y_no, _, _, _ = self.simulate_once()
                self.attack = True  # restore attack mode

                # 3. Compute the per-round attack effect (% of jurors)
                effect = (votes_Y - votes_Y_no) / self.num_jurors * 100
                attack_effect_percent.append(effect)

                votes_X_no_attack_array[i] = votes_X_no
                votes_Y_no_attack_array[i] = votes_Y_no

            # Tally the outcome of the attacked run as usual
            outcomes[outcome] += 1
            votes_X_array[i] = votes_X
            votes_Y_array[i] = votes_Y
            payoff_X_array[i] = avg_X_payoff
            payoff_Y_array[i] = avg_Y_payoff

            self.qre_prob_X_list.append(avg_qre_prob_X)
            self.utility_X_list.append(avg_X_payoff)
            self.utility_Y_list.append(avg_Y_payoff)

            vote_std_X.append(votes_X)
            vote_std_Y.append(votes_Y)
            payoff_std_X.append(avg_X_payoff)
            payoff_std_Y.append(avg_Y_payoff)

            self.qre_vote_probs_all.append(self.qre_vote_probs)

            # Update progress UI if provided
            if progress_bar and i % 100 == 0:
                progress_bar.progress((i + 1) / num_simulations)
            if status_text and i % 100 == 0:
                status_text.text(f"Running simulation {i + 1} / {num_simulations}")

        # Store histories from the attacked run (original behavior)
        self.history_X = votes_X_array.tolist()
        self.history_Y = votes_Y_array.tolist()

        # If attack was enabled, store the additional histories
        if self.attack:
            self.history_Y_attack = votes_Y_attack_array.tolist()
            self.history_Y_no_attack = votes_Y_no_attack_array.tolist()

        # Compute summary statistics
        attack_success_rate = None
        if self.attack:
            attack_success_rate = np.mean(attack_effect_percent)
        avg_votes_X = np.mean(votes_X_array)
        avg_votes_Y = np.mean(votes_Y_array)
        self.avg_payoff_X = payoff_X_array.tolist()
        self.avg_payoff_Y = payoff_Y_array.tolist()

        std_votes_X = np.std(vote_std_X)
        std_votes_Y = np.std(vote_std_Y)
        std_payoff_X = np.std(payoff_std_X)
        std_payoff_Y = np.std(payoff_std_Y)

        self.std_votes_X = std_votes_X
        self.std_votes_Y = std_votes_Y
        self.std_payoff_X = std_payoff_X
        self.std_payoff_Y = std_payoff_Y

        # Build the results dict
        results = {

            #input parameters
            "total_runs": num_simulations,
            "juror_number": self.num_jurors,
            "p": self.p,
            "d": self.d,
            "noise": self.noise,
            "x_mean": self.x_mean,
            # "x_guess_noise": self.x_guess_noise,
            "lambda_qre": self.lambda_qre,
            "payoff_type": self.payoff_type,

            # attack input parameters
            "attack": self.attack,
            "epsilon": self.epsilon,

            # output parameters            

            "outcome_counts": outcomes,
            "average_votes_X": avg_votes_X,
            "average_votes_Y": avg_votes_Y,
            "history_X": self.history_X,
            "history_Y": self.history_Y,
            "avg_payoff_X": self.avg_payoff_X,
            "avg_payoff_Y": self.avg_payoff_Y,
            "std_votes_X": std_votes_X,
            "std_votes_Y": std_votes_Y,
            "std_payoff_X": std_payoff_X,    
            "std_payoff_Y": std_payoff_Y,
            "avg_qre_prob_X": avg_qre_prob_X,
            "qre_prob_X_list": self.qre_prob_X_list,
            "utility_X_list": self.utility_X_list,
            "utility_Y_list": self.utility_Y_list,

            # attack output parameters

            "attack_success_rate": attack_success_rate,
        }

        if self.attack:
            # Add the attack-vs-no-attack results
            results["history_Y_attack"] = self.history_Y_attack
            results["history_Y_no_attack"] = self.history_Y_no_attack
            results["attack_effect_percent"] = attack_effect_percent
            results["history_X_no_attack"] = votes_X_no_attack_array.tolist()
            results["history_Y_no_attack"] = votes_Y_no_attack_array.tolist()

        return results