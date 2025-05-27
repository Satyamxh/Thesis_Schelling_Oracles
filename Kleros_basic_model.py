import random

class Juror:
    def __init__(self, stake, truth_value):
        """
        stake: the amount of tokens the juror has
        truth_value: the juror's internal belief (e.g., True or False)
        """
        self.stake = stake
        self.truth_value = truth_value
        self.vote = None

    def decide_vote(self):
        # In a basic model, the juror votes based on its truth_value
        self.vote = self.truth_value

def simulate_round(jurors, reward=1, penalty=0.5):
    # Each juror makes their decision
    for juror in jurors:
        juror.decide_vote()

    # Collect votes
    votes = [juror.vote for juror in jurors]
    # Determine the majority vote
    majority_vote = max(set(votes), key=votes.count)

    # Apply the payoff matrix:
    # - Jurors who voted with the majority receive a reward (e.g., +1)
    # - Jurors who voted against the majority incur a penalty (e.g., -0.5)
    for juror in jurors:
        if juror.vote == majority_vote:
            juror.stake += reward
        else:
            juror.stake -= penalty

    return majority_vote

# Example: Create 100 jurors with a random true belief (True/False) and initial stake of 10 tokens.
num_jurors = 100
jurors = [Juror(stake=10, truth_value=random.choice([True, False])) for _ in range(num_jurors)]

# Run a single simulation round.
majority_vote = simulate_round(jurors)
print("Majority vote:", majority_vote)

# Print average stake after the round to see overall rewards/penalties.
average_stake = sum(juror.stake for juror in jurors) / num_jurors
print("Average stake after simulation:", average_stake)
