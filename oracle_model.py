import mesa
from mesa.time import RandomActivation
from juror import Juror
from attacker import Attacker
import random

class SchellingOracleModel(mesa.Model):
    """Schelling Point Oracle System that mirrors Kleros."""

    def __init__(self, num_jurors=5, total_jurors=20, enable_attacks=True, 
                 bribe_amount=0, bribe_acceptance_prob=0, honesty_level=0.8,
                 rationality=0.7, bribed_juror_ratio=0):
        super().__init__()
        self.num_jurors = num_jurors
        self.total_jurors = total_jurors
        self.enable_attacks = enable_attacks
        self.bribe_amount = bribe_amount
        self.bribe_acceptance_prob = bribe_acceptance_prob
        self.honesty_level = honesty_level
        self.rationality = rationality
        self.bribed_juror_ratio = bribed_juror_ratio

        self.votes = {"X": 0, "Y": 0}
        self.schedule = RandomActivation(self)

        self.selected_jurors = []
        for i in range(num_jurors):
            juror = Juror(i, self, stake=random.randint(5, 20), belief=random.choice(["X", "Y"]),
                          honesty_level=honesty_level, rationality=rationality)
            self.selected_jurors.append(juror)
            self.schedule.add(juror)

        self.attacker = None if not enable_attacks else Attacker(
            self, bribe_amount=bribe_amount, bribe_acceptance_prob=bribe_acceptance_prob, bribed_juror_ratio=bribed_juror_ratio
        )

    def step(self):
        self.schedule.step()

    def collect_vote(self, vote):
        if vote in self.votes:
            self.votes[vote] += 1
        else:
            self.votes[vote] = 1

def run_baseline_simulation(num_jurors, total_jurors, beliefs, honesty_level, rationality):
    """Runs a single simulation without an attacker to establish the natural vote outcome."""
    model = SchellingOracleModel(
        num_jurors=num_jurors, 
        total_jurors=total_jurors, 
        enable_attacks=False,
        honesty_level=1.0,  # fixed value for baseline
        rationality=0  
    )
    for i, juror in enumerate(model.selected_jurors):
        juror.belief = beliefs[i]
        # Leave the baseline honesty and rationality as defined in the model
    model.step()
    return model.votes

def run_attack_simulation(num_jurors, total_jurors, beliefs, honesty_level, rationality, 
                          bribe_amount, bribe_acceptance_prob, bribed_juror_ratio):
    """Runs a single simulation with an attacker attempting a p+ε attack."""
    model = SchellingOracleModel(
        num_jurors=num_jurors, 
        total_jurors=total_jurors, 
        enable_attacks=True,
        bribe_amount=bribe_amount, 
        bribe_acceptance_prob=bribe_acceptance_prob, 
        honesty_level=honesty_level,  
        rationality=rationality,  
        bribed_juror_ratio=bribed_juror_ratio
    )
    for i, juror in enumerate(model.selected_jurors):
        juror.belief = beliefs[i]
        juror.honesty_level = honesty_level
        juror.rationality = rationality
    model.step()
    return model.votes

def was_attack_successful(baseline_votes, attack_votes):
    """Compares votes between the baseline and attack model to determine how many jurors switched their vote."""
    return sum(abs(baseline_votes[key] - attack_votes.get(key, 0)) for key in baseline_votes)
