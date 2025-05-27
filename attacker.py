import mesa
import random

class Attacker(mesa.Agent):
    """Attacker performing p+ε attacks by bribing jurors."""

    def __init__(self, model, bribe_amount=0.5, bribe_acceptance_prob=0.3, bribed_juror_ratio=0.2):
        """Initialise attacker with dynamic bribery parameters."""
        super().__init__("Attacker", model)
        self.bribe_amount = bribe_amount
        self.bribe_acceptance_prob = bribe_acceptance_prob
        self.bribed_juror_ratio = bribed_juror_ratio
        self.target_vote = None
        self.bribed_jurors = []

    def step(self):
        """Bribe jurors based on defined parameters."""
        if sum(self.model.votes.values()) == 0:
            return  # No attack on the first round

        # if attack is disabled, do nothing (this is to test for no attack as well)
        if self.model.bribed_juror_ratio == 0 or self.model.bribe_amount == 0:
            return  

        # choose a random target vote
        self.target_vote = random.choice(["X", "Y"])

        # determine how many jurors to bribe
        num_bribed = max(1, int(len(self.model.selected_jurors) * self.model.bribed_juror_ratio))
        self.bribed_jurors = random.sample(self.model.selected_jurors, k=num_bribed)

        for juror in self.bribed_jurors:
            if random.random() < self.model.bribe_acceptance_prob:
                juror.vote = self.target_vote
                juror.bribed = True