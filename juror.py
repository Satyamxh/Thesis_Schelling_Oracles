import mesa
import random

class Juror(mesa.Agent):
    """A juror in the Schelling Point Oracle System with human-like decision-making."""
    
    def __init__(self, unique_id, model, stake, belief, honesty_level=0.8, rationality=0.7):
        """
        - honesty_level: Probability of voting honestly.
        - rationality: Probability of making a game-theoretic decision.
        """
        super().__init__(unique_id, model)
        self.stake = stake
        self.belief = belief
        self.vote = None
        self.honesty_level = honesty_level
        self.rationality = rationality
        self.bribed = False

    def step(self):
        """Jurors vote based on honesty, rationality, or bribery."""
        if self.vote is None:  # thiss ensures jurors vote only once
            # if bribed, check probability of accepting bribe
            if self.model.attacker and self in self.model.attacker.bribed_jurors:
                if random.random() < self.model.attacker.bribe_acceptance_prob:
                    self.vote = self.model.attacker.target_vote
                    self.bribed = True
                    self.model.collect_vote(self.vote)
                    return

            # the higher `honesty_level`, the more likely to vote true belief
            if random.random() < (1 - self.rationality):
                self.vote = self.belief

            # if the rationality is high, they try to match the expected majority
            else:
                self.vote = random.choice(["X", "Y"])

            self.model.collect_vote(self.vote)