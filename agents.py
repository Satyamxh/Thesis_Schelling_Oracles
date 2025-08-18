import random
import math

class Juror:
    """Represents a juror in the Schelling oracle simulation.
    
    Attributes:
        lambda_qre (float): Sensitivity to utility differences (QRE parameter).
        noise (float): Standard deviation of noise in payoff estimation.
        belief (str): The juror's belief about the true outcome ("A" or "B").
        vote (str): The vote cast by this juror ("X" or "Y").
        p_vote_X (float): Probability of voting "X" based on QRE.
    """
    def __init__(self, lambda_qre: float, noise: float):
        self.lambda_qre = lambda_qre  # QRE sensitivity (formerly rationality)
        self.noise = noise            
        self.belief = None
        self.vote = None
        self.p_vote_X = None

    def decide_vote(self, exp_payoff_X: float, exp_payoff_Y: float) -> str:
        """
        Decide the vote ("X" or "Y") using quantal response (logit choice rule)
        based on noisy perceived utilities.
        """

        # Add perception noise
        perceived_X = exp_payoff_X + random.gauss(0, self.noise)
        perceived_Y = exp_payoff_Y + random.gauss(0, self.noise)

        # Logit QRE probability
        lambda_ = self.lambda_qre
        exp_lambda_X = math.exp(lambda_ * perceived_X)
        exp_lambda_Y = math.exp(lambda_ * perceived_Y)
        self.p_vote_X = exp_lambda_X / (exp_lambda_X + exp_lambda_Y)

        # Sample vote
        self.vote = "X" if random.random() < self.p_vote_X else "Y"
        return self.vote