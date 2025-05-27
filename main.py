from oracle_model import SchellingOracleModel
import numpy as np
import matplotlib.pyplot as plt

def run_monte_carlo(num_simulations=5000, num_jurors=5, total_jurors=20, enable_attacks=True):
    """Runs the Schelling Oracle model multiple times and collects attack success statistics based on payoff changes."""
    attack_successes = []

    for _ in range(num_simulations):
        model = SchellingOracleModel(num_jurors=num_jurors, total_jurors=total_jurors, enable_attacks=enable_attacks)
        
        model.step()  # single-step voting (Kleros-style, votes are final)

        # attack success depends on incentive shift, not just vote count
        attack_success = False
        if enable_attacks and model.attacker:
            # extract vote counts
            votes_x = model.votes.get("X", 0)
            votes_y = model.votes.get("Y", 0)

            # compute juror payoffs (using the p+ε payoff structure)
            p = 1  # Example base payout for a coherent vote
            d = 0.5  # Example deposit loss for an incoherent vote
            epsilon = model.attacker.bribe_amount  # Bribe offered to vote Y

            payoff_x = p if votes_x > votes_y else -d  # Jurors voting X get p if X wins, else they lose d
            payoff_y = (p + epsilon) if votes_x > votes_y else p  # Jurors voting Y get p+ε if X wins, else just p

            # attack is successful if Y became the rational choice due to bribes
            if payoff_y > payoff_x:
                attack_success = True

        attack_successes.append(attack_success)

    # compute success rate
    success_rate = np.mean(attack_successes) * 100
    print(f"Attack Success Rate ({'Enabled' if enable_attacks else 'Disabled'}): {success_rate:.2f}% over {num_simulations} simulations")

    return attack_successes

# run Monte Carlo simulations for both scenarios
if __name__ == "__main__":
    num_simulations = 5000  # increased for better statistical accuracy

    results_with_attack = run_monte_carlo(num_simulations=num_simulations, enable_attacks=True)
    results_without_attack = run_monte_carlo(num_simulations=num_simulations, enable_attacks=False)

    # plot results side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(results_without_attack, bins=2, edgecolor='black', alpha=0.7)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Attack Failed', 'Attack Succeeded'])
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f"Monte Carlo Simulation Without p+ε Attacks ({num_simulations} Runs)")

    axes[1].hist(results_with_attack, bins=2, edgecolor='black', alpha=0.7)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Attack Failed', 'Attack Succeeded'])
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f"Monte Carlo Simulation With p+ε Attacks ({num_simulations} Runs)")

    plt.tight_layout()
    plt.show()
