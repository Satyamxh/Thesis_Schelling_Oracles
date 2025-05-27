import matplotlib.pyplot as plt

def plot_results(model):
    """Visualizes the voting dynamics in the Schelling Point Oracle system."""
    data = model.datacollector.get_model_vars_dataframe()
    vote_data = {answer: data["Vote Distribution"].apply(lambda x: x[answer]).tolist() for answer in model.possible_outcomes}

    plt.figure(figsize=(8, 5))
    for answer, votes in vote_data.items():
        plt.plot(range(len(votes)), votes, label=f"Answer {answer}")

    plt.xlabel("Step")
    plt.ylabel("Number of Votes")
    plt.title("Schelling Point Oracle Voting with p+ε Attacks")
    plt.legend()
    plt.show()
