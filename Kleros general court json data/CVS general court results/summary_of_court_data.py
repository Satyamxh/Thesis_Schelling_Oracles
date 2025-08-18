import pandas as pd
from pathlib import Path

# Directory with your CSV files (same as this script)
folder = Path(__file__).parent

all_cases = []

# Read each case CSV
for csv_file in folder.glob("case_*.csv"):
    df = pd.read_csv(csv_file)
    total_jurors = df['Total Jurors'].iloc[0]
    x_votes = df['Vote Count'].iloc[0]  # First row = X
    y_votes = df['Vote Count'].iloc[1]  # Second row = Y

    all_cases.append({
        "case_id": csv_file.stem,
        "Total Jurors": total_jurors,
        "X_votes": x_votes,
        "Y_votes": y_votes
    })

# Combine into one DataFrame
df_all = pd.DataFrame(all_cases)

# Save all cases combined
df_all.to_csv(folder / "combined_cases.csv", index=False)

# Get highest juror count cases
max_jurors = df_all['Total Jurors'].max()
highest_cases = df_all[df_all['Total Jurors'] == max_jurors]
highest_cases.to_csv(folder / "highest_juror_cases.csv", index=False)

# Stats
x_votes_mean = df_all['X_votes'].mean()
x_votes_std = df_all['X_votes'].std()

print(f"Mean X votes: {x_votes_mean:.2f}")
print(f"Std of X votes: {x_votes_std:.2f}")
print("\nFull stats:\n", df_all.describe())
