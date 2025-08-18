import json
from collections import Counter
import pandas as pd
import os
from pathlib import Path

class DisputeParser:
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.rounds = self.data.get("rounds", [])
        self.filepath = filepath

    def get_metadata(self):
        return {
            "dispute_id": self.data.get("id"),
            "current_ruling": self.data.get("currentRulling"),
            "start_time": self.data.get("startTime")
        }

    def get_final_round_summary(self):
        if not self.rounds:
            return None
        final_round = self.rounds[0]
        choices = [str(v.get("choice")) for v in final_round.get("votes", [])
                   if str(v.get("choice")) in ("1", "2")]
        count = Counter(choices)
        votes_1 = count.get("1", 0)
        votes_2 = count.get("2", 0)
        total = votes_1 + votes_2
        if total == 0:
            return None
        
        # --- tie handling (1 vs 2) ---
        if votes_1 == votes_2 and total > 0:
            x_votes, y_votes = votes_1, votes_2
            x_pct = y_pct = round(100 * votes_1 / total, 2)
            return {
                "X_votes": x_votes,
                "Y_votes": y_votes,
                "X_percent": x_pct,
                "Y_percent": y_pct,
                "X_is": "Tie",
                "majority_choice": "Tie",
                "total_votes": total
            }

        if votes_1 > 0 and votes_2 == 0:
            x_votes, y_votes = votes_1, 0
            majority = "1"
        elif votes_2 > 0 and votes_1 == 0:
            x_votes, y_votes = votes_2, 0
            majority = "2"
        else:
            majority = "1" if votes_1 > votes_2 else "2"
            minority = "2" if majority == "1" else "1"
            x_votes = count.get(majority, 0)
            y_votes = count.get(minority, 0)
        x_pct = round(100 * x_votes / total, 2)
        y_pct = round(100 * y_votes / total, 2)

        return {
            "X_votes": x_votes,
            "Y_votes": y_votes,
            "X_percent": x_pct,
            "Y_percent": y_pct,
            "X_is": "Tie" if majority == "Tie" else ("Yes" if majority == "2" else "No"),
            "majority_choice": majority,
            "total_votes": total
        }

    def export_final_round_to_csv(self, output_path):
        final = self.get_final_round_summary()
        if not final:
            return False  # nothing to export

        df = pd.DataFrame({
            "Vote Count": [final["X_votes"], final["Y_votes"]],
            "Total Jurors": [final["total_votes"], final["total_votes"]],
            "Ratio": [round(final["X_percent"] / 100, 2), round(final["Y_percent"] / 100, 2)]
        }, index=["X", "Y"])

        df.to_csv(output_path)  # overwrites by default
        return True


def batch_export(
    json_dir=None, # put this file into the same directory as the json court data
    output_folder_name="CVS general court results"
):    
    if json_dir is None:
        json_dir = Path(__file__).parent
    else:
        json_dir = Path(json_dir)
    out_dir = json_dir / output_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("dispute*.json"))
    done, skipped = 0, 0

    skipped_no_votes = []
    errors = []

    for jf in json_files:
        try:
            parser = DisputeParser(jf)
            # prefer ID from file, fallback to filename stem
            meta = parser.get_metadata()
            dispute_id = meta.get("dispute_id") or jf.stem.replace("dispute", "")
            csv_name = f"case_{dispute_id}.csv"
            out_path = out_dir / csv_name

            final = parser.get_final_round_summary()
            if not final:  # <-- needs to be indented here
                skipped += 1
                skipped_no_votes.append(jf.name)
                print(f" Skipped {jf.name}: no valid votes in final round")
                continue

            ok = parser.export_final_round_to_csv(out_path)
            done += 1
            print(f" Saved {out_path.name}")

        except Exception as e:
            skipped += 1
            errors.append((jf.name, str(e)))
            print(f" Error on {jf.name}: {e}")

    if skipped_no_votes:
        print("\nSkipped (no valid votes):")
        for name in skipped_no_votes:
            print(f" - {name}")

    if errors:
        print("\nErrors:")
        for name, msg in errors:
            print(f" - {name}: {msg}")

    print("\n=== Summary ===")
    print(f"Processed: {len(json_files)}")
    print(f"Saved CSVs: {done}")
    print(f"Skipped/Errors: {skipped}")
    print(f"Output folder: {out_dir}")

if __name__ == "__main__":
    batch_export()
