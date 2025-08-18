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
        final_round = self.rounds[-1]
        choices = [v["choice"] for v in final_round.get("votes", []) 
                   if v.get("voted") and v.get("choice") in ("1", "2")]
        count = Counter(choices)
        total = count["1"] + count["2"]
        if total == 0:
            return None

        majority = "1" if count["1"] > count["2"] else "2"
        minority = "2" if majority == "1" else "1"
        x_votes = count[majority]
        y_votes = count[minority]
        x_pct = round(100 * x_votes / total, 2)
        y_pct = round(100 * y_votes / total, 2)

        return {
            "X_votes": x_votes,
            "Y_votes": y_votes,
            "X_percent": x_pct,
            "Y_percent": y_pct,
            "X_is": "Yes" if majority == "2" else "No",
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
    json_dir=r"C:\Users\Satyam\OneDrive\Desktop\Satyam\Master's Thesis\python code\Kleros 2.0",
    output_folder_name="CVS general court results"
):
    json_dir = Path(json_dir)
    out_dir = json_dir / output_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("dispute*.json"))
    done, skipped = 0, 0

    for jf in json_files:
        try:
            parser = DisputeParser(jf)
            # prefer ID from file, fallback to filename stem
            meta = parser.get_metadata()
            dispute_id = meta.get("dispute_id") or jf.stem.replace("dispute", "")
            out_path = out_dir / f"dispute{dispute_id}.csv"

            ok = parser.export_final_round_to_csv(out_path)
            if ok:
                done += 1
                print(f" Saved {out_path.name}")
            else:
                skipped += 1
                print(f" Skipped {jf.name}: no votes in final round")
        except Exception as e:
            skipped += 1
            print(f" Error on {jf.name}: {e}")

    print("\n=== Summary ===")
    print(f"Processed: {len(json_files)}")
    print(f"Saved CSVs: {done}")
    print(f"Skipped/Errors: {skipped}")
    print(f"Output folder: {out_dir}")

if __name__ == "__main__":
    batch_export()
