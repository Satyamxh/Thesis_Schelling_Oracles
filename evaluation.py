from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np
from math import log
from typing import Iterable, Optional, Tuple, Dict
from scipy.special import gammaln
import matplotlib.pyplot as plt


# --------------------------
# Data loading / wrangling
# --------------------------

def _parse_case_id(p: Path) -> str:
    # expects filenames like "case_123.csv" or "case_1170.csv"
    m = re.search(r"case[_-]?(\w+)\.csv$", p.name, re.IGNORECASE)
    return m.group(1) if m else p.stem

def read_one_case_csv(p: Path) -> dict:
    """
    Read a single exported 2-row CSV (index ["X","Y"], columns: Vote Count, Total Jurors, Ratio]).
    Returns a compact dict with both original counts (as far as recoverable) and majority/minority views.
    """
    df = pd.read_csv(p, index_col=0)
    # Basic sanity
    req_cols = {"Vote Count", "Total Jurors", "Ratio"}
    if not req_cols.issubset(set(df.columns)):
        raise ValueError(f"{p.name}: missing expected columns {req_cols}")

    if "X" not in df.index or "Y" not in df.index:
        raise ValueError(f"{p.name}: expected index to include 'X' and 'Y'")

    x_cnt = int(df.loc["X", "Vote Count"])
    y_cnt = int(df.loc["Y", "Vote Count"])
    M = int(df.loc["X", "Total Jurors"])
    # Detect tie from equality; in your exporter, ties write equal X/Y rows
    is_tie = (x_cnt == y_cnt)
    # With majority relabeling in the exporter, "X" row is majority (except ties)
    k_majority = max(x_cnt, y_cnt)
    k_minority = min(x_cnt, y_cnt)
    share_majority = (k_majority / M) if M > 0 else np.nan

    return {
        "case_id": _parse_case_id(p),
        "file": p.name,
        "M": M,
        "k_majority": k_majority,
        "k_minority": k_minority,
        "is_tie": is_tie,
        "x_share_majority": share_majority
    }

def load_cases(folder: str | Path) -> pd.DataFrame:
    """
    Load all case_*.csv files from a folder into a tidy dataframe with one row per case.
    """
    folder = Path(folder)
    rows = []
    for p in sorted(folder.glob("case_*.csv")):
        try:
            rows.append(read_one_case_csv(p))
        except Exception as e:
            # Skip unusable files but keep going
            print(f"Skipping {p.name}: {e}")
            continue
    return pd.DataFrame(rows)


# --------------------------
# Metrics
# --------------------------

def _log_binom_pmf(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Log Binomial PMF using gammaln for stability.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = np.asarray(p, dtype=float)
    # clamp p for numerical safety
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + k * np.log(p) + (n - k) * np.log(1 - p))

def binomial_nll(df: pd.DataFrame, p_col: str, k_col: str = "k_majority", n_col: str = "M") -> float:
    """
    Standard Binomial negative log-likelihood, treating k_col as successes out of n_col with prob p_col.
    If your CSVs are majority-labeled, you likely want symmetric_majority_nll() instead.
    """
    logpmf = _log_binom_pmf(df[k_col].to_numpy(), df[n_col].to_numpy(), df[p_col].to_numpy())
    return float(-np.nansum(logpmf))

def symmetric_majority_nll(df: pd.DataFrame, p_col: str, k_col: str = "k_majority", n_col: str = "M") -> float:
    """
    NLL that is invariant to flipping labels (useful when your per-case CSV relabels majority as X).
    We transform p to p* = max(p, 1 - p) and k to k* = max(votes1, votes2).
    Ties (k = n/2) are naturally handled by the same binomial PMF.
    """
    p = df[p_col].to_numpy()
    p_star = np.maximum(p, 1 - p)
    logpmf = _log_binom_pmf(df[k_col].to_numpy(), df[n_col].to_numpy(), p_star)
    return float(-np.nansum(logpmf))

def beta_binomial_logpmf(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    Log PMF for Beta-Binomial(n, alpha, beta).
    Useful when there is overdispersion relative to Binomial.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    a, b = float(alpha), float(beta)
    return (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            + gammaln(k + a) + gammaln(n - k + b) + gammaln(a + b)
            - gammaln(a) - gammaln(b) - gammaln(n + a + b))

def beta_binomial_nll(df: pd.DataFrame, p_col: str, rho: float, k_col: str = "k_majority", n_col: str = "M") -> float:
    """
    Negative log-likelihood under a Beta-Binomial with mean p and overdispersion rho in (0,1).
    Convert (p, rho) to (alpha, beta) via:
        alpha = p * (1 - rho) / rho
        beta  = (1 - p) * (1 - rho) / rho
    """
    p = np.clip(df[p_col].to_numpy(), 1e-9, 1 - 1e-9)
    rho = float(np.clip(rho, 1e-9, 1 - 1e-9))
    alpha = p * (1 - rho) / rho
    beta = (1 - p) * (1 - rho) / rho
    logpmf = beta_binomial_logpmf(df[k_col].to_numpy(), df[n_col].to_numpy(), alpha, beta)
    return float(-np.nansum(logpmf))

def brier_score_majority(df: pd.DataFrame, p_col: str) -> float:
    """
    Brier score for majority-coded success. For ties we use y=0.5.
    """
    p = np.maximum(df[p_col].to_numpy(), 1 - df[p_col].to_numpy())
    y = np.where(df["is_tie"].to_numpy(), 0.5, 1.0)
    return float(np.nanmean((p - y) ** 2))

def ece_majority(df: pd.DataFrame, p_col: str, n_bins: int = 10) -> Tuple[float, pd.DataFrame]:
    """
    Expected Calibration Error for majority-coded success.
    Returns (ECE, bin_df) where bin_df contains per-bin stats for plotting.
    """
    p = np.maximum(df[p_col].to_numpy(), 1 - df[p_col].to_numpy())
    y = np.where(df["is_tie"].to_numpy(), 0.5, 1.0)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    bin_df = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            bin_df.append({"bin": b, "p_mean": np.nan, "y_mean": np.nan, "count": 0})
            continue
        bin_df.append({
            "bin": b,
            "p_mean": float(np.nanmean(p[mask])),
            "y_mean": float(np.nanmean(y[mask])),
            "count": int(np.sum(mask)),
        })
    bin_df = pd.DataFrame(bin_df)
    # Weighted average absolute gap
    total = bin_df["count"].sum()
    if total == 0:
        return 0.0, bin_df
    ece = float((bin_df["count"] * np.abs(bin_df["p_mean"] - bin_df["y_mean"])).sum() / total)
    return ece, bin_df

def plot_calibration(bin_df: pd.DataFrame, title: str = "Calibration (majority-coded)") -> None:
    """
    Plot a simple reliability diagram using matplotlib (no seaborn, no styles).
    """
    # Clean NaNs
    plot_df = bin_df.dropna(subset=["p_mean", "y_mean"])
    plt.figure()
    plt.plot([0.5, 1.0], [0.5, 1.0])
    plt.scatter(plot_df["p_mean"].to_numpy(), plot_df["y_mean"].to_numpy())
    plt.xlabel("Predicted majority probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.show()


# --------------------------
# Prediction alignment
# --------------------------

def merge_predictions(cases: pd.DataFrame, preds: pd.DataFrame, on: str = "case_id") -> pd.DataFrame:
    """
    Merge a cases dataframe from load_cases(...) with a predictions dataframe.
    `preds` must contain at least [on, 'p_pred'] where p_pred is the model's P(vote = 'X').
    If your model only produces p by juror count M (not per-case), merge on 'M' instead.
    """
    if on not in preds.columns:
        raise ValueError(f"preds must contain a column '{on}'")
    if "p_pred" not in preds.columns:
        raise ValueError("preds must contain a column 'p_pred' with predicted P(X)")

    out = cases.merge(preds, on=on, how="inner")
    if out.empty:
        raise ValueError("merge produced no rows — check your join key and inputs")
    return out


# --------------------------
# Tiny demonstration (doctest style)
# --------------------------
if __name__ == "__main__":
    # This block is only a smoke test with synthetic data.
    # Replace 'data_folder' with your real folder of exported case_*.csv.
    data_folder = Path("./CVS general court results")  # example
    if data_folder.exists():
        cases = load_cases(data_folder)
        # Fake predictions: say p_pred depends only on M via a simple mapping
        m_to_p = {m: 0.7 for m in cases["M"].unique()}
        preds = cases[["case_id", "M"]].copy()
        preds["p_pred"] = preds["M"].map(m_to_p).fillna(0.7)
        merged = merge_predictions(cases, preds, on="case_id")

        # Metrics
        nll_sym = symmetric_majority_nll(merged, "p_pred")
        brier = brier_score_majority(merged, "p_pred")
        ece, bin_df = ece_majority(merged, "p_pred", n_bins=6)
        print(f"Symmetric NLL: {nll_sym:.3f} | Brier: {brier:.4f} | ECE: {ece:.4f}")

        # Plot calibration
        plot_calibration(bin_df, title="Calibration demo")
    else:
        print("Demo skipped: data folder not found.")
