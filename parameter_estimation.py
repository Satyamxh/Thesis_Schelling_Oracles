import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import tensor
from torch.special import gammaln
import pandas as pd
from pathlib import Path
import torch.nn.functional as F

# === PARAMETERS ===
CASES_DIR = Path(__file__).parent / "Kleros general court json data" / "CVS general court results"
P_VAL = 33.40
D_VAL = 99.49

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- hyperparameters (start tiny; you can 2x if collapse persists)
USE_PINN = True
GAMMA_FP_START, GAMMA_FP_END, ANNEAL_STEPS = 0.0, 0.3, 600   # QRE fixed-point weight
W_BETA, W_LAMBDA, W_ENT = 0.05, 0.01, 0.02                   # x prior, lambda prior, early-entropy

def gamma_fp(epoch: int):
    t = min(max(epoch, 0), ANNEAL_STEPS)
    return GAMMA_FP_START + (GAMMA_FP_END - GAMMA_FP_START) * (t / ANNEAL_STEPS)

def beta_prior_penalty(x, alpha=None, beta=None):
    if alpha is None: alpha = ALPHA_X
    if beta  is None: beta  = BETA_X
    eps = 1e-6
    return -((alpha-1)*torch.log(x.clamp_min(eps))
             + (beta-1)*torch.log((1-x).clamp_min(eps))).mean()

def safe_logit(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    z = z.clamp(min=eps, max=1 - eps)
    return torch.log(z) - torch.log(1 - z)

def two_sided_binom_loglik(m, n, q):
    l1 = torch_binom_logpmf(m,     n, q)
    l2 = torch_binom_logpmf(n - m, n, q)
    return torch.logsumexp(torch.stack([l1, l2], dim=0), dim=0)

W_EDGE = 0.01  # small; tune 0.005–0.02
def edge_barrier(x, eps=1e-6):
    t = x.clamp(eps, 1-eps)
    return -torch.log(t * (1 - t)).mean()  # discourages x→{0,1} but doesn't pick a center

def payoff_redistrib_torch(vote_is_X: torch.Tensor,
                           outcome_is_X: torch.Tensor,
                           x_votes_for_X: torch.Tensor,
                           M: int,
                           p: float, d: float) -> torch.Tensor:
    # Clamp to avoid divide-by-zero in (M - x_votes_for_X)
    x_votes_for_X = x_votes_for_X.clamp(min=1e-6, max=M - 1e-6)

    # outcome and vote as strings for original functions is NOT differentiable
    # Replace with differentiable logic here, without .item() or int()
    # We'll vectorize: both outcomes and votes as tensor masks

    is_outcome_X = outcome_is_X.to(torch.float32)
    is_vote_X = vote_is_X.to(torch.float32)

    # Replicate compute_payoff_redistributive_no_attack directly in differentiable torch:
    # payoff if you vote Y:
    payoff_Y_if_outcome_X = -d
    payoff_Y_if_outcome_Y = (x_votes_for_X * d + M * p) / (M - x_votes_for_X)

    # payoff if you vote X:
    payoff_X_if_outcome_X = ( (M - x_votes_for_X) * d + M * p ) / (M - (M - x_votes_for_X))
    payoff_X_if_outcome_Y = -d

    payoff_if_voteX = is_outcome_X * payoff_X_if_outcome_X + (1 - is_outcome_X) * payoff_X_if_outcome_Y
    payoff_if_voteY = is_outcome_X * payoff_Y_if_outcome_X + (1 - is_outcome_X) * payoff_Y_if_outcome_Y

    return is_vote_X * payoff_if_voteX + (1 - is_vote_X) * payoff_if_voteY


def torch_binom_logpmf(m, n, q):
    # m,n,q are torch tensors (same shape)
    # log C(n,m) = lgamma(n+1) - lgamma(m+1) - lgamma(n-m+1)
    return (
        gammaln(n + 1.0) - gammaln(m + 1.0) - gammaln(n - m + 1.0)
        + m * torch.log(q.clamp(1e-12, 1-1e-12))
        + (n - m) * torch.log((1 - q).clamp(1e-12, 1-1e-12))
    )

def qre_choice_prob_torch_vec(num_jurors_vec, p, d, lambda_qre_vec, x_mean_vec):
    device = lambda_qre_vec.device
    # make sure num_jurors_vec is long/int on the right device
    num_jurors_vec = num_jurors_vec.to(device).long()

    B = num_jurors_vec.shape[0]
    q_out = torch.zeros(B, device=device)

    for M_unique in torch.unique(num_jurors_vec):
        mask = (num_jurors_vec == M_unique)
        idxs = mask.nonzero(as_tuple=True)[0]

        M = int(M_unique.item())
        majority_needed = M // 2 + 1
        others = M - 1

        # k: [K]
        k = torch.arange(0, others + 1, dtype=torch.float32, device=device)  # [K]

        # subset params: [B_sub]
        xm_sub = x_mean_vec[idxs].to(device)
        lam_sub = lambda_qre_vec[idxs].to(device)

        # make shapes [B_sub, K]
        K = k.shape[0]
        Bsub = xm_sub.shape[0]
        k_broadcast = k.unsqueeze(0).expand(Bsub, K)                      # [B_sub, K]
        n_others = torch.full((Bsub, 1), float(others), device=device)    # [B_sub, 1]
        x_broadcast = xm_sub.unsqueeze(1).clamp(1e-12, 1-1e-12)           # [B_sub, 1]

        # Binomial pmf(k | n=others, p=xm_sub) as [B_sub, K]
        logpmf = (
            gammaln(n_others + 1.0)
            - gammaln(k_broadcast + 1.0)
            - gammaln(n_others - k_broadcast + 1.0)
            + k_broadcast * torch.log(x_broadcast)
            + (n_others - k_broadcast) * torch.log(1.0 - x_broadcast)
        )
        pmf = torch.exp(logpmf)  # [B_sub, K]

        # votes for X if I vote X/Y : [K]
        votes_X_ifX = k + 1.0
        votes_X_ifY = k

        # outcomes as float masks [K]
        out_ifX_is_X = (votes_X_ifX >= majority_needed).float()
        out_ifY_is_X = (votes_X_ifY >= majority_needed).float()

        # payoffs for each k (shape [K]); payoff_redistrib_torch returns same shape as masks
        payX_k = payoff_redistrib_torch(
            vote_is_X=torch.ones_like(out_ifX_is_X),
            outcome_is_X=out_ifX_is_X,
            x_votes_for_X=votes_X_ifX,
            M=M, p=p, d=d
        )  # [K]

        payY_k = payoff_redistrib_torch(
            vote_is_X=torch.zeros_like(out_ifY_is_X),
            outcome_is_X=out_ifY_is_X,
            x_votes_for_X=votes_X_ifY,
            M=M, p=p, d=d
        )  # [K]

        # broadcast payoffs to [B_sub, K] to weight by pmf
        payX = payX_k.unsqueeze(0).expand(Bsub, K)  # [B_sub, K]
        payY = payY_k.unsqueeze(0).expand(Bsub, K)  # [B_sub, K]

        exp_payoff_X = (pmf * payX).sum(dim=1)  # [B_sub]
        exp_payoff_Y = (pmf * payY).sum(dim=1)  # [B_sub]

        q_sub = torch.sigmoid(lam_sub * (exp_payoff_X - exp_payoff_Y))  # [B_sub]
        q_out[idxs] = q_sub

    return q_out

# === Neural Network ===
class GlobalQREHead(nn.Module):
    def __init__(self, log10_lambda_init=-0.3, x_init=0.5):
        super().__init__()
        self.log10_lambda = nn.Parameter(torch.tensor(float(log10_lambda_init)))
        # param for x in (0,1)
        self.x_logit = nn.Parameter(torch.tensor(float(math.log(x_init/(1-x_init)))))

    def forward(self, x):
        # x is ignored; we just broadcast global params per batch
        B = x.shape[0]
        loglam = self.log10_lambda.expand(B)              # shape [B]
        lam    = torch.exp(loglam * math.log(10.0))       # λ > 0
        xhat   = torch.sigmoid(self.x_logit).expand(B)    # (0,1), shape [B]
        return lam, xhat, loglam

class ByMHead(nn.Module):
    def __init__(self, ax=0.0, bx=0.0, al=-1.0, bl=0.0):
        super().__init__()
        self.ax = nn.Parameter(torch.tensor(ax))
        self.bx = nn.Parameter(torch.tensor(bx))
        self.al = nn.Parameter(torch.tensor(al))  # log10 λ intercept
        self.bl = nn.Parameter(torch.tensor(bl))

    def forward(self, X):  # X[:,0] = log(M+1)
        logM = X[:,0]
        x = torch.sigmoid(self.ax + self.bx * logM)
        log10lam = self.al + self.bl * logM
        lam = torch.exp(log10lam * math.log(10.0))
        return lam, x, log10lam

# === Load Data ===
data_list = []

def read_case_counts(csv_path: Path):
    # restore "X"/"Y" labels that were written as the CSV index
    df = pd.read_csv(csv_path, index_col=0)

    # normalize index just in case (strip/upper)
    df.index = df.index.astype(str).str.strip().str.upper()

    # Handle the two formats:
    if "ROUND" in (c.upper() for c in df.columns):
        # multi-round CSVs: keep your original behavior (earliest round)
        first_round = df.sort_values([c for c in df.columns if c.lower() == "round"][0]).iloc[0]
        n = int(first_round["Total Jurors"])
        m = int(first_round["Vote Count"])
        return n, m

    # 2-row summary CSVs: always take the row explicitly labeled "X" (or "1")
    key = "X" if "X" in df.index else ("1" if "1" in df.index else df.index[0])
    n = int(df.loc[key, "Total Jurors"])
    m = int(df.loc[key, "Vote Count"])
    return n, m

for csv_path in CASES_DIR.glob("case_*.csv"):
    n, m = read_case_counts(csv_path)
    data_list.append((n, m))

if not data_list:
    raise RuntimeError(f"No case_*.csv files found in {CASES_DIR}")

# --- Build bootstrap training data (synthetic) ---
# Keep *all* real cases for test; train only on bootstrapped draws
BOOT_SAMPLES_PER_CASE = 50  # adjust as needed
rng = np.random.default_rng(42)

# Load your batch simulation CSV that has n_jurors, vote_X_count, lambda, x
BATCH_RESULTS_PATH = Path(__file__).parent / "batch_results.csv"
batch_df = (
    pd.read_csv(
        BATCH_RESULTS_PATH,
        usecols=["Number of Jurors", "lambda_qre", "x_mean"],
        dtype={"Number of Jurors": "int32", "lambda_qre": "float32", "x_mean": "float32"},
        engine="c",
        memory_map=True
    )
    .rename(columns={
        "Number of Jurors": "n_jurors",
        "lambda_qre": "lambda",
        "x_mean": "x",
    })
)

train_boot = []   # list of (n_j, m_j, lambda_true, x_true)
test_real  = []   # list of (n_j, m_j) real pairs (the originals)

for (n, m) in data_list:
    s = m / n

    # Filter batch simulation results for this number of jurors
    batch_subset = batch_df[batch_df["n_jurors"] == n]
    if batch_subset.empty:
        continue  # skip if no sim data for this n

    # Bootstrap vote counts from the real observed proportion
    draws = rng.binomial(n=n, p=s, size=BOOT_SAMPLES_PER_CASE)

    for m_boot in draws:
        # Pick a random λ, x from the matching batch sim results
        sim_row = batch_subset.sample(
            n=1, replace=True,
            random_state=rng.integers(0, 1e6)
        ).iloc[0]
        lam_val = sim_row["lambda"]
        x_val   = sim_row["x"]

        train_boot.append((n, int(m_boot), lam_val, x_val))

    test_real.append((n, m))

# --- Compute global prior x from real cases (size-weighted) ---
sum_m = sum(m for _, m in test_real)
sum_n = sum(n for n, _ in test_real)
x_prior = (sum_m / sum_n)

mu = float(x_prior)      # e.g., ~0.6–0.65 typically
conc = 50.0              # 30–80 is a good starting band
ALPHA_X, BETA_X = mu*conc, (1-mu)*conc

# === Tensors: training from bootstrap, testing on real ===
X_train = torch.tensor(
    [[math.log(n+1), m / n] for n, m, _, _ in train_boot],
    dtype=torch.float32, device=device
)
Y_train = torch.tensor([[n, m] for n, m, _, _ in train_boot],
                       dtype=torch.float32, device=device)

# Store true λ and x (sim priors) on device
lam_true_train = torch.tensor([lam_true for _, _, lam_true, _ in train_boot],
                              dtype=torch.float32, device=device)
x_true_train   = torch.tensor([x_true   for _, _, _, x_true in train_boot],
                              dtype=torch.float32, device=device)
lam_true_log10 = torch.log10(lam_true_train.clamp(min=1e-12))

# Test tensors (real cases)
X_test  = torch.tensor([[math.log(n+1), m / n] for n, m in test_real],
                       dtype=torch.float32, device=device)
Y_test  = torch.tensor([[n, m] for n, m in test_real],
                       dtype=torch.float32, device=device)

# === Baseline grid search for best global (λ, x) on the test set ===
with torch.no_grad():
    loglam_grid = torch.linspace(-3, 1, 81, device=device)   # 10^-3 .. 10^1
    x_grid      = torch.linspace(0.05, 0.95, 181, device=device)

    n = Y_test[:,0].long(); m = Y_test[:,1]
    w = Y_test[:,0] / Y_test[:,0].mean()
    best = (float('inf'), None, None)

    for ll in loglam_grid:
        lam = torch.exp(ll * math.log(10.0))
        lam_b = lam.expand(n.shape[0])
        for x in x_grid:
            x_b = x.expand(n.shape[0])
            q = qre_choice_prob_torch_vec(n, P_VAL, D_VAL, lam_b, x_b).clamp(1e-6, 1-1e-6)

            # WITH this two-sided version:
            nll = (w * (-two_sided_binom_loglik(m, n.float(), q))).mean().item()

            if nll < best[0]:
                best = (nll, float(10**ll), float(x))
print("Best global NLL (test):", best)  # (nll, lambda*, x*)
best_lambda, best_x = best[1], best[2]

# center lambda prior at your grid-search best (already computed as best_lambda)
MU_LOG10_LAM, SIGMA_LOG10_LAM = math.log10(best_lambda), 0.75  # wide, gentle
def lambda_prior_penalty(log10lam, mu=MU_LOG10_LAM, sigma=SIGMA_LOG10_LAM):
    return ((log10lam - mu) / sigma).pow(2).mean()

def bernoulli_entropy(p, eps=1e-6):  # high near 0.5, low near 0/1
    p = p.clamp(eps, 1-eps)
    return -(p*torch.log(p) + (1-p)*torch.log(1-p))

# === DataLoader (mini-batch) ===
from torch.utils.data import TensorDataset, DataLoader

train_ds = TensorDataset(X_train, Y_train, lam_true_train, x_true_train, lam_true_log10)
BATCH_SIZE = 1024  # tune 256–4096 depending on GPU/CPU memory
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

# === Model Init ===
init_al = math.log10(best_lambda)       # log10(best_lambda)
init_ax = 0.0  # logit(0.5)
model = ByMHead(ax=init_ax, bx=0.0, al=init_al, bl=0.0).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)

# Optional: two-stage curriculum
PRETRAIN_SUP_ONLY_EPOCHS = 0   # first fit sim-prior (λ,x), then add NLL
alpha_x = 0.05     # weight for x supervision
beta_l  = 0.05     # weight for log10(λ) supervision

for epoch in range(2000):
    model.train()
    total_loss = 0.0
    for Xb, Yb, lam_true_b, x_true_b, lam_true_log10_b in train_loader:
        optimizer.zero_grad()
        lam_b, x_b, loglam_b = model(Xb)

        # QRE probabilities for this batch
        q_b = qre_choice_prob_torch_vec(Yb[:, 0].long(), P_VAL, D_VAL, lam_b, x_b)
        q_b = q_b.clamp(1e-6, 1 - 1e-6)
        # Binomial NLL (votes)
        m_b = Yb[:, 1]; n_b = Yb[:, 0]

        # importance weights ~ n (or n / mean(n) to keep scale stable)
        #w_b = n_b / n_b.mean()
        #nll_vec = -torch_binom_logpmf(m_b, n_b, q_b)     # [B]
        #nll_b = (w_b * nll_vec).mean()
        #nll_b = (-torch_binom_logpmf(m_b, n_b, q_b)).mean()
        w_b = n_b / n_b.mean()
        nll_vec = -two_sided_binom_loglik(m_b, n_b, q_b)
        nll_b   = (w_b * nll_vec).mean()

        

        if epoch < PRETRAIN_SUP_ONLY_EPOCHS:
            sup_scale = 1.0
        else:
            sup_scale = 0.0   # <- instead of 0.05 or 0.1

        # Sim-prior supervision on (λ,x)
        sup_b = sup_scale * (
            alpha_x * F.smooth_l1_loss(x_b, x_true_b) +
            beta_l  * F.smooth_l1_loss(loglam_b, lam_true_log10_b)
        )

        # (1) QRE fixed-point PINN residual with ramp + interior mask (as you had)
        if USE_PINN:
            mask = (q_b > 0.15) & (q_b < 0.85) & (x_b > 0.15) & (x_b < 0.85)
            s = gamma_fp(epoch)
            if mask.any():
                pinn_prob  = F.mse_loss(q_b[mask], x_b[mask])
                diff_logit = safe_logit(q_b[mask]) - safe_logit(x_b[mask])
                pinn_logit = F.smooth_l1_loss(diff_logit, torch.zeros_like(diff_logit), beta=0.5)
                pinn_fp    = (1.0 - s) * pinn_prob + s * pinn_logit
            else:
                pinn_fp = torch.tensor(0.0, device=q_b.device)
        else:
            pinn_fp = torch.tensor(0.0, device=q_b.device)
        s_fp = gamma_fp(epoch)
    
        # (2) soft, informative (Beta) prior on x 
        pen_x = beta_prior_penalty(x_b, ALPHA_X, BETA_X)

        # (3) mild prior on log10(lambda) around grid best
        pen_lam = lambda_prior_penalty(loglam_b)

        # (4) early entropy nudge on q, annealed to zero (linear)
        ent_weight = max(0.0, 1.0 - epoch/600.0) * W_ENT
        ent_q = -bernoulli_entropy(q_b).mean()  # NEGATIVE entropy (so minimizing loss == maximizing entropy)
    
        # Final loss (no boundaries anywhere)
        loss = nll_b + sup_b + gamma_fp(epoch)*pinn_fp \
                + W_EDGE*edge_barrier(x_b) \
                + W_LAMBDA*pen_lam + ent_weight*ent_q
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # --- eval every 200 epochs ---
    if epoch % 200 == 0:
        model.eval()
        with torch.no_grad():
            lam_t, x_t, _ = model(X_test)
            q_t = qre_choice_prob_torch_vec(Y_test[:,0].long(), P_VAL, D_VAL, lam_t, x_t)

            m_t = Y_test[:,1]; n_t = Y_test[:,0]
            w_t = n_t / n_t.mean()

            s = m_t / n_t  # recorded share for X
            mae_q     = (q_t - s).abs().mean()   
            mae_flip  = (1 - q_t - s).abs().mean()
            FLIP = (mae_flip < mae_q)  # True -> model is on the mirrored branch
    
            # For reporting and simulation, use:
            q_oriented = q_t if not FLIP else (1.0 - q_t)
            x_oriented = x_t if not FLIP else (1.0 - x_t)


            # label-invariant (two-sided) NLL
            test_nll = (w_t * (-two_sided_binom_loglik(m_t, n_t, q_t))).mean().item()

            # label-invariant MAE: distance to the closer of m/n or 1 - m/n
            mae_sym = torch.minimum((q_t - (m_t / n_t)).abs(),(q_t - (1 - m_t / n_t)).abs()).mean().item()

            # use the same relaxed mask as the final LBFGS eval
            if USE_PINN:
                mask_t = (q_t > 0.05) & (q_t < 0.95) & (x_t > 0.05) & (x_t < 0.95)
                T0, WIDTH = 700, 400
                s = max(0.0, min(1.0, (epoch - T0) / float(max(1, WIDTH))))
                if mask_t.any():
                    pinn_prob_val  = F.mse_loss(q_t[mask_t], x_t[mask_t]).item()
                    diff_logit     = safe_logit(q_t[mask_t]) - safe_logit(x_t[mask_t])
                    pinn_logit_val = F.smooth_l1_loss(diff_logit, torch.zeros_like(diff_logit), beta=0.5).item()
                    pinn_fp_val    = (1.0 - s) * pinn_prob_val + s * pinn_logit_val
                    pin_mask_cov   = mask_t.float().mean().item()
                else:
                    pinn_prob_val = pinn_logit_val = pinn_fp_val = 0.0
                    pin_mask_cov  = 0.0
            else:
                pinn_prob_val = pinn_logit_val = pinn_fp_val = 0.0
                pin_mask_cov  = 0.0


            # Diagnostics on train supervision (no grad)
            lam_pred_train, x_pred_train, loglam_pred_train = model(X_train)
            lam_mae_log = (loglam_pred_train - lam_true_log10).abs().mean().item()
            x_mae       = (x_pred_train   - x_true_train  ).abs().mean().item()

            frac_in_band = ((x_oriented >= 0.50) & (x_oriented <= 0.77)).float().mean().item()

            print(f"\n=== Epoch {epoch} ===")
            print(f"[Train]")
            print(f"  Mean Total Loss                       : {total_loss/len(train_loader):.4f}")

            print(f"[Test Performance]")
            print(f"  NLL (votes)                           : {test_nll:.4f}")
            print(f"  MAE (\U0001D45E vs \U0001D45A / \U0001D45B)                      : {mae_sym:.4f}")

            print(f"[Test Parameter Stats]")
            print(f"  Mean \u03BB                                : {lam_t.mean().item():.4f}")
            print(f"  Mean \U0001D465                                : {x_oriented.mean().item():.4f}")
            print(f"  PINN FP (blend, MSE/Huber)             : {pinn_fp_val:.4f}")
            print(f"    ├─ prob-space MSE                    : {pinn_prob_val:.4f}")
            print(f"    └─ logit-space Huber                 : {pinn_logit_val:.4f}")

            print(f"[Train Supervision Diagnostics]")
            print(f"  Train MAE (log\u2081\u2080(\u03BB) vs synthetic data): {lam_mae_log:.4f}")
            print(f"  MAE \U0001D465 vs synthetic data               : {x_mae:.4f}")
            print(f"  PINN mask coverage                     : {pin_mask_cov:.3f}")

            print(f"[Test] Share x (oriented) in [0.50,0.77] : {frac_in_band:.3f} | FLIP={bool(FLIP)}")
