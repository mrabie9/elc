import numpy as np
from TrainValTest import accuracy

def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)

def maxprob_uncertainty(p):
    # Lower is better; convert to "uncertainty" where higher means worse:
    return 1.0 - p.max(axis=1)

def selective_metrics(y_true, y_pred, u, tau):
    accept = u <= tau
    cov = accept.mean()
    if cov == 0:
        return {"coverage": 0.0, "sel_acc": np.nan, "sel_risk": np.nan}
    sel_acc = (y_pred[accept] == y_true[accept]).mean()
    sel_risk = 1.0 - sel_acc
    return {"coverage": cov, "sel_acc": sel_acc, "sel_risk": sel_risk}

def sweep_thresholds(y_true, y_proba, u=None, taus=None, utility=None):
    """
    y_true: (N,) integer labels
    y_proba: (N,C) probabilities
    u: (N,) uncertainty score (higher = more uncertain). If None, uses entropy.
    taus: thresholds to try. If None, uses quantiles of u.
    utility: dict with costs/rewards, e.g. {"correct": +1, "error": -5, "abstain": -0.5}
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = y_proba.argmax(axis=1) if y_proba.ndim > 1 else y_proba

    if y_true.size == 0:
        out = {
            "tau": np.array([]),
            "coverage": np.array([]),
            "sel_acc": np.array([]),
            "sel_risk": np.array([]),
        }
        if utility is not None:
            out["utility"] = np.array([])
        return out

    if u is None:
        u = entropy(y_proba)
    else:
        u = np.asarray(u, dtype=np.float64)

    if u.size != y_true.size:
        raise ValueError(f"Uncertainty array length {u.size} does not match number of samples {y_true.size}.")

    if taus is None:
        taus = np.quantile(u, np.linspace(0, 1, 10001)) if u.size > 0 else np.array([])
    else:
        taus = np.asarray(taus)

    if taus.size == 0:
        out = {
            "tau": np.array([]),
            "coverage": np.array([]),
            "sel_acc": np.array([]),
            "sel_risk": np.array([]),
        }
        if utility is not None:
            out["utility"] = np.array([])
        return out

    coverages, sel_accs, sel_risks, utils = [], [], [], []
    for t in taus:
        accept = u <= t
        cov = float(accept.mean())
        if cov > 0:
            sel_acc = accuracy(y_true[accept], y_pred[accept]) #float(np.mean( ==))
            sel_risk = 1.0 - sel_acc
        else:
            sel_acc = np.nan
            sel_risk = np.nan

        coverages.append(cov); sel_accs.append(sel_acc); sel_risks.append(sel_risk)

        if utility is not None:
            correct_mask = np.logical_and(accept, y_pred == y_true)
            error_mask = np.logical_and(accept, y_pred != y_true)
            abstain = (~accept).sum()
            correct = correct_mask.sum()
            error = error_mask.sum()
            total = len(y_true)
            util = (utility["correct"]*correct +
                    utility["error"]*error +
                    utility["abstain"]*abstain) / total
            utils.append(util)

    out = {
        "tau": np.array(taus),
        "coverage": np.array(coverages),
        "sel_acc": np.array(sel_accs),
        "sel_risk": np.array(sel_risks),
    }
    if utility is not None:
        out["utility"] = np.array(utils)
    return out

import numpy as np

def choose_tau(results, target_coverage=None, max_risk=None, maximize_utility=False):
    cov, risk = results["coverage"], results["sel_risk"]
    if cov.size == 0:
        return {k: np.nan for k in results}

    idx = None

    if maximize_utility and "utility" in results:
        idx = np.nanargmax(results["utility"])

    elif target_coverage is not None:
        # Match a specific coverage (float-safe). If none match, choose closest.
        close = np.isclose(cov, target_coverage, rtol=1e-5, atol=1e-8)

        if np.any(close):
            # Among exact matches, pick the one with smallest risk
            candidates = np.where(close)[0]
            idx = candidates[np.nanargmin(risk[candidates])]
        else:
            # No exact match: choose coverage closest to target; break ties by lowest risk
            diffs = np.abs(cov - target_coverage)
            min_diff = np.nanmin(diffs)
            candidates = np.where(np.isclose(diffs, min_diff, rtol=0, atol=1e-12))[0]
            idx = candidates[np.nanargmin(risk[candidates])]

    elif max_risk is not None:
        # Keep this branch unchanged: largest coverage with risk ≤ max_risk
        mask = risk <= max_risk
        if mask.any():
            idx = np.nanargmax(np.where(mask, cov, -np.inf))

    if idx is None:
        # Fallback: maximize accuracy*coverage (balanced)
        score = (1.0 - risk) * cov
        idx = np.nanargmax(score)

    return {k: v[idx] for k, v in results.items()}


import numpy as np

def _to_numpy(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def top2_acc_rejected(
    y_true, 
    u, 
    tau, 
    y_proba=None,      # (N,C) predictive probs
    y_proba_mc=None,   # (N,S,C) predictive probs from Bayesian layer (samples)
):
    """
    Compute top-2 accuracy on rejected samples (u > tau).
    If y_proba_mc is provided, uses its mean across samples as the predictive mean.
    
    Returns:
      {
        'coverage': float,
        'selective_accuracy': float,
        'n_rejected': int,
        'top2_accuracy_rejected': float,
        'contains_true_rate_rejected_k2': float,  # alias
        'avg_conf_top2_rejected': float           # optional diagnostic
      }
    """
    y_true = _to_numpy(y_true).astype(int)
    u = _to_numpy(u)

    if y_proba_mc is not None:
        P = _to_numpy(y_proba_mc).mean(axis=1)     # (N,C) predictive mean
    elif y_proba is not None:
        P = _to_numpy(y_proba)
    else:
        raise ValueError("Provide either y_proba (N,C) or y_proba_mc (N,S,C).")

    if P.ndim != 2:
        raise ValueError(f"Expected probs shape (N,C) after reduction, got {P.shape}")
    N, C = P.shape

    accepted = (u <= tau)
    coverage = float(accepted.mean())

    # selective accuracy on accepted (single-label argmax correctness)
    sel_acc = float(np.nan)
    if accepted.any():
        y_hat = P.argmax(axis=1)
        sel_acc = float((y_hat[accepted] == y_true[accepted]).mean())

    # rejected indices
    rej_idx = np.flatnonzero(~accepted)
    n_rej = int(rej_idx.size)
    if n_rej == 0:
        return {
            "coverage": coverage,
            "selective_accuracy": sel_acc,
            "n_rejected": 0,
            "top2_accuracy_rejected": float("nan"),
            "contains_true_rate_rejected_k2": float("nan"),
            "avg_conf_top2_rejected": float("nan"),
        }

    # top-2 correctness on rejected
    P_rej = P[rej_idx]                # (R,C)
    y_rej = y_true[rej_idx]           # (R,)

    # indices of top-2 per row
    top2_idx = np.argpartition(-P_rej, kth=1, axis=1)[:, :2]  # unordered top-2
    # ensure they are ordered by prob (optional)
    row = np.arange(top2_idx.shape[0])[:, None]
    order = np.argsort(-P_rej[row, top2_idx], axis=1)
    top2_idx = top2_idx[row, order]    # (R,2) ordered

    # contains-true (k=2)
    contains_true_k2 = (top2_idx == y_rej[:, None]).any(axis=1)
    top2_acc_rej = float(contains_true_k2.mean())

    # optional diagnostic: average total probability mass of the top-2 set
    avg_conf_top2 = float(P_rej[row, top2_idx].sum(axis=1).mean())

    return {
        "coverage": coverage,
        "selective_accuracy": sel_acc,
        "n_rejected": n_rej,
        "top2_accuracy_rejected": top2_acc_rej,
        "contains_true_rate_rejected_k2": top2_acc_rej,
        "avg_conf_top2_rejected": avg_conf_top2,
    }
