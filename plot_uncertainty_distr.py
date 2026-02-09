import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_uncertainty_density(uncertainty, correct, title="Uncertainty density by correctness"):
    u = np.asarray(uncertainty, dtype=float)
    c = np.asarray(correct, dtype=bool)

    # Keep only finite values
    mask = np.isfinite(u) & ~np.isnan(u) & (c==c)  # handles weird bools
    u, c = u[mask], c[mask]

    u_correct   = u[c]
    u_incorrect = u[~c]

    # Robust common x-range (avoid outliers skewing the view)
    lo = np.nanpercentile(u, 0.5)
    hi = np.nanpercentile(u, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(u), np.nanmax(u)
    x = np.linspace(lo, hi, 512)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=350)

    try:
        # KDE (smooth density curves)
        from scipy.stats import gaussian_kde

        # Small jitter helps when many identical values
        def kde_vals(values):
            if len(values) < 2:
                return np.zeros_like(x)
            jitter = 1e-9 * (hi - lo + 1.0)
            vals = values + np.random.default_rng(0).normal(0, jitter, size=len(values))
            kde = gaussian_kde(vals)  # Scott’s rule by default
            return kde(x)

        y1 = kde_vals(u_correct)
        y2 = kde_vals(u_incorrect)
        ax.plot(x, y1, label="Correct Predictions", linewidth=2)
        ax.plot(x, y2, label="Incorrect Predictions", linewidth=2, linestyle="--")
        print("KDE Successful")

    except Exception:
        print("KDE Unsuccessful")
        # Fallback: normalized histograms
        bins = np.linspace(lo, hi, 40)
        h1, _ = np.histogram(u_correct, bins=bins, density=True)
        h2, _ = np.histogram(u_incorrect, bins=bins, density=True)
        xc = 0.5 * (bins[1:] + bins[:-1])
        ax.step(xc, h1, where="mid", label="Correct Predictions", linewidth=2)
        ax.step(xc, h2, where="mid", label="Incorrect Predictions", linewidth=2, linestyle="--")

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

from sklearn.metrics import roc_auc_score
import numpy as np

def auc_uncertainty(u, correct):
    u = np.asarray(u, dtype=float)
    c = np.asarray(correct, dtype=bool)
    # Higher uncertainty ⇒ more likely incorrect
    if np.unique(c).size < 2:   # all-correct or all-incorrect
        return np.nan
    return roc_auc_score(~c, u)

def bootstrap_diff(u1, c1, u2, c2, n=1000, seed=0, paired=False, plot=True):
    u1 = np.asarray(u1, dtype=float); c1 = np.asarray(c1, dtype=bool)
    u2 = np.asarray(u2, dtype=float); c2 = np.asarray(c2, dtype=bool)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=float)

    for i in range(n):
        if paired and len(u1) == len(u2) == len(c1) == len(c2):
            idx = rng.integers(0, len(u1), len(u1))
            a1 = auc_uncertainty(u1[idx], c1[idx])
            a2 = auc_uncertainty(u2[idx], c2[idx])
        else:
            idx1 = rng.integers(0, len(u1), len(u1))
            idx2 = rng.integers(0, len(u2), len(u2))
            a1 = auc_uncertainty(u1[idx1], c1[idx1])
            a2 = auc_uncertainty(u2[idx2], c2[idx2])

        diffs[i] = a1 - a2
        
        # Means and percentile CIs
    def mean_ci(x):
        return np.nanmean(x), np.nanpercentile(x, [2.5, 97.5])

    auc1_m, ci1 = mean_ci(a1)
    auc2_m, ci2 = mean_ci(a2)
    _,      ci_diff = mean_ci(diffs)

    if plot:
        means = [auc1_m, auc2_m]
        ci_low  = [auc1_m - ci1[0], auc2_m - ci2[0]]
        ci_high = [ci1[1] - auc1_m, ci2[1] - auc2_m]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([0, 1], means, tick_label=["Model 1", "Model 2"])
        ax.errorbar([0, 1], means, yerr=[ci_low, ci_high], fmt='none', capsize=6, linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC")
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    ci = np.percentile(diffs, [2.5, 97.5])
    return np.mean(a1), np.mean(a2), np.mean(diffs), ci

from sklearn.metrics import roc_auc_score, RocCurveDisplay, roc_curve

def plot_uncertainty_roc(u, correct, name="Model", ax=None, *, set_title=True, title=None):
    """
    Plot an ROC curve treating incorrect predictions as the positive class and
    return both the AUC and the uncertainty threshold that maximizes Youden's J.
    """
    u = np.asarray(u, dtype=float)
    c = np.asarray(correct, dtype=bool)

    # y_true: 1 = incorrect, 0 = correct
    y_true = (~c).astype(int)
    if np.unique(y_true).size < 2:
        raise ValueError("Need both correct and incorrect samples to plot ROC.")

    auc = roc_auc_score(y_true, u)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,2))

    RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_pred=u,
        name=f"{name}",
        ax=ax
    )
    fpr, tpr, thresholds = roc_curve(y_true, u)
    youden = tpr - fpr
    finite_mask = np.isfinite(thresholds)
    if np.all(np.isnan(youden)):
        best_threshold = np.nan
    elif finite_mask.any():
        idx = int(np.nanargmax(youden[finite_mask]))
        best_threshold = thresholds[finite_mask][idx]
    else:
        idx = int(np.nanargmax(youden))
        best_threshold = thresholds[idx]

    ax.set_aspect("auto")        # undo square aspect
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if set_title:
        if title is None:
            title = "ROC — uncertainty detects incorrect predictions"
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return auc, ax, best_threshold

def compare_models_roc(u1, c1, u2, c2, names=("Model 1", "Model 2"), *, ax=None, title=None):
    """
    Plot both models' ROC curves on shared axes, printing AUCs and best thresholds.
    Returns (auc_model1, auc_model2, thr_model1, thr_model2).
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,4), dpi=300)
        created_fig = True

    auc1, _, thr1 = plot_uncertainty_roc(u1, c1, name=names[0], ax=ax, set_title=False)
    auc2, _, thr2 = plot_uncertainty_roc(u2, c2, name=names[1], ax=ax, set_title=False)
    ax.legend(loc="lower right")
    if title is None:
        title = "ROC — uncertainty detects incorrect predictions"
    ax.set_title(title)

    if created_fig:
        plt.tight_layout()
        plt.show()
    print(
        f"{names[0]} AUC: {auc1:.3f} (best τ={thr1})  |  "
        f"{names[1]} AUC: {auc2:.3f} (best τ={thr2})"
    )
    return auc1, auc2, thr1, thr2

def load_uncertainty_samples(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    return np.asarray(payload["all_eu"], dtype=float), np.asarray(payload["all_correct"], dtype=bool)

def plot_task_roc_grid(task_data, model_names=("Model 1", "Model 2"), figure_title=""):
    """
    task_data: list of (task_label, (u_model_A, c_model_A), (u_model_B, c_model_B))
    """
    if not task_data:
        raise ValueError("No task data provided for ROC grid.")

    num_tasks = len(task_data)
    fig, axes = plt.subplots(1, num_tasks, figsize=(6 * num_tasks, 4), dpi=300, sharex=True, sharey=True)
    if num_tasks == 1:
        axes = [axes]

    auc_table = {}
    for ax, (task_label, model_a_data, model_b_data) in zip(axes, task_data):
        auc1, auc2, _, _ = compare_models_roc(
            model_a_data[0],
            model_a_data[1],
            model_b_data[0],
            model_b_data[1],
            names=model_names,
            ax=ax,
            title=task_label,
        )
        auc_table[task_label] = dict(zip(model_names, (auc1, auc2)))

    if figure_title:
        fig.suptitle(figure_title, fontsize=11)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
    else:
        fig.tight_layout()
    plt.show()
    return auc_table

def density_overlap(uncertainty, correct, *, bins=128, grid=1024, seed=0):
    """
    Returns:
      overlap: float in [0,1], area under min(pdf_correct, pdf_incorrect)
      x, y_c, y_i: grid and the two estimated densities on that grid
      crossings: np.ndarray of x where the two densities cross (y_c == y_i approximately)
    """
    u = np.asarray(uncertainty, dtype=float)
    c = np.asarray(correct, dtype=bool)
    mask = np.isfinite(u) & (c == c)
    u, c = u[mask], c[mask]

    u_c = u[c]
    u_i = u[~c]

    if len(u_c) == 0 or len(u_i) == 0:
        return np.nan, None, None, None, np.array([])

    # Common x-range with a little padding
    lo = np.nanpercentile(u, 0.5)
    hi = np.nanpercentile(u, 99.5)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = np.nanmin(u), np.nanmax(u)
        if lo == hi:
            # All values identical → overlap is 1 if both groups non-empty
            return 1.0, np.array([lo, hi]), np.array([1.0, 1.0]), np.array([1.0, 1.0]), np.array([lo])

    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    x = np.linspace(lo, hi, grid)

    # Try KDE first
    y_c = y_i = None
    try:
        from scipy.stats import gaussian_kde
        rng = np.random.default_rng(seed)

        def kde_pdf(vals):
            if len(vals) < 2:
                # Degenerate: represent as near-delta to avoid crashes
                eps = 1e-6 * (hi - lo + 1.0)
                vals = np.array([vals[0] - eps, vals[0] + eps])
            # Tiny jitter helps when many identical values
            jitter = 1e-9 * (hi - lo + 1.0)
            vals = vals + rng.normal(0, jitter, size=len(vals))
            kde = gaussian_kde(vals)  # Scott’s rule
            return kde(x)

        y_c = kde_pdf(u_c)
        y_i = kde_pdf(u_i)

        # Numerical integration of min(pdf1, pdf2)
        overlap = np.trapezoid(np.minimum(y_c, y_i), x)

    except Exception:
        # Fallback: normalized histograms on common bins
        bins = np.linspace(lo, hi, bins + 1)
        h_c, _ = np.histogram(u_c, bins=bins, density=True)
        h_i, _ = np.histogram(u_i, bins=bins, density=True)
        xc = 0.5 * (bins[:-1] + bins[1:])
        dx = np.diff(bins)[0]
        overlap = np.sum(np.minimum(h_c, h_i) * dx)
        x, y_c, y_i = xc, h_c, h_i

    # Crossing points (approximate): where the sign of (y_c - y_i) flips
    diff = y_c - y_i
    s = np.sign(diff)
    flips = np.where(np.diff(s) != 0)[0]
    # Linear interpolation for better crossing estimates
    crossings = []
    for i in flips:
        x0, x1 = x[i], x[i+1]
        y0, y1 = diff[i], diff[i+1]
        if (y1 - y0) != 0:
            t = -y0 / (y1 - y0)
            crossings.append(x0 + t * (x1 - x0))
    crossings = np.array(crossings)

    return overlap, x, y_c, y_i, crossings

import numpy as np

def normalize_uncertainty(u, clip=False):
    """
    Normalize uncertainty scores to the range [-1, +1].

    Args:
        u (array-like): uncertainty scores (any numeric range).
        clip (bool): if True, clamp values strictly within [-1, +1].

    Returns:
        np.ndarray: normalized array in [-1, +1].
    """
    u = np.asarray(u, dtype=float)
    mask = np.isfinite(u)
    if not np.any(mask):
        return np.full_like(u, np.nan)

    u_valid = u[mask]
    u_min, u_max = np.nanmin(u_valid), np.nanmax(u_valid)

    if np.isclose(u_min, u_max):
        # constant values → all map to 0
        norm = np.zeros_like(u, dtype=float)
    else:
        # normalize to [0,1], then rescale to [-1,1]
        norm = (u - u_min) / (u_max - u_min)
        norm = 2 * norm - 1

    if clip:
        norm = np.clip(norm, -1.0, 1.0)

    return norm

def standardize_uncertainty(u, eps=1e-8):
    """
    Standardize uncertainty scores to zero mean and unit standard deviation.

    Args:
        u (array-like): uncertainty scores (any numeric range)
        eps (float): small value to avoid division by zero

    Returns:
        np.ndarray: standardized array with mean≈0 and std≈1
    """
    u = np.asarray(u, dtype=float)
    mask = np.isfinite(u)
    if not np.any(mask):
        return np.full_like(u, np.nan)

    mean = np.nanmean(u[mask])
    std = np.nanstd(u[mask])

    if std < eps:
        # all values identical → return zeros
        return np.zeros_like(u, dtype=float)

    z = (u - mean) / (std + eps)
    return z


pkl_paths = ["eu_comparison/ELC/distrib/task0_uncertainty_distr_2.pkl", "eu_comparison/BLC/distrib/task1_snr_results_distrib.pkl"]
# for pkl_path in pkl_paths:
#     with open(pkl_path, "rb") as f:
#         payload = pickle.load(f)
#     uncertainty_values = payload['all_eu']
#     # uncertainty_values = standardize_uncertainty(uncertainty_values)
#     # uncertainty_values = normalize_uncertainty(uncertainty_values)
#     correctness_bools = payload['all_correct']
#     overlap, x, y_correct, y_incorrect, crossings = density_overlap(uncertainty_values, correctness_bools)
#     print(f"Overlap coefficient: {overlap:.3f}")
#     print("Crossing points (candidate decision thresholds):", crossings)
#     plot_uncertainty_density(uncertainty_values, correctness_bools)

# plt.tight_layout()
# plt.show()
model_names = ("ELC", "BLC")
task_sources = [
    (
        "RadNIST",
        {
            "ELC": "eu_comparison/ELC_kll/task0_uncertainty_distr_calibrated.pkl",
            "BLC": "eu_comparison/BLC/distrib/task0_snr_results_distrib.pkl",
        },
    ),
    (
        "RadChar",
        {
            "ELC": "eu_comparison/ELC_kll/task1_uncertainty_distr_calibrated.pkl",
            "BLC": "eu_comparison/BLC/distrib/task1_snr_results_distrib.pkl",
        },
    ),
]

task_payloads = []
for label, paths in task_sources:
    model_a_data = load_uncertainty_samples(paths[model_names[0]])
    model_b_data = load_uncertainty_samples(paths[model_names[1]])
    task_payloads.append((label, model_a_data, model_b_data))

plot_task_roc_grid(task_payloads, model_names=model_names)
