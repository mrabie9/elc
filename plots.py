import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    roc_curve, auc, RocCurveDisplay,
    precision_recall_curve, average_precision_score, PrecisionRecallDisplay,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def _clean_xy(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def plot_hexbin(aleatoric, epistemic, gridsize=40, log_counts=True, mincnt=1):
    """
    2D hexbin density of Aleatoric vs Epistemic entropy.

    Parameters
    ----------
    aleatoric : array-like
    epistemic : array-like
    gridsize : int
        Number of hexes in the x-direction.
    log_counts : bool
        If True, color by log(count).
    mincnt : int
        Minimum count per hex to color.
    """
    x, y = _clean_xy(aleatoric, epistemic)
    plt.figure()
    hb = plt.hexbin(
        x, y,
        gridsize=gridsize,
        bins='log' if log_counts else None,
        mincnt=mincnt
    )
    cbar = plt.colorbar(hb)
    cbar.set_label("log(count)" if log_counts else "count")
    plt.xlabel("Aleatoric entropy")
    plt.ylabel("Epistemic entropy")
    plt.title("Hexbin: Aleatoric vs Epistemic")
    plt.tight_layout()
    plt.show()

def plot_kde2d(aleatoric, epistemic, n=200, levels=(0.5, 0.8, 0.95), shade=True):
    """
    2D KDE (gaussian_kde) of Aleatoric vs Epistemic entropy with filled contours.

    Parameters
    ----------
    aleatoric : array-like
    epistemic : array-like
    n : int
        Grid resolution per axis.
    levels : tuple of floats in (0,1]
        Credible mass levels to show as contours (approximate).
    shade : bool
        If True, fill the highest-density region; always draw contour lines.
    """
    x, y = _clean_xy(aleatoric, epistemic)
    if x.size < 5:
        raise ValueError("Not enough points for KDE.")

    # Fit KDE
    kde = gaussian_kde(np.vstack([x, y]))

    # Grid bounds with small padding
    xpad = 0.05 * (np.nanmax(x) - np.nanmin(x) + 1e-12)
    ypad = 0.05 * (np.nanmax(y) - np.nanmin(y) + 1e-12)
    xmin, xmax = np.nanmin(x) - xpad, np.nanmax(x) + xpad
    ymin, ymax = np.nanmin(y) - ypad, np.nanmax(y) + ypad

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    grid = np.vstack([xx.ravel(), yy.ravel()])
    zz = kde(grid).reshape(xx.shape)

    # Convert density to cumulative mass thresholds for contour levels
    # (Sort densities high->low and map to cumulative probability)
    z_flat = zz.ravel()
    order = np.argsort(z_flat)[::-1]
    z_sorted = z_flat[order]
    cdf = np.cumsum(z_sorted)
    cdf /= cdf[-1]

    # For a target mass p, find the density threshold t where mass inside {z >= t} ≈ p
    def mass_to_threshold(p):
        idx = np.searchsorted(cdf, p)
        return z_sorted[idx if idx < z_sorted.size else -1]

    thresholds = [mass_to_threshold(p) for p in levels]
    thresholds = sorted(thresholds)  # increasing for contour plotting

    plt.figure()
    if shade:
        plt.contourf(xx, yy, zz, levels=[thresholds[0], zz.max()], alpha=0.4)
    cs = plt.contour(xx, yy, zz, levels=thresholds)
    plt.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{levels[list(thresholds).index(v)]*100:.0f}%")
    plt.scatter(x, y, s=4, alpha=0.5)
    plt.xlabel("Aleatoric entropy")
    plt.ylabel("Epistemic entropy")
    plt.title("2D KDE: Aleatoric vs Epistemic")
    plt.tight_layout()
    plt.show()

def plot_multiclass_roc_auc(y_true, y_proba, class_names=None, title="ROC (multiclass OvR)"):
    """
    y_true: shape (n_samples,), integer class labels 0..C-1
    y_proba: shape (n_samples, C), per-class probabilities/scores
    class_names: list of length C (optional)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    Y = label_binarize(y_true, classes=np.arange(n_classes))

    # Per-class ROC/AUC
    fprs, tprs, aucs = {}, {}, {}
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
        fprs[i], tprs[i] = fpr, tpr
        aucs[i] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={aucs[i]:.3f})")
    # Micro-average (aggregate decisions)
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average (AUC={auc_micro:.3f})")

    # Macro-average (mean of per-class AUCs)
    auc_macro = np.mean(list(aucs.values()))
    print(auc_macro)

    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} | macro={auc_macro:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {"per_class_auc": aucs, "micro_auc": auc_micro, "macro_auc": auc_macro}

def get_auc(y_true, y_proba, class_names=None, plot=False):
    """
    y_true: shape (n_samples,), integer class labels 0..C-1
    y_proba: shape (n_samples, C), per-class probabilities/scores
    class_names: list of length C (optional)
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    Y = label_binarize(y_true, classes=np.arange(n_classes))

    # Per-class ROC/AUC
    fprs, tprs, aucs = {}, {}, {}
    # plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(Y[:, i], y_proba[:, i])
        # print(fpr)
        # print(tpr)
        fprs[i], tprs[i] = fpr, tpr
        aucs[i] = auc(fpr, tpr)
        # plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={aucs[i]:.3f})")
        # print("positives:", (Y[:, i] == 1).sum(), "negatives:", (y_proba[:, i] == 0).sum())
        # print("nan in y_true:", np.isnan(Y[:, i]).any(), "nan in y_score:", np.isnan(y_proba[:, i]).any())

    # Micro-average (aggregate decisions)
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), y_proba.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    # Macro-average (mean of per-class AUCs)
    auc_macro = np.mean(list(aucs.values()))

    if plot:
        return auc_macro, fpr_micro, tpr_micro, auc_micro, aucs
    else:
        return auc_macro
   
def _finite_mask(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def aurc(coverage, risk):
    """
    Compute AURC = ∫ risk d(coverage) over coverage∈[0,1].
    Assumes 'coverage' is increasing; sorts if needed.
    """
    cov, r = _finite_mask(np.asarray(coverage), np.asarray(risk))
    order = np.argsort(cov)
    cov, r = cov[order], r[order]
    return np.trapz(r, cov)

def plot_acc_cov(results_list, labels=None, mark_idx=None, title="Accuracy–Coverage"):
    """
    Plot selective Accuracy–Coverage curves.

    results_list: list of dicts from sweep_thresholds (each must have 'coverage' and 'sel_acc')
    labels: list of legend labels (same length as results_list)
    mark_idx: optional list of indices to mark (e.g., chosen τ per result), or single int for all
    """
    if labels is None: labels = [f"Model {i+1}" for i in range(len(results_list))]
    plt.figure()
    for i, res in enumerate(results_list):
        cov = np.asarray(res["coverage"])
        acc = np.asarray(res["sel_acc"])
        cov, acc = _finite_mask(cov, acc)
        plt.plot(cov, acc, label=labels[i])
        # Optional marker at chosen τ index
        if mark_idx is not None:
            idx = mark_idx[i] if isinstance(mark_idx, (list, tuple)) else mark_idx
            if idx is not None and 0 <= idx < len(res["coverage"]):
                c0, a0 = res["coverage"][idx], res["sel_acc"][idx]
                if np.isfinite(c0) and np.isfinite(a0):
                    plt.scatter([c0], [a0], s=30)

    plt.xlabel("Coverage")
    plt.ylabel("Selective Accuracy")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_risk_cov(results_list, labels=None, title="Risk–Coverage", show_aurc=True):
    """
    Plot Risk–Coverage; optionally prints AURC per curve.
    """
    if labels is None: labels = [f"Model {i+1}" for i in range(len(results_list))]
    plt.figure()
    for i, res in enumerate(results_list):
        cov = np.asarray(res["coverage"])
        risk = np.asarray(res["sel_risk"])
        cov, risk = _finite_mask(cov, risk)
        plt.plot(cov, risk, label=labels[i])
        if show_aurc:
            val = aurc(cov, risk)
            print(f"AURC [{labels[i]}]: {val:.4f}")

    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk (1 - accuracy)")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()