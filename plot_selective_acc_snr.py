import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Allow loading pickles created with numpy>=2 while running numpy<2."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def load_payload(path):
    payload = _load_pickle(path)
    accs = np.array(payload["results"][0][0], dtype=float)  # from (accs, uncerts)
    aucs = np.array(payload["AUC"], dtype=float)
    return accs, aucs


def load_base_sel_from_single_file(path):
    """
    Return (snr_values, base_acc, sel_acc) extracted from payload['per_snr'] entries.
    """
    payload = _load_pickle(path)
    per_snr = payload.get("per_snr")
    if not per_snr:
        raise KeyError(f"{path} does not contain 'per_snr' entries.")

    snrs, base_acc, sel_acc = [], [], []
    for entry in per_snr:
        snrs.append(entry.get("snr"))
        base_acc.append(entry.get("base_acc", np.nan))
        sel_acc.append(entry.get("sel_acc", np.nan))

    snrs = np.array(snrs, dtype=float)
    base_acc = np.array(base_acc, dtype=float)
    sel_acc = np.array(sel_acc, dtype=float)

    finite_sel = np.isfinite(sel_acc)
    if finite_sel.any() and np.nanmax(sel_acc[finite_sel]) <= 1.5:   # stored as fraction → convert to %
        sel_acc *= 100.0

    return snrs, base_acc, sel_acc

def plot_acc_auc(
    file_map,
    snr_range,
    title="Selective Accuracy vs SNR",
    save_path=None,
    show_auc=False,          # NEW: False -> only accuracy; True -> add AUC on right axis
    acc_ylim=(0.5, 100),
    auc_ylim=(0.5, 1.0)
):
    snr_range = np.array(snr_range, dtype=float)

    fig, ax_acc = plt.subplots(figsize=(12, 10),  dpi=350)
    ax_auc = ax_acc.twinx() if show_auc else None

    unique_models = list({k[0] for k in file_map.keys()})
    unique_tasks  = list({k[1] for k in file_map.keys()})
    model_colors = {m: c for m, c in zip(unique_models, ['#0072B2', '#D55E00', '#009E73', 'maroon'])}
    task_styles = {t: ls for t, ls in zip(sorted(unique_tasks), ["-", "--", "-.", ":"])}
    marker_styles = {t: ls for t, ls in zip(sorted(unique_models), ["o", "s", "o", "s"])}

    lines, labels = [], []
    i = 0
    for (model, task, type), path in file_map.items():
        accs, aucs = load_payload(path)
        accs *= 100 if np.nanmax(accs) <= 1.0 else 1.0
        print(path, "Avg acc", np.mean(accs))
        if len(accs) != len(snr_range):
            if len(accs) < len(snr_range): snr_range = snr_range[:len(accs)]
            # raise ValueError(f"Length mismatch for {(model, task)}: SNRs={len(snr_range)}, accs={len(accs)}")
        if show_auc and len(aucs) != len(snr_range):
            raise ValueError(f"Length mismatch for {(model, task)}: SNRs={len(snr_range)}, aucs={len(aucs)}")
        color = '#0072B2' if "elc" in model.lower() else '#D55E00'
        ls = "-" if "selective" in type.lower() else "--"
        # color = model_colors[model]
        # ls = task_styles[task]; i+=1
        marker = marker_styles[model]
        label = f"{model} · {task} · Sel. Acc" if "selective" in type.lower() else f"{model} · {task} · Base Acc"
        # Accuracy (left y-axis)
        l_acc, = ax_acc.plot(
            snr_range, accs, linestyle=ls, linewidth=2, color=color,
            marker=marker, markersize=4, label=f"{task} · {model} "
        )
        lines.append(l_acc); labels.append(l_acc.get_label())

        # Optional AUC (right y-axis)
        if show_auc:
            l_auc, = ax_auc.plot(
                snr_range, aucs, linestyle=ls, linewidth=2, color=color,
                marker="s", markersize=4, alpha=0.85, label=f"{model} · {task} · AUC"
            )
            lines.append(l_auc); labels.append(l_auc.get_label())

    # Labels, limits, grid
    ax_acc.set_xlabel("SNR (dB)")
    ax_acc.set_ylabel("Recall")
    ax_acc.set_xlim([-20.5,20.5])
    # ax_acc.set_ylim([50,102])
    ax_acc.grid(True, linestyle=":", linewidth=0.8)

    if show_auc:
        ax_auc.set_ylabel("AUC")
        ax_auc.set_ylim([*auc_ylim])
        ax_acc.set_title(title.replace("Accuracy", "Accuracy & AUC"))
    # else:
        # ax_acc.set_title(title)

    ax_acc.legend(lines, labels, loc="lower right", fontsize=9, ncols=1)

    fig.tight_layout()
    # if save_path:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path, dpi=200, bbox_inches="tight")
    fig.subplots_adjust(
        left=0.25,   # space from the left edge
        bottom=0.44, # space from the bottom edge
        right=0.98,  # space from the right edge
        top=0.98,    # space from the top edge
        wspace=0.27,  # width spacing between columns
        hspace=0.474   # height spacing between rows
    )
    plt.show()


def plot_base_sel_from_single_file(
    file_map,
    snr_range=None,
    title="Base & Selective Accuracy vs SNR",
    save_path=None,
    acc_ylim=(50, 102),
    base_only=False
):
    """
    Plot base and selective accuracy curves loaded from a single pickle file.

    file_map: { (model, task): path_to_pickle }
    snr_range: optional iterable to enforce a shared SNR axis/order.
    base_only: when True, only plot base accuracy curves.
    """
    def _align_series(src_snrs, values, target_snrs):
        aligned = np.full(target_snrs.shape, np.nan, dtype=float)
        for i, snr in enumerate(target_snrs):
            matches = np.where(np.isclose(src_snrs, snr, atol=1e-8))[0]
            if matches.size:
                aligned[i] = values[matches[0]]
        return aligned

    snr_override = None
    if snr_range is not None:
        snr_override = np.array(snr_range, dtype=float)

    fig, ax = plt.subplots(figsize=(12, 10), dpi=350)
    unique_models = list({k[0] for k in file_map.keys()})
    unique_tasks = list({k[1] for k in file_map.keys()})
    model_colors = {m: c for m, c in zip(unique_models, ['#0072B2', '#D55E00', '#009E73', 'maroon'])}
    marker_styles = {m: mk for m, mk in zip(sorted(unique_models), ["o", "s", "^", "d"])}
    task_styles = {t: ls for t, ls in zip(sorted(unique_tasks), ["-", "--", "-.", ":"])}

    lines, labels = [], []
    for (model, task), path in file_map.items():
        file_snrs, base_acc, sel_acc = load_base_sel_from_single_file(path)
        if snr_override is not None:
            snrs = snr_override
            base_vals = _align_series(file_snrs, base_acc, snrs)
            sel_vals = _align_series(file_snrs, sel_acc, snrs)
        else:
            snrs, base_vals, sel_vals = file_snrs, base_acc, sel_acc

        base_avg = float(np.nanmean(base_vals)) if base_vals.size else float("nan")
        print(f"[{task} · {model}] Base avg: {base_avg:.4f}")
        if not base_only:
            sel_avg = float(np.nanmean(sel_vals)) if sel_vals.size else float("nan")
            print(f"[{task} · {model}] Selective avg: {sel_avg:.4f}")

        # marker = marker_styles.get(model, "o")
        color = '#0072B2' if "elc" in model.lower() else '#D55E00'

        l_base, = ax.plot(
            snrs, base_vals,
            linestyle="--", linewidth=2, color=color,
            marker='o', markersize=4,
            label=f"{task} · {model} · Base"
        )
        lines.append(l_base); labels.append(l_base.get_label())

        if not base_only:
            l_sel, = ax.plot(
                snrs, sel_vals,
                linestyle="-", linewidth=2, color=color,
                marker="s", markersize=4,
                label=f"{task} · {model} · Selective"
            )
            lines.append(l_sel); labels.append(l_sel.get_label())

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Recall")
    if snr_override is not None:
        ax.set_xlim([snr_override.min() - 0.5, snr_override.max() + 0.5])
    # ax.set_ylim(acc_ylim)
    ax.grid(True, linestyle=":", linewidth=0.8)
    ax.set_title(title)
    ax.legend(lines, labels, loc="lower right", fontsize=9, ncols=1)

    fig.tight_layout()
    fig.subplots_adjust(
        left=0.34,   # space from the left edge
        bottom=0.55, # space from the bottom edge
        right=0.98,  # space from the right edge
        top=0.98,    # space from the top edge
        wspace=0.27,  # width spacing between columns
        hspace=0.474   # height spacing between rows
    )
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ======================
# Example usage
# ======================
# Define your SNRs in the same order used during evaluation
snr_range = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Point these to your actual 4 pickle files (1 per task per model)
file_map = {
    ("ELC (selective)", "RadNIST", "Selective"): "eu_comparison/ELC/task0_snr_results_selective_noisy_test.pkl",
    ("BLC (selective)", "RadNIST", "Selective"): "eu_comparison/BLC/task1_snr_results_selective.pkl",
    ("ELC (base)", "RadNIST", "Base"): "eu_comparison/ELC/task0_snr_results_noisy_test.pkl",
    ("BLC (base)", "RadNIST", "Base"): "eu_comparison/BLC/task1_snr_results.pkl",
    # ("ELC (base)", "RadChar", "Base"): "eu_comparison/ELC/task1_snr_results_noisy_test.pkl",
    # ("BLC (base)", "RadChar", "Base"): "eu_comparison/BLC/task0_snr_results.pkl",
    # ("ELC (selective)", "RadChar", "Selective"): "eu_comparison/ELC/task1_snr_results_selective_noisy_test.pkl",
    # ("BLC (selective)", "RadChar", "Selective"): "eu_comparison/BLC/task0_snr_results_selective.pkl",
}

file_map_single = {
    # ("BLC", "RadNIST"): "eu_comparison/BLC/task0_snr_results_noisy_test.pkl",
    # ("ELC", "RadNIST"): "eu_comparison/ELC_kll/task0_snr_results_calibrated.pkl",
    # ("ELC", "RadNIST"): "eu_comparison/ELC_calibrated/task0_snr_results_calibrated.pkl",
    ("BLC", "RadChar"): "eu_comparison/BLC/task1_snr_results_noisy_test.pkl",
    ("ELC", "RadChar"): "eu_comparison/ELC_kll/task1_snr_results_calibrated.pkl",
    # ("ELC", "RadChar"): "eu_comparison/ELC_calibrated/task1_snr_results_calibrated.pkl",
    # ("BLC", "RadChar"): "eu_comparison/BLC/task1_snr_results_noisy_test.pkl",
    # ("ELC", "RadChar"): "eu_comparison/ELC/task1_snr_results_noisy_test.pkl",
    # ("ELC (base)", "RadNIST", "Base"): "eu_comparison/ELC/task0_snr_results_noisy_test.pkl",
    # ("BLC (base)", "RadNIST", "Base"): "eu_comparison/BLC/task1_snr_results.pkl",
    # ("ELC (base)", "RadChar", "Base"): "eu_comparison/ELC/task1_snr_results_noisy_test.pkl",
    # ("BLC (base)", "RadChar", "Base"): "eu_comparison/BLC/task0_snr_results.pkl",
    # ("ELC (selective)", "RadChar", "Selective"): "eu_comparison/ELC/task1_snr_results_selective_noisy_test.pkl",
    # ("BLC (selective)", "RadChar", "Selective"): "eu_comparison/BLC/task0_snr_results_selective.pkl",
}

# plot_acc_auc(file_map, snr_range,
#              title="Selective Accuracy vs SNR for BLC and ELC",
#              save_path="acc_auc_vs_snr.png")

plot_base_sel_from_single_file(file_map_single, snr_range,
                               title="Base & Selective Accuracy vs SNR for ELC",
                               base_only=False)

