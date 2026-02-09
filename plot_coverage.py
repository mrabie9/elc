import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scripts.mod_conf_matrix import _average_sweep_results

def _finite_mask(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def load_results_from_pickle(pkl_path):
    """Load pickle with structure {'results': [results_epi, results_ale, results_tot]}."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if "results" in payload:
        results_list = payload["results"]
    elif "avg_curve" in payload:
        avg_curve = _average_sweep_results(payload['per_snr'], snr_bounds=(-20, 0)) #payload["avg_curve"]
        if isinstance(avg_curve, dict):
            ordered = []
            for key in ("eu", "ale", "tot"):
                if key in avg_curve:
                    ordered.append(avg_curve[key])
            if not ordered:
                ordered = list(avg_curve.values())
            results_list = ordered
        else:
            results_list = avg_curve
    else:
        raise KeyError(f"{pkl_path} does not contain a 'results' or 'avg_curve key.")
    
    
    
    if not isinstance(results_list, (list, tuple)) or len(results_list) < 3:
        raise ValueError(f"'results' must be a list like [epi, ale, tot].")
    return results_list  # [results_epi, results_ale, results_tot]

def plot_acc_cov_two_tasks_from_payloads(
    task_pickles,                       # [path_task0.pkl, path_task1.pkl]
    task_labels=("RadChar", "RadNIST"),
    type_labels=("Epistemic", "Aleatoric", "Total"),
    res0 = None,
    res1 = None,
    save_path=None
):
    """
    Load selective prediction results for two tasks and plot Accuracy–Coverage.

    Each pickle file must contain:
        {'results': [results_epi, results_ale, results_tot]}
    where each results_* is a dict from sweep_thresholds().
    """
    if len(task_pickles) != 2:
        raise ValueError("Provide exactly two pickle paths (one per task).")

    # Load lists of results for both tasks
    res_task0 = res0 or load_results_from_pickle(task_pickles[0])
    res_task1 = res1 or load_results_from_pickle(task_pickles[1])

    colors = ["tab:blue", "tab:orange", "tab:green"]  # uncertainty types
    linestyles = [":", "--"]                          # per task

    plt.figure(figsize=(7, 5), dpi=350)
    for t, res_list in enumerate([res_task0, res_task1]):
        for k, res in enumerate(res_list[:3]):  # epi, ale, tot
            cov = np.asarray(res["coverage"])
            acc = np.asarray(res["sel_acc"])
            cov, acc = _finite_mask(cov, acc)
            plt.plot(
                cov, acc,
                linestyle=linestyles[t],
                color=colors[k],
                label=f"{task_labels[t]} - {type_labels[k]}"
            )

    plt.xlabel("Coverage")
    plt.ylabel("Selective Recall")
    # plt.title("Accuracy–Coverage Tradeoff Across Tasks")
    plt.xlim(0, 1.01)
    plt.ylim(0.2, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.34,   # space from the left edge
        bottom=0.55, # space from the bottom edge
        right=0.98,  # space from the right edge
        top=0.98,    # space from the top edge
        wspace=0.27,  # width spacing between columns
        hspace=0.474   # height spacing between rows
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def _finite_mask(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def load_epistemic_pickle(pkl_path):
    """
    Load a pickle file containing epistemic results in any of these formats:
      {'results': [res_epi]}        # common
      {'coverage': ..., 'sel_acc': ...}  # single dict
    Returns: a single result dict with keys ['coverage', 'sel_acc']
    """
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    # Case 1: dict with 'results' list
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
        return results  # first entry = epistemic
    
    if isinstance(payload, dict) and "avg_curve" in payload:
        results = payload["avg_curve"]
        return results  # first entry = epistemic

    # Case 2: dict with direct keys
    elif isinstance(payload, dict) and {"coverage", "sel_acc"}.issubset(payload.keys()):
        return payload

    raise ValueError(f"{pkl_path}: unsupported file structure.")

def plot_epistemic_acc_cov(pkl_paths, labels=None, save_path=None, title="Epistemic Accuracy–Coverage"):
    """
    Plot selective accuracy vs. coverage for epistemic uncertainty results.

    Parameters
    ----------
    pkl_paths : list[str]
        List of pickle file paths (each containing epistemic results).
    labels : list[str] or None
        Legend labels corresponding to each file.
    save_path : str or None
        Optional path to save the figure.
    title : str
        Plot title.
    """
    if labels is None:
        labels = [os.path.basename(p).replace(".pkl", "") for p in pkl_paths]

    plt.figure(figsize=(7, 5), dpi=350)
    for path, label in zip(pkl_paths, labels):
        res = load_epistemic_pickle(path)
        cov = np.asarray(res["coverage"])
        acc = np.asarray(res["sel_acc"])
        cov, acc = _finite_mask(cov, acc)
        plt.plot(cov, acc, label=label, linewidth=2)

    plt.xlabel("Coverage")
    plt.ylabel("Selective Recall")
    # plt.title(title)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.subplots_adjust(
        left=0.34,   # space from the left edge
        bottom=0.55, # space from the bottom edge
        right=0.98,  # space from the right edge
        top=0.98,    # space from the top edge
        wspace=0.27,  # width spacing between columns
        hspace=0.474   # height spacing between rows
    )
    plt.show()


def test(pkl_path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    
    payload['avg_curve'] = _average_sweep_results(payload['per_snr'])

    return payload
    
    # snr_n18 = payload['per_snr'][0]
    # snr_n18_eu = snr_n18[0]
    # snr_n18_ale = snr_n18[1]
    # snr_n18_tot = snr_n18[2]

    # avg_curve = [np.zeros(10001), [np.zeros(10001)], [np.zeros(10001)]]
    # for u in range(0,3):
    #     avg_rec = []
    #     for i in range(10001):
    #         avg_snr = 0
    #         for snr in payload['per_snr']:
    #             # print(snr[u]['coverage'].shape, snr[u]['coverage'][0], snr[u]['sel_acc'][0])
    #             avg_snr += snr[u]['sel_acc'][i]/len(payload['per_snr'])
    #         avg_rec.append(avg_snr)
    #     avg_curve[u] = np.array(avg_rec)
        
    # # print(payload['avg_curve']['eu']['coverage'][0], payload['avg_curve']['eu']['sel_acc'][0])
    # for u in avg_curve:
    #     x = [i/10000 for i in range(10001)]
    #     y = u
    #     plt.plot(x, y)
    # plt.xlabel("Coverage")
    # plt.ylabel("Selective Recall")
    # plt.title("Epistemic Accuracy–Coverage Tradeoff Across SNRs")
    # plt.xlim(0, 1.01)
    # plt.ylim(0, 1)
    # plt.grid(True, linestyle="--", alpha=0.4)
    # plt.legend(["Avg Epistemic", "Avg Aleatoric", "Avg Total"])
    # plt.tight_layout()
    # plt.show()

    
# test("eu_comparison/BLC/snr_coverage/task1_snr_coverage.pkl")
plot_acc_cov_two_tasks_from_payloads(
    # ["eu_comparison/BLC/snr_coverage/task0_snr_coverage.pkl", "eu_comparison/BLC/snr_coverage/task1_snr_coverage.pkl"],
    ["eu_comparison/BLC/coverage/task0_coverage_data_noisy_dcheck.pkl", "eu_comparison/BLC/coverage/task1_coverage_data_noisy_dcheck.pkl"],
    task_labels=("RadNIST", "RadChar"),
    type_labels=("Epistemic", "Aleatoric", "Total"),
    save_path="figs/acc_cov_two_tasks.png"
)
# # --- Example usage ---
plot_epistemic_acc_cov(
    # ["eu_comparison/ELC/coverage/task0_coverage_data_noisy_dcheck.pkl", "eu_comparison/ELC/coverage/task1_coverage_data_noisy_dcheck.pkl"],
    # ["eu_comparison/ELC/snr_coverage/task0_snr_coverage.pkl", "eu_comparison/ELC/snr_coverage/task1_snr_coverage.pkl"],
    # ["eu_comparison/ELC_calibrated/coverage/task0_coverage_data_calibrated.pkl", "eu_comparison/ELC_calibrated/coverage/task1_coverage_data_calibrated.pkl"],
    ["eu_comparison/ELC_kll/coverage/task0_coverage_data_calibrated.pkl", "eu_comparison/ELC_kll/coverage/task1_coverage_data_calibrated.pkl"],
    labels=["RadNIST", "RadChar"],
    save_path="figs/epistemic_acc_cov.png"
)

# "eu_comparison/BLC/coverage/task0_coverage_data.pkl", "eu_comparison/BLC/coverage/task1_coverage_data.pkl"]
# "eu_comparison/ELC/coverage/task0_coverage_data.pkl", "eu_comparison/ELC/coverage/task1_coverage_data.pkl"],
