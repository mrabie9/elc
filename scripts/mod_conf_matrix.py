import os
import pickle
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Allow running the script directly
sys.path.append("/home/lunet/wsmr11/repos/LPSforECNN/")

from scripts.helper import (  # noqa: E402
    args,
    dataset,
    dataset_paths,
    datasets,
    model_path,
    snr_range,
    task_configs,
    to_serializable,
)
from models.bayes_resnet import Net
from TrainValTest import CVTrainValTest, accuracy  # noqa: E402
from sweep_eu import choose_tau, sweep_thresholds  # noqa: E402

def _filter_noise_only_test_split(pipeline: CVTrainValTest) -> None:
    """Filter class-0 samples from pipeline test data and rebuild test loader."""
    y_test_np = np.asarray(pipeline.y_test)
    x_test_np = np.asarray(pipeline.x_test)
    keep_mask = y_test_np != 0
    removed = int((~keep_mask).sum())
    if removed <= 0:
        return

    pipeline.y_test = y_test_np[keep_mask]
    pipeline.x_test = x_test_np[keep_mask]
    kept_classes = np.unique(pipeline.y_test).tolist()

    pipeline.test_set = type(pipeline.test_set)(pipeline.x_test, pipeline.y_test)
    batch_size = getattr(pipeline.test_generator, "batch_size", 256) or 256
    pipeline.test_generator = DataLoader(
        pipeline.test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(
        f"Removed {removed} noise-only test samples (class 0). "
        f"Kept classes: {kept_classes}"
    )


def _filter_noise_only_train_split(pipeline: CVTrainValTest) -> None:
    """Filter class-0 samples from pipeline train data and rebuild train loader."""
    y_train_np = np.asarray(pipeline.y_train)
    x_train_np = np.asarray(pipeline.x_train)
    keep_mask = y_train_np != 0
    removed = int((~keep_mask).sum())
    if removed <= 0:
        return

    pipeline.y_train = y_train_np[keep_mask]
    pipeline.x_train = x_train_np[keep_mask]
    kept_classes = np.unique(pipeline.y_train).tolist()

    pipeline.training_set = type(pipeline.training_set)(pipeline.x_train, pipeline.y_train)
    batch_size = getattr(pipeline.train_generator, "batch_size", 256) or 256
    pipeline.train_generator = DataLoader(
        pipeline.training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(
        f"Removed {removed} noise-only train samples (class 0). "
        f"Kept classes: {kept_classes}"
    )


def load_model_and_mask(task: str, model_file: str):
    """Load a model checkpoint and its optional cumulative mask."""
    model = Net(input_size=1024, num_classes=datasets[dataset]["num_classes"], args=args)
    model.load_state_dict(torch.load(os.path.join(model_path, model_file)))

    mask_path = os.path.join(datasets[dataset]["models"], f"{task}/cumu_mask.pkl")
    trained_mask = pickle.load(open(mask_path, "rb")) if os.path.exists(mask_path) else None

    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, trained_mask


def prepare_pipeline(task: str) -> CVTrainValTest:
    base_path = datasets[dataset]["tasks"] + task
    return CVTrainValTest(base_path=base_path, save_path=base_path)


def _resolve_target_coverage(target_coverage, task_name: str, task_index: int) -> float:
    """
    Accepts a scalar, list/tuple, or dict keyed by task name/index and returns the
    coverage value for the requested task.
    """
    if isinstance(target_coverage, dict):
        if task_name in target_coverage:
            return target_coverage[task_name]
        if task_index in target_coverage:
            return target_coverage[task_index]
        str_index = str(task_index)
        if str_index in target_coverage:
            return target_coverage[str_index]
        if target_coverage:
            return float(next(iter(target_coverage.values())))
    if isinstance(target_coverage, (list, tuple)):
        if 0 <= task_index < len(target_coverage):
            return target_coverage[task_index]
    return float(target_coverage)


def init_task_context(task_index: int, config: Dict) -> Dict:
    """Initialize model, mask, and pipeline for a task index."""
    args.current_task = task_index
    task = config["task"]
    offset = datasets[dataset]["offset"][task_index]
    model, trained_mask = load_model_and_mask(task, config["file"])
    pipeline = prepare_pipeline(task)
    return {
        "task": task,
        "offset": offset,
        "model": model,
        "trained_mask": trained_mask,
        "pipeline": pipeline,
        "index": task_index,
    }


def evaluate_split(
    pipeline: CVTrainValTest,
    model: torch.nn.Module,
    trained_mask,
    offset: int,
    data_path: str = None,
    mixed_snrs: bool = False,
    remove_noise_only: bool = False,
    split: str = "test",
):
    """Run inference for a single split and return labels, predictions, and uncertainties."""
    pipeline.load_data_dronerc(256, offset=offset, data=data_path, mixed_snrs=mixed_snrs, args=args)
    if remove_noise_only:
        if split == "train":
            _filter_noise_only_train_split(pipeline)
        else:
            _filter_noise_only_test_split(pipeline)

    orig_test_generator = None
    test_flag = True
    if split == "train":
        if not hasattr(pipeline, "train_generator"):
            raise RuntimeError("Training generator not initialized; cannot calibrate on train split.")
        orig_test_generator = pipeline.test_generator
        pipeline.test_generator = pipeline.train_generator
        test_flag = True
    elif split == "val":
        test_flag = False

    try:
        labels, preds, unique_labels, eu, h_pred, f_eu, y_proba = pipeline.test_model(
                    args, model, trained_mask, eval_entropy=True,cm=True, enable_diagnostics=False, return_logits=True, test=test_flag)
    finally:
        if orig_test_generator is not None:
            pipeline.test_generator = orig_test_generator

    labels = np.asarray(labels) - offset
    preds = np.asarray(preds) - offset
    unique_labels = np.asarray(unique_labels) - offset
    eu = np.asarray(eu)
    h_pred = np.asarray(h_pred)

    if remove_noise_only:
        keep_mask = labels != 0
        labels = labels[keep_mask]
        preds = preds[keep_mask]
        eu = eu[keep_mask]
        h_pred = h_pred[keep_mask]
        unique_labels = unique_labels[unique_labels != 0]

    return labels, preds, unique_labels, eu, h_pred


def pick_threshold(all_labels, all_preds, eu, target_coverage: float):
    """Select tau using sweep results for a desired coverage."""
    sweep_results = sweep_thresholds(all_labels, all_preds, u=list(eu))
    return choose_tau(sweep_results, target_coverage=target_coverage), sweep_results


def evaluate_snr_range_for_task(
    task_index: int,
    config: Dict,
    target_coverage: float = 0.90,
    context: Dict = None,
    remove_noise_only: bool = False,
    global_coverage: bool = False,
    calibrate_on_train: bool = False,
    uncertainty_key: str = "eu",
) -> Tuple[Dict, Dict]:
    """Evaluate selective metrics across the configured SNR range for one task."""
    ctx = context or init_task_context(task_index, config)

    if task_index >= len(dataset_paths):
        raise IndexError(f"No dataset path configured for task index {task_index}")
    data_root = dataset_paths[task_index]

    sel_accs: List[float] = []
    base_accs: List[float] = []
    covs: List[float] = []
    accepted_uncerts: List[np.ndarray] = []
    per_snr: List[Dict] = []
    all_u: List[float] = []
    all_labels: List[int] = []
    all_preds: List[int] = []
    per_snr_data: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    calibrated_tau: float = None

    if calibrate_on_train:
        cal_labels, cal_preds, _, cal_eu, cal_hpred = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=None,
            mixed_snrs=False,
            remove_noise_only=remove_noise_only,
            split="train",
        )
        if uncertainty_key == "h_pred":
            cal_u = np.asarray(cal_hpred)
        else:
            cal_u = np.asarray(cal_eu)
        cal_best, _ = pick_threshold(cal_labels, cal_preds, cal_u, target_coverage=target_coverage)
        calibrated_tau = float(cal_best["tau"])
        print(
            f"[{ctx['task']}] calibrated tau on training data: {calibrated_tau:.6f} "
            f"(target coverage {target_coverage:.2f})"
        )

    for snr in snr_range:
        # if snr > -8: break
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        labels, preds, unique_labels, eu, h_pred = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=snr_path,
            mixed_snrs=True,
            remove_noise_only=remove_noise_only,
        )
        base_acc = accuracy(torch.tensor(labels), torch.tensor(preds)) * 100
        print(f"[{ctx['task']}] SNR {snr} dB - Base accuracy: {base_acc:.2f}%")

        eu = np.asarray(eu)
        h_pred = np.asarray(h_pred)
        u = h_pred if uncertainty_key == "h_pred" else eu
        if global_coverage or calibrate_on_train:
            per_snr_data.append((snr, np.asarray(labels), np.asarray(preds), np.asarray(unique_labels), u, base_acc))
            all_u.extend(u.tolist())
            all_labels.extend(np.asarray(labels).tolist())
            all_preds.extend(np.asarray(preds).tolist())
            base_accs.append(base_acc)
        else:
            best, _ = pick_threshold(labels, preds, u, target_coverage=target_coverage)
            accept_mask = u <= best["tau"]
            accepted = int(accept_mask.sum())
            total = int(u.size)
            print(
                f"[{ctx['task']}] SNR {snr} dB - accepted {accepted}/{total} "
                f"({best['coverage']:.2%})"
            )

            base_accs.append(base_acc)
            sel_accs.append(best["sel_acc"])
            covs.append(best["coverage"])
            accepted_uncerts.append(u[accept_mask])
            per_snr.append(
                {
                    "snr": snr,
                    "base_acc": base_acc,
                    "tau": best["tau"],
                    "coverage": best["coverage"],
                    "sel_acc": best["sel_acc"],
                    "sel_risk": best["sel_risk"],
                    "unique_labels": unique_labels,
                }
            )

    global_tau = None
    if global_coverage or calibrate_on_train:
        if calibrate_on_train:
            global_tau = float(calibrated_tau)
        else:
            best, _ = pick_threshold(all_labels, all_preds, all_u, target_coverage=target_coverage)
            global_tau = float(best["tau"])
        for snr, labels, preds, unique_labels, u, base_acc in per_snr_data:
            accept_mask = u <= global_tau
            coverage = float(accept_mask.mean()) if u.size else 0.0
            accepted = int(accept_mask.sum())
            total = int(u.size)
            print(
                f"[{ctx['task']}] SNR {snr} dB - accepted {accepted}/{total} "
                f"({coverage:.2%})"
            )
            if accept_mask.any():
                sel_acc = accuracy(torch.tensor(labels[accept_mask]), torch.tensor(preds[accept_mask]))
                sel_risk = 1.0 - sel_acc
            else:
                sel_acc = float("nan")
                sel_risk = float("nan")

            sel_accs.append(float(sel_acc))
            covs.append(coverage)
            accepted_uncerts.append(u[accept_mask])
            per_snr.append(
                {
                    "snr": snr,
                    "base_acc": base_acc,
                    "tau": global_tau,
                    "coverage": coverage,
                    "sel_acc": float(sel_acc),
                    "sel_risk": float(sel_risk),
                    "unique_labels": unique_labels,
                }
            )

    payload = {
        "results": to_serializable([(sel_accs, accepted_uncerts)]),
        "base_accs": to_serializable(base_accs),
        "AUC": [],
        "per_snr": to_serializable(per_snr),
    }
    if global_coverage:
        payload["global_tau"] = global_tau
    stats = {
        "mean_sel_acc": float(np.mean(sel_accs)) if sel_accs else float("nan"),
        "mean_base_acc": float(np.mean(base_accs)) if base_accs else float("nan"),
        "mean_cov": float(np.mean(covs)) if covs else float("nan"),
    }
    return payload, stats


def export_uncertainty_distribution_for_task(
    task_index: int,
    config: Dict = None,
    snr_upper_limit: float = -8,
    context: Dict = None,
    output_root: str = "eu_comparison/BLC/distrib",
    remove_noise_only: bool = False,
) -> str:
    """
    Aggregate correctness indicators and epistemic uncertainty values across SNR
    splits up to ``snr_upper_limit`` and persist them for downstream analysis.

    The resulting pickle contains ``all_correct`` (bool list) and ``all_eu`` as
    expected by ``plot_uncertainty_distr.py`` along with simple metadata.
    """
    config = config or task_configs[task_index]
    ctx = context or init_task_context(task_index, config)

    if task_index >= len(dataset_paths):
        raise IndexError(f"No dataset path configured for task index {task_index}")
    data_root = dataset_paths[task_index]

    all_correct: List[bool] = []
    all_eu: List[float] = []
    snrs: List[int] = []

    for snr in snr_range:
        if snr_upper_limit is not None and snr > snr_upper_limit:
            break
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        labels, preds, _, eu, _ = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=snr_path,
            mixed_snrs=True,
            remove_noise_only=remove_noise_only,
        )
        correct = np.asarray(labels) == np.asarray(preds)
        all_correct.extend(correct.astype(bool).tolist())
        all_eu.extend(np.asarray(eu, dtype=float).tolist())
        snrs.extend([snr] * len(eu))

    payload = {
        "task": ctx["task"],
        "dataset": dataset,
        "snr_upper_limit": snr_upper_limit,
        "snrs": snrs,
        "all_correct": all_correct,
        "all_eu": all_eu,
    }
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{ctx['task']}_snr_results_distrib.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    print(
        f"[{ctx['task']}] exported {len(all_eu)} samples "
        f"(SNR ≤ {snr_upper_limit} dB) to {output_path}"
    )
    return output_path


def _average_sweep_results(
    sweep_results: List[Dict],
    snr_bounds: Tuple[float, float] = None,
    snr_values: Sequence[float] = None,
) -> Dict:
    """
    Average coverage curves across multiple sweep outputs (e.g., different SNRs).

    Parameters
    ----------
    sweep_results : list
        A list whose entries correspond to individual SNR sweeps. Each entry is
        either ``[epi, ale, tot]`` or a dict keyed by uncertainty type.
    snr_bounds : tuple(float, float), optional
        Inclusive (min, max) bounds specifying which SNRs to include. When
        provided, ``snr_values`` (or the module-level ``snr_range``) is used to
        align bounds with ``sweep_results`` order.
    snr_values : sequence, optional
        Explicit SNR values aligned with ``sweep_results``. If omitted, the
        global ``snr_range`` is used.
    """
    keys = ["tau", "coverage", "sel_acc", "sel_risk"]
    uncertainties = ["eu", "ale", "tot"]

    if not sweep_results:
        return {u: {k: np.array([]) for k in keys} for u in uncertainties}

    indices = list(range(len(sweep_results)))
    if snr_bounds is not None:
        snr_vals = list(snr_values) if snr_values is not None else list(snr_range)
        if not snr_vals:
            raise ValueError("snr_bounds specified but no SNR values were provided.")
        low, high = snr_bounds
        indices = [
            idx
            for idx, snr in enumerate(snr_vals[: len(sweep_results)])
            if low <= snr <= high
        ]
    if not indices:
        return {u: {k: np.array([]) for k in keys} for u in uncertainties}

    averaged = {u: {} for u in uncertainties}
    for u_idx, u in enumerate(uncertainties):
        per_uncert_results = []
        for idx in indices:
            snr_results = sweep_results[idx]
            if isinstance(snr_results, dict):
                # Handle dict structure from serialized payloads.
                snr_results = [snr_results.get(name) for name in uncertainties]
            if snr_results and u_idx < len(snr_results):
                per_uncert_results.append(snr_results[u_idx])
        if not per_uncert_results:
            averaged[u] = {k: np.array([]) for k in keys}
            continue

        min_len = min(len(np.asarray(res["coverage"])) for res in per_uncert_results)
        for key in keys:
            stacked = np.stack([np.asarray(res[key])[:min_len] for res in per_uncert_results], axis=0)
            averaged[u][key] = np.nanmean(stacked, axis=0)
    return averaged


def evaluate_snr_coverage_for_task(
    task_index: int,
    config: Dict,
    target_coverage: float = 0.95,
    context: Dict = None,
    output_root: str = "eu_comparison/BLC/snr_coverage",
    remove_noise_only: bool = False,
    calibrate_on_train: bool = False,
):
    """
    Sweep uncertainty thresholds for each SNR split, then average the selective
    accuracy/coverage curves across SNR values for plotting.
    """
    ctx = context or init_task_context(task_index, config)

    if task_index >= len(dataset_paths):
        raise IndexError(f"No dataset path configured for task index {task_index}")
    data_root = dataset_paths[task_index]

    sweep_by_snr: List[Dict] = []
    best_by_snr: List[Dict] = []
    calibrated_tau = None

    if calibrate_on_train:
        cal_labels, cal_preds, _, cal_eu, cal_hpred = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=None,
            mixed_snrs=False,
            remove_noise_only=remove_noise_only,
            split="train",
        )
        cal_alea = [total - eu for total, eu in zip(cal_hpred, cal_eu)]
        cal_tot = cal_hpred
        cal_results_epi = sweep_thresholds(cal_labels, cal_preds, u=list(cal_eu))
        cal_results_ale = sweep_thresholds(cal_labels, cal_preds, u=list(cal_alea))
        cal_results_tot = sweep_thresholds(cal_labels, cal_preds, u=list(cal_tot))
        cal_best_epi = choose_tau(cal_results_epi, target_coverage=target_coverage)
        cal_best_ale = choose_tau(cal_results_ale, target_coverage=target_coverage)
        cal_best_tot = choose_tau(cal_results_tot, target_coverage=target_coverage)
        calibrated_tau = {
            "epi": float(cal_best_epi["tau"]),
            "ale": float(cal_best_ale["tau"]),
            "tot": float(cal_best_tot["tau"]),
        }
        print(
            f"[{ctx['task']}] calibrated taus on training data (target {target_coverage:.2f}): "
            f"epi={calibrated_tau['epi']:.6f}, ale={calibrated_tau['ale']:.6f}, "
            f"tot={calibrated_tau['tot']:.6f}"
        )

    for snr in snr_range:
        # if snr > -18: break
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        all_labels, all_preds, unique_labels, eu, h_pred = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=snr_path,
            mixed_snrs=True,
            remove_noise_only=remove_noise_only,
        )
        alea = [total-eu for total, eu in zip(h_pred, eu)]
        results_epi = sweep_thresholds(all_labels, all_preds, u=list(eu))
        results_ale = sweep_thresholds(all_labels, all_preds, u=list(alea))
        results_tot = sweep_thresholds(all_labels, all_preds, u=list(h_pred))
        if calibrate_on_train:
            best_epi = {"tau": calibrated_tau["epi"]}
            best_ale = {"tau": calibrated_tau["ale"]}
            best_tot = {"tau": calibrated_tau["tot"]}
        else:
            best_epi = choose_tau(results_epi, target_coverage=target_coverage)
            best_ale = choose_tau(results_ale, target_coverage=target_coverage)
            best_tot = choose_tau(results_tot, target_coverage=target_coverage)

        sweep_by_snr.append([results_epi, results_ale, results_tot])
        if calibrate_on_train:
            epi_accept = eu <= best_epi["tau"]
            ale_accept = np.asarray(alea) <= best_ale["tau"]
            tot_accept = np.asarray(h_pred) <= best_tot["tau"]
            best_epi = {
                "snr": snr,
                "tau": best_epi["tau"],
                "coverage": float(epi_accept.mean()) if eu.size else 0.0,
                "sel_acc": float(accuracy(torch.tensor(all_labels[epi_accept]), torch.tensor(all_preds[epi_accept])))
                if epi_accept.any()
                else float("nan"),
                "sel_risk": float("nan"),
                "unique_labels": unique_labels,
            }
            if epi_accept.any():
                best_epi["sel_risk"] = 1.0 - best_epi["sel_acc"]

            best_ale = {
                "snr": snr,
                "tau": best_ale["tau"],
                "coverage": float(ale_accept.mean()) if np.asarray(alea).size else 0.0,
                "sel_acc": float(accuracy(torch.tensor(all_labels[ale_accept]), torch.tensor(all_preds[ale_accept])))
                if ale_accept.any()
                else float("nan"),
                "sel_risk": float("nan"),
                "unique_labels": unique_labels,
            }
            if ale_accept.any():
                best_ale["sel_risk"] = 1.0 - best_ale["sel_acc"]

            best_tot = {
                "snr": snr,
                "tau": best_tot["tau"],
                "coverage": float(tot_accept.mean()) if np.asarray(h_pred).size else 0.0,
                "sel_acc": float(accuracy(torch.tensor(all_labels[tot_accept]), torch.tensor(all_preds[tot_accept])))
                if tot_accept.any()
                else float("nan"),
                "sel_risk": float("nan"),
                "unique_labels": unique_labels,
            }
            if tot_accept.any():
                best_tot["sel_risk"] = 1.0 - best_tot["sel_acc"]

        best_by_snr.append(
            {'epi': {
                "snr": snr,
                "tau": best_epi["tau"],
                "coverage": best_epi["coverage"],
                "sel_acc": best_epi["sel_acc"],
                "sel_risk": best_epi["sel_risk"],
                "unique_labels": unique_labels,}, 
            'ale': {
                "snr": snr,
                "tau": best_ale["tau"],
                "coverage": best_ale["coverage"],
                "sel_acc": best_ale["sel_acc"],
                "sel_risk": best_ale["sel_risk"],
                "unique_labels": unique_labels,}, 
            'tot': {
                "snr": snr,
                "tau": best_tot["tau"],
                "coverage": best_tot["coverage"],
                "sel_acc": best_tot["sel_acc"],
                "sel_risk": best_tot["sel_risk"],
                "unique_labels": unique_labels,
            }}
        )

    averaged_curve = _average_sweep_results(sweep_by_snr)
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{ctx['task']}_snr_coverage.pkl")
    payload = {
        "per_snr": to_serializable(sweep_by_snr),
        "avg_curve": to_serializable(averaged_curve),
        "per_snr_best": to_serializable(best_by_snr),
    }
    if calibrate_on_train:
        payload["calibrated_tau"] = calibrated_tau
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    stats = {
        # "mean_sel_acc_curve": float(np.nanmean(averaged_curve["sel_acc"])) if averaged_curve["sel_acc"].size else float("nan"),
        # "mean_cov_curve": float(np.nanmean(averaged_curve["coverage"])) if averaged_curve["coverage"].size else float("nan"),
        # "mean_sel_acc_by_snr": float(np.nanmean([b["sel_acc"] for b in best_by_snr])) if best_by_snr else float("nan"),
        # "mean_cov_by_snr": float(np.nanmean([b["coverage"] for b in best_by_snr])) if best_by_snr else float("nan"),
    }

    # print(
    #     f"[{ctx['task']}] avg sel_acc (curve): {stats['mean_sel_acc_curve']:.3f}, "
    #     f"avg coverage (curve): {stats['mean_cov_curve']:.3f} -> saved to {output_path}"
    # )
    return stats, output_path


def save_snr_payload(task: str, payload: Dict, output_root: str):
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{task}_snr_results_noisy_test.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)
    return output_path


def run_uncertainty_distribution_exports(
    snr_upper_limit: float = -8,
    output_root: str = "eu_comparison/BLC/distrib",
    contexts: List[Dict] = None,
    remove_noise_only: bool = False,
) -> Dict[str, str]:
    """Export correctness/uncertainty payloads for each configured task."""
    results: Dict[str, str] = {}
    ctxs = contexts or [init_task_context(i, cfg) for i, cfg in enumerate(task_configs)]
    for ctx in ctxs:
        args.current_task = ctx["index"]
        cfg = task_configs[ctx["index"]]
        output_path = export_uncertainty_distribution_for_task(
            task_index=ctx["index"],
            config=cfg,
            snr_upper_limit=snr_upper_limit,
            context=ctx,
            output_root=output_root,
            remove_noise_only=remove_noise_only,
        )
        results[ctx["task"]] = output_path
    return results


def run_snr_experiment(
    target_coverage: float = 0.90,
    output_root: str = "eu_comparison/BLC",
    contexts: List[Dict] = None,
    remove_noise_only: bool = False,
    global_coverage: bool = False,
    calibrate_on_train: bool = False,
    uncertainty_key: str = "eu",
):
    """Run the selective SNR evaluation for all configured tasks and persist outputs."""
    summary = {}
    ctxs = contexts or [init_task_context(i, cfg) for i, cfg in enumerate(task_configs)]
    for i, ctx in enumerate(ctxs):
        args.current_task = ctx["index"]
        # if i ==0: continue
        cfg = task_configs[ctx["index"]]
        per_task_target = _resolve_target_coverage(target_coverage, ctx["task"], ctx["index"])
        payload, stats = evaluate_snr_range_for_task(
            ctx["index"],
            cfg,
            target_coverage=per_task_target,
            context=ctx,
            remove_noise_only=remove_noise_only,
            global_coverage=global_coverage,
            calibrate_on_train=calibrate_on_train,
            uncertainty_key=uncertainty_key,
        )
        save_path = save_snr_payload(ctx["task"], payload, output_root)
        print(f"[{ctx['task']}] mean selective acc: {stats['mean_sel_acc']:.2f}, "
              f"mean base acc: {stats['mean_base_acc']:.2f}",
              f"mean coverage: {stats['mean_cov']:.2f} -> saved to {save_path}")
        summary[ctx["task"]] = stats
    return summary


def evaluate_baseline_for_task(task_index: int, config: Dict, remove_noise_only: bool = False) -> Dict:
    """Run a single evaluation pass on the task's standard split."""
    ctx = init_task_context(task_index, config)
    labels, preds, unique_labels, eu, h_pred = evaluate_split(
        ctx["pipeline"],
        ctx["model"],
        ctx["trained_mask"],
        ctx["offset"],
        mixed_snrs=False,
        remove_noise_only=remove_noise_only,
    )
    ctx.update(
        {
            "labels": labels,
            "preds": preds,
            "unique_labels": unique_labels,
            "eu": eu,
            "hpred": h_pred,
        }
    )
    return ctx


def evaluate_coverage_for_task(
    task_index: int,
    config: Dict,
    target_coverage: float = 0.95,
    baseline: Dict = None,
    output_root: str = "eu_comparison/BLC/coverage",
    remove_noise_only: bool = False,
    calibrate_on_train: bool = False,
    uncertainty_key: str = "eu",
):
    """Evaluate coverage/accuracy trade-off for a single task and optionally save it."""
    base = baseline or evaluate_baseline_for_task(
        task_index,
        config,
        remove_noise_only=remove_noise_only,
    )
    cal_tau = None
    if calibrate_on_train:
        train_labels, train_preds, _, train_eu, train_hpred = evaluate_split(
            base["pipeline"],
            base["model"],
            base["trained_mask"],
            base["offset"],
            data_path=None,
            mixed_snrs=False,
            remove_noise_only=remove_noise_only,
            split="train",
        )
        train_ale = np.asarray([total - e for total, e in zip(train_hpred, train_eu)])
        train_tot = np.asarray(train_hpred)
        cal_tau = {
            "eu": float(choose_tau(sweep_thresholds(train_labels, train_preds, u=list(train_eu)), target_coverage=target_coverage)["tau"]),
            "ale": float(choose_tau(sweep_thresholds(train_labels, train_preds, u=list(train_ale)), target_coverage=target_coverage)["tau"]),
            "tot": float(choose_tau(sweep_thresholds(train_labels, train_preds, u=list(train_tot)), target_coverage=target_coverage)["tau"]),
        }
        print(
            f"[{base['task']}] calibrated taus on training data (target {target_coverage:.2f}): "
            f"eu={cal_tau['eu']:.6f}, ale={cal_tau['ale']:.6f}, tot={cal_tau['tot']:.6f}"
        )
    eu = np.asarray(base["eu"])
    ale = np.asarray([total - e for total, e in zip(base["hpred"], eu)])
    tot = np.asarray(base["hpred"])
    results = []
    sweep_by_key = {}
    for key, u in [("eu", eu), ("ale", ale), ("tot", tot)]:
        best, sweep_results = pick_threshold(base["labels"], base["preds"], u, target_coverage)
        sweep_by_key[key] = {"best": best, "sweep": sweep_results}
        results.append(sweep_results)
    payload = {"results": to_serializable(results)}
    if calibrate_on_train:
        payload["calibrated_tau"] = cal_tau
    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{base['task']}_coverage_data_noisy_dcheck.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    key_map = {"h_pred": "tot"}
    selected_key = key_map.get(uncertainty_key, uncertainty_key)
    if selected_key not in sweep_by_key:
        raise ValueError(f"Unsupported uncertainty_key: {uncertainty_key}")
    best = sweep_by_key[selected_key]["best"]

    stats = {
        "sel_acc": best["sel_acc"],
        "coverage": best["coverage"],
        "tau": best["tau"],
    }
    if calibrate_on_train:
        cal_key = "tot" if uncertainty_key == "h_pred" else uncertainty_key
        accept = (tot if cal_key == "tot" else eu if cal_key == "eu" else ale) <= cal_tau[cal_key]
        coverage = float(accept.mean()) if eu.size else 0.0
        if accept.any():
            sel_acc = float(accuracy(torch.tensor(base["labels"][accept]), torch.tensor(base["preds"][accept])))
            sel_risk = 1.0 - sel_acc
        else:
            sel_acc = float("nan")
            sel_risk = float("nan")
        stats.update(
            {
                "calibrated_tau": cal_tau[cal_key],
                "calibrated_coverage": coverage,
                "calibrated_sel_acc": sel_acc,
                "calibrated_sel_risk": sel_risk,
            }
        )

    print(
        f"[{base['task']}] base acc: {accuracy(torch.tensor(base['labels']), torch.tensor(base['preds'])):.2f}, "
        f"sel_acc@{best['coverage']:.2f}: {best['sel_acc']:.2f}, tau: {best['tau']:.4f} -> saved to {output_path}"
    )
    return stats, output_path


def _plot_confusion_matrix_ax(ax, labels, preds, unique_labels, title: str):
    cm = confusion_matrix(labels, preds, labels=unique_labels, normalize="true") * 100
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)


def _plot_confusion_matrix(base: Dict, dataset_name: str):
    """Plot a confusion matrix for a task."""
    fig, ax = plt.subplots(1, 1, dpi=200, figsize=(6, 5))
    title = f"Confusion Matrix ({dataset_name} - {base['task']})"
    _plot_confusion_matrix_ax(ax, base["labels"], base["preds"], base["unique_labels"], title)
    plt.tight_layout()
    return fig


def evaluate_snr_confusion_for_task(
    task_index: int,
    config: Dict,
    snr_upper_limit: float = -8,
    context: Dict = None,
    remove_noise_only: bool = False,
) -> Dict:
    """Aggregate labels/predictions across SNRs for confusion matrix plotting."""
    ctx = dict(context) if context is not None else init_task_context(task_index, config)

    if task_index >= len(dataset_paths):
        raise IndexError(f"No dataset path configured for task index {task_index}")
    data_root = dataset_paths[task_index]

    labels_all: List[int] = []
    preds_all: List[int] = []
    unique_labels = None

    for snr in snr_range:
        if snr_upper_limit is not None and snr > snr_upper_limit:
            break
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        labels, preds, labels_unique, _, _ = evaluate_split(
            ctx["pipeline"],
            ctx["model"],
            ctx["trained_mask"],
            ctx["offset"],
            data_path=snr_path,
            mixed_snrs=True,
            remove_noise_only=remove_noise_only,
        )
        labels_all.extend(labels)
        preds_all.extend(preds)
        if unique_labels is None:
            unique_labels = labels_unique

    if unique_labels is None:
        unique_labels = np.unique(labels_all) if labels_all else np.array([])

    ctx.update(
        {
            "labels": np.asarray(labels_all),
            "preds": np.asarray(preds_all),
            "unique_labels": np.asarray(unique_labels),
        }
    )
    return ctx


def run_confusion_plots(
    baselines: List[Dict] = None,
    output_dir: str = None,
    dataset_name: str = None,
    remove_noise_only: bool = False,
) -> List[str]:
    """Generate confusion matrix plots for each task."""
    paths = []
    base_outputs = baselines or [
        evaluate_baseline_for_task(i, config, remove_noise_only=remove_noise_only)
        for i, config in enumerate(task_configs)
    ]
    for base in base_outputs:
        fig = _plot_confusion_matrix(base, dataset_name or dataset)
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{base['task']}_cm_pred_uncertainty.png")
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        if save_path:
            print(f"[{base['task']}] confusion matrix plot saved to {save_path}")
            paths.append(save_path)
    return paths


def run_snr_confusion_comparison(
    baselines: List[Dict] = None,
    contexts: List[Dict] = None,
    output_dir: str = None,
    dataset_name: str = None,
    snr_upper_limit: float = -8,
    remove_noise_only: bool = False,
) -> List[str]:
    """Generate before/after confusion matrices comparing baseline vs SNR data."""
    paths = []
    base_outputs = baselines or [
        evaluate_baseline_for_task(i, config, remove_noise_only=remove_noise_only)
        for i, config in enumerate(task_configs)
    ]
    ctxs = contexts or [init_task_context(i, cfg) for i, cfg in enumerate(task_configs)]

    ctx_by_index = {ctx["index"]: ctx for ctx in ctxs}

    for base in base_outputs:
        cfg = task_configs[base["index"]]
        snr_ctx = evaluate_snr_confusion_for_task(
            base["index"],
            cfg,
            snr_upper_limit=snr_upper_limit,
            context=ctx_by_index.get(base["index"]),
            remove_noise_only=remove_noise_only,
        )
        fig, axs = plt.subplots(1, 2, dpi=200, figsize=(12, 5))
        ds_name = dataset_name or dataset
        _plot_confusion_matrix_ax(
            axs[0],
            base["labels"],
            base["preds"],
            base["unique_labels"],
            f"Baseline ({ds_name} - {base['task']})",
        )
        _plot_confusion_matrix_ax(
            axs[1],
            snr_ctx["labels"],
            snr_ctx["preds"],
            snr_ctx["unique_labels"],
            f"SNR ≤ {snr_upper_limit} dB ({ds_name} - {base['task']})",
        )
        plt.tight_layout()

        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"{base['task']}_cm_before_after_snr.png")
            fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        if save_path:
            print(f"[{base['task']}] baseline vs SNR confusion plot saved to {save_path}")
            paths.append(save_path)
    return paths


def run_coverage_experiment(
    target_coverage: float = 0.95,
    output_root: str = "eu_comparison/BLC/coverage",
    baselines: List[Dict] = None,
    remove_noise_only: bool = False,
    calibrate_on_train: bool = False,
    uncertainty_key: str = "eu",
):
    """Evaluate coverage curves for all tasks, reusing baselines if provided."""
    results = {}
    base_outputs = baselines or [
        evaluate_baseline_for_task(i, cfg, remove_noise_only=remove_noise_only)
        for i, cfg in enumerate(task_configs)
    ]
    for base in base_outputs:
        cfg = task_configs[base["index"]]
        per_task_target = _resolve_target_coverage(target_coverage, base["task"], base["index"])
        stats, output_path = evaluate_coverage_for_task(
            base["index"],
            cfg,
            target_coverage=per_task_target,
            baseline=base,
            output_root=output_root,
            remove_noise_only=remove_noise_only,
            calibrate_on_train=calibrate_on_train,
            uncertainty_key=uncertainty_key,
        )
        results[base["task"]] = {"stats": stats, "path": output_path}
    return results


def run_snr_coverage_experiment(
    target_coverage: float = 0.95,
    output_root: str = "eu_comparison/BLC/snr_coverage",
    contexts: List[Dict] = None,
    remove_noise_only: bool = False,
    calibrate_on_train: bool = False,
):
    """Average selective accuracy/coverage curves across SNR values for each task."""
    results = {}
    ctxs = contexts or [init_task_context(i, cfg) for i, cfg in enumerate(task_configs)]
    for ctx in ctxs:
        current_task = ctx["index"]
        # if current_task == 0: continue
        args.current_task = current_task
        cfg = task_configs[current_task]
        per_task_target = _resolve_target_coverage(target_coverage, ctx["task"], current_task)
        stats, output_path = evaluate_snr_coverage_for_task(
            ctx["index"],
            cfg,
            target_coverage=per_task_target,
            context=ctx,
            output_root=output_root,
            remove_noise_only=remove_noise_only,
            calibrate_on_train=calibrate_on_train,
        )
        results[ctx["task"]] = {"stats": stats, "path": output_path}
    return results


def collect_baselines(remove_noise_only: bool = False) -> List[Dict]:
    """Helper to build baseline outputs once and reuse for multiple stages."""
    return [
        evaluate_baseline_for_task(i, cfg, remove_noise_only=remove_noise_only)
        for i, cfg in enumerate(task_configs)
    ]


if __name__ == "__main__":
    baselines = collect_baselines()
    run_snr_experiment()
    run_coverage_experiment(baselines=baselines)
    run_confusion_plots(baselines=baselines, output_dir=os.path.join(datasets[dataset]["models"], "figs"))
