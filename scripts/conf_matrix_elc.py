import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required: {exc}")

# Allow running the script directly
sys.path.append("/home/lunet/wsmr11/repos/LPSforECNN")

from scripts.helper_elc import (  # noqa: E402
    args,
    dataset,
    datasets,
    dataset_paths,
    snr_range,
    task_configs,
)
from models.dst_resnet import ResNet18_1d  # noqa: E402
from TrainValTest import CVTrainValTest, accuracy  # noqa: E402
from sweep_eu import choose_tau, sweep_thresholds  # noqa: E402


def load_model_and_mask(task: str, model_file: str):
    model = ResNet18_1d(slice_size=1024, num_classes=datasets[dataset]["num_classes"], classes_per_task=datasets[dataset]["cpt"])
    print(f"Loading model from {model_file} for task {task}...")
    model.load_state_dict(torch.load(os.path.join(datasets[dataset]["models"], model_file)))

    mask_path = os.path.join(datasets[dataset]["models"], f"{task}/cumu_mask.pkl")
    trained_mask = pickle.load(open(mask_path, "rb")) if os.path.exists(mask_path) else None

    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model, trained_mask


def prepare_pipeline(task: str) -> CVTrainValTest:
    base_path = datasets[dataset]["tasks"] + task
    return CVTrainValTest(base_path=base_path, save_path=base_path)


def _collect_features(
    pipeline: CVTrainValTest,
    model: torch.nn.Module,
    trained_mask,
    offset: int,
    data_path: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipeline.load_data_dronerc(256, offset=offset, data=data_path, mixed_snrs=data_path is not None, args=args)
    labels, preds, _, _, _, util, ds_omega = pipeline.test_model(
        args, model, trained_mask, cm=True, enable_diagnostics=False, mixed_snrs=data_path is not None
    )
    if ds_omega is None:
        raise ValueError("DS omega not returned from test_model.")

    labels = np.asarray(labels) - offset
    preds = np.asarray(preds) - offset
    incorrect = (preds != labels).astype(np.int32)

    util_max, _ = torch.max(util, dim=1)
    util_unc = (1 - util_max).detach().cpu().numpy()
    ds_omega_np = ds_omega.detach().cpu().numpy()

    X = np.stack([ds_omega_np, util_unc], axis=1)
    return X, incorrect, preds, labels


def fit_calibrator(X: np.ndarray, y: np.ndarray) -> Tuple[LogisticRegression, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xs, y)
    return clf, scaler


def evaluate_snr_range_with_calibrator(
    task_index: int,
    cfg: Dict,
    target_coverage: float,
    output_root: str,
    tau_source: str = "global_snr",
    metric: str = "calibrated",
) -> Tuple[Dict, Dict]:
    args.current_task = task_index
    model, trained_mask = load_model_and_mask(cfg["task"], cfg["file"])
    pipeline = prepare_pipeline(cfg["task"])
    offset = datasets[dataset]["offset"][task_index]

    # Fit calibrator on base split (no SNR path)
    X_base, y_base, preds_base, labels_base = _collect_features(
        pipeline, model, trained_mask, offset=offset, data_path=None
    )
    clf = None
    scaler = None
    if metric == "calibrated":
        clf, scaler = fit_calibrator(X_base, y_base)

    data_root = dataset_paths[task_index]
    per_snr: List[Dict] = []

    # First pass: collect all SNR samples to choose a single global tau.
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_p_error: List[float] = []
    snr_cache: Dict[int, Dict] = {}

    for snr in snr_range:
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        X_snr, _, preds, labels = _collect_features(
            pipeline, model, trained_mask, offset=offset, data_path=snr_path
        )
        if metric == "calibrated":
            p_error = clf.predict_proba(scaler.transform(X_snr))[:, 1]
        else:
            p_error = X_snr[:, 1]

        all_labels.extend(labels.tolist())
        all_preds.extend(preds.tolist())
        all_p_error.extend(p_error.tolist())

        snr_cache[snr] = {
            "labels": labels,
            "preds": preds,
            "p_error": p_error,
        }

    # Choose a single tau to meet target coverage using requested source.
    if tau_source == "train":
        if metric == "calibrated":
            p_base = clf.predict_proba(scaler.transform(X_base))[:, 1]
        else:
            p_base = X_base[:, 1]
        sweep_results = sweep_thresholds(np.array(labels_base), np.array(preds_base), u=list(p_base))
        best = choose_tau(sweep_results, target_coverage=target_coverage)
    else:
        sweep_results = sweep_thresholds(np.array(all_labels), np.array(all_preds), u=list(all_p_error))
        best = choose_tau(sweep_results, target_coverage=target_coverage)
    global_tau = best["tau"]

    # Second pass: evaluate per SNR using the global tau.
    sel_accs: List[float] = []
    base_accs: List[float] = []
    covs: List[float] = []

    for snr in snr_range:
        labels = snr_cache[snr]["labels"]
        preds = snr_cache[snr]["preds"]
        p_error = snr_cache[snr]["p_error"]

        accept = p_error <= global_tau
        cov = float(accept.mean()) if len(accept) else 0.0
        sel_acc = float((preds[accept] == labels[accept]).mean()) if accept.any() else float("nan")
        sel_risk = 1.0 - sel_acc if np.isfinite(sel_acc) else float("nan")

        base_acc = accuracy(torch.tensor(labels), torch.tensor(preds)) * 100

        sel_accs.append(sel_acc)
        base_accs.append(base_acc)
        covs.append(cov)
        per_snr.append(
            {
                "snr": snr,
                "base_acc": base_acc,
                "tau": global_tau,
                "coverage": cov,
                "sel_acc": sel_acc,
                "sel_risk": sel_risk,
            }
        )
        print(
            f"[{cfg['task']}] SNR {snr} dB - Base acc: {base_acc:.2f}%, "
            f"Sel Recall: {sel_acc*100:.2f}% "
            f"Accepted: {cov*100:.2f}% (tau_source={tau_source})"
        )

    payload = {
        "per_snr": per_snr,
        "global_tau": global_tau,
        "target_coverage": target_coverage,
        "overall_coverage": best["coverage"],
        "overall_sel_acc": best["sel_acc"],
        "overall_sel_risk": best["sel_risk"],
        "tau_source": tau_source,
    }
    stats = {
        "mean_sel_acc": float(np.mean(sel_accs)) if sel_accs else float("nan"),
        "mean_base_acc": float(np.mean(base_accs)) if base_accs else float("nan"),
        "mean_cov": float(np.mean(covs)) if covs else float("nan"),
    }

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{cfg['task']}_snr_results_calibrated.pkl")
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)
    print(f"[{cfg['task']}] saved to {output_path}")
    return stats, output_path


def run_snr_experiment_calibrated(
    target_coverage: float,
    output_root: str,
    tau_source: str,
    metric: str,
):
    summary = {}
    for i, cfg in enumerate(task_configs):
        stats, _ = evaluate_snr_range_with_calibrator(
            i,
            cfg,
            target_coverage=target_coverage,
            output_root=output_root,
            tau_source=tau_source,
            metric=metric,
        )
        summary[cfg["task"]] = stats
    return summary


def run_coverage_experiment_calibrated(output_root: str, metric: str) -> Dict[str, Dict]:
    """
    Generate coverage curve data using calibrated p_error on the base split.
    Output format matches scripts/mod_conf_matrix.py (payload["results"]=sweep_results).
    """
    results = {}
    for i, cfg in enumerate(task_configs):
        args.current_task = i
        model, trained_mask = load_model_and_mask(cfg["task"], cfg["file"])
        pipeline = prepare_pipeline(cfg["task"])
        offset = datasets[dataset]["offset"][i]

        X_base, y_base, preds_base, labels_base = _collect_features(
            pipeline, model, trained_mask, offset=offset, data_path=None
        )
        if metric == "calibrated":
            clf, scaler = fit_calibrator(X_base, y_base)
            p_base = clf.predict_proba(scaler.transform(X_base))[:, 1]
        else:
            p_base = X_base[:, 1]

        sweep_results = sweep_thresholds(np.array(labels_base), np.array(preds_base), u=list(p_base))
        payload = {"results": sweep_results}

        os.makedirs(output_root, exist_ok=True)
        output_path = os.path.join(output_root, f"{cfg['task']}_coverage_data_calibrated.pkl")
        with open(output_path, "wb") as handle:
            pickle.dump(payload, handle)

        results[cfg["task"]] = {"path": output_path}
        print(f"[{cfg['task']}] coverage curve saved to {output_path}")
    return results


def export_uncertainty_distribution_for_task(
    task_index: int,
    snr_upper_limit: int,
    output_root: str,
    metric: str,
    tau_source: str = "global_snr",
) -> Dict:
    """
    Collect correctness labels and uncertainty values up to a maximum SNR (inclusive)
    using calibrated p_error or 1-max(util). Mirrors scripts/conf_matrix.py behavior.
    """
    cfg = task_configs[task_index]
    args.current_task = task_index
    model, trained_mask = load_model_and_mask(cfg["task"], cfg["file"])
    pipeline = prepare_pipeline(cfg["task"])
    offset = datasets[dataset]["offset"][task_index]

    # Fit calibrator on base split if needed.
    X_base, y_base, preds_base, labels_base = _collect_features(
        pipeline, model, trained_mask, offset=offset, data_path=None
    )
    clf = None
    scaler = None
    if metric == "calibrated":
        clf, scaler = fit_calibrator(X_base, y_base)

    if task_index >= len(dataset_paths):
        raise IndexError(f"No dataset path configured for task index {task_index}")
    data_root = dataset_paths[task_index]

    snrs = [snr for snr in snr_range if snr <= snr_upper_limit]
    if not snrs:
        raise ValueError(f"No SNR values at or below {snr_upper_limit} found in snr_range.")

    all_correct: List[bool] = []
    all_u: List[float] = []

    for snr in snrs:
        snr_path = os.path.join(data_root, f"{snr}db.npz")
        X_snr, _, preds, labels = _collect_features(
            pipeline, model, trained_mask, offset=offset, data_path=snr_path
        )
        if metric == "calibrated":
            u = clf.predict_proba(scaler.transform(X_snr))[:, 1]
        else:
            u = X_snr[:, 1]
        correct = preds == labels
        all_correct.extend(correct.tolist())
        all_u.extend(u.tolist())

    os.makedirs(output_root, exist_ok=True)
    output_path = os.path.join(output_root, f"{cfg['task']}_uncertainty_distr_calibrated.pkl")
    payload = {
        "all_correct": all_correct,
        "all_eu": all_u,
        "snrs_used": snrs,
        "snr_upper_limit": snr_upper_limit,
        "task": cfg["task"],
        "metric": metric,
        "tau_source": tau_source,
    }
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle)

    print(f"[{cfg['task']}] saved {len(all_correct)} samples to {output_path}")
    return {"count": len(all_correct), "path": output_path, "task": cfg["task"]}


def main():
    parser = argparse.ArgumentParser(description="Calibrated SNR experiment using logistic p_error.")
    parser.add_argument("--target-coverage", type=float, default=0.90)
    parser.add_argument("--output-root", default="eu_comparison/ELC_kll")
    parser.add_argument(
        "--tau-source",
        choices=["global_snr", "train"],
        default="train",
        help="Choose tau using all SNR data or base/train data only.",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage curve (non-SNR) using calibrated p_error.",
    )
    parser.add_argument(
        "--export-uncertainty",
        action="store_true",
        help="Export uncertainty distribution up to snr_upper_limit (like conf_matrix.py).",
    )
    parser.add_argument("--snr-upper-limit", type=int, default=0)
    parser.add_argument(
        "--metric",
        choices=["calibrated", "util"],
        default="calibrated",
        help="Use calibrated p_error or raw 1-max(util).",
    )
    args_cli = parser.parse_args()

    if args_cli.export_uncertainty:
        for i, _cfg in enumerate(task_configs):
            export_uncertainty_distribution_for_task(
                i,
                snr_upper_limit=args_cli.snr_upper_limit,
                output_root=args_cli.output_root,
                metric=args_cli.metric,
                tau_source=args_cli.tau_source,
            )
    elif args_cli.coverage:
        run_coverage_experiment_calibrated(
            output_root=args_cli.output_root,
            metric=args_cli.metric,
        )
    else:
        run_snr_experiment_calibrated(
            target_coverage=args_cli.target_coverage,
            output_root=args_cli.output_root,
            tau_source=args_cli.tau_source,
            metric=args_cli.metric,
        )


if __name__ == "__main__":
    main()
