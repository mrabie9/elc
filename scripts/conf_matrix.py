import os
import sys

# Allow running the script directly
sys.path.append("/home/lunet/wsmr11/repos/LPSforECNN/")

from scripts.helper import dataset, datasets
from scripts.mod_conf_matrix import (
    collect_baselines,
    run_confusion_plots,
    run_coverage_experiment,
    run_snr_coverage_experiment,
    run_snr_experiment,
    run_snr_confusion_comparison,
    export_uncertainty_distribution_for_task,
)

# These flags mirror the manual toggles in the legacy script.
RUN_SNR_EVAL = True
RUN_SNR_COVERAGE = False
RUN_COVERAGE = True
RUN_CONFUSION = False
RUN_SNR_CONFUSION_COMPARE = False
RUN_EU_CORRECTNESS = True
REMOVE_NOISE_ONLY_CLASS = False
SNR_GLOBAL_COVERAGE = True
SNR_CALIBRATE_ON_TRAIN = True
SNR_COVERAGE_CALIBRATE_ON_TRAIN = SNR_CALIBRATE_ON_TRAIN
COVERAGE_CALIBRATE_ON_TRAIN = SNR_CALIBRATE_ON_TRAIN
SELECTIVE_UNCERTAINTY = "h_pred"  # options: "eu", "h_pred"

TARGET_COVERAGE_SNR = {"task0": 0.80, "task1": 0.80}  # RadNIST, RadChar
TARGET_COVERAGE_BASE = {"task0": 0.95, "task1": 0.95}
SNR_OUTPUT_ROOT = "eu_comparison/BLC"
COVERAGE_SNR_OUTPUT_ROOT = "eu_comparison/BLC/snr_coverage"
COVERAGE_OUTPUT_ROOT = "eu_comparison/BLC/coverage"
CONFUSION_FIG_DIR = os.path.join(datasets[dataset]["models"], "figs")

RADNIST = 0
RADCHAR = 1
task = [0,1]


if __name__ == "__main__":
    baselines = (
        collect_baselines(remove_noise_only=REMOVE_NOISE_ONLY_CLASS)
        if (RUN_COVERAGE or RUN_CONFUSION or RUN_SNR_COVERAGE or RUN_SNR_CONFUSION_COMPARE)
        else None
    )

    if RUN_SNR_EVAL:
        run_snr_experiment(
            target_coverage=TARGET_COVERAGE_SNR,
            output_root=SNR_OUTPUT_ROOT,
            contexts=baselines,
            remove_noise_only=REMOVE_NOISE_ONLY_CLASS,
            global_coverage=SNR_GLOBAL_COVERAGE,
            calibrate_on_train=SNR_CALIBRATE_ON_TRAIN,
            uncertainty_key=SELECTIVE_UNCERTAINTY,
        )
    if RUN_SNR_COVERAGE:
        run_snr_coverage_experiment(
            target_coverage=TARGET_COVERAGE_SNR,
            output_root=COVERAGE_SNR_OUTPUT_ROOT,
            contexts=baselines,
            remove_noise_only=REMOVE_NOISE_ONLY_CLASS,
            calibrate_on_train=SNR_COVERAGE_CALIBRATE_ON_TRAIN,
        )
    if RUN_COVERAGE:
        run_coverage_experiment(
            target_coverage=TARGET_COVERAGE_BASE,
            output_root=COVERAGE_OUTPUT_ROOT,
            baselines=baselines,
            remove_noise_only=REMOVE_NOISE_ONLY_CLASS,
            calibrate_on_train=COVERAGE_CALIBRATE_ON_TRAIN,
            uncertainty_key=SELECTIVE_UNCERTAINTY,
        )
    if RUN_CONFUSION:
        run_confusion_plots(
            baselines=baselines,
            output_dir=CONFUSION_FIG_DIR,
            dataset_name=dataset,
            remove_noise_only=REMOVE_NOISE_ONLY_CLASS,
        )
    if RUN_SNR_CONFUSION_COMPARE:
        run_snr_confusion_comparison(
            baselines=baselines,
            output_dir=CONFUSION_FIG_DIR,
            dataset_name=dataset,
            remove_noise_only=REMOVE_NOISE_ONLY_CLASS,
        )

    if RUN_EU_CORRECTNESS:
        for t in task:
            export_uncertainty_distribution_for_task(
                task_index = t,
                snr_upper_limit = -10,
                context = None,
                output_root = "eu_comparison/BLC/distrib",
                remove_noise_only = REMOVE_NOISE_ONLY_CLASS,
            )
