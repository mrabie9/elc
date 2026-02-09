#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CONFIG_DIRS=(dronerc mixed radar rfmls usrp)
failures=()

for dir in "${CONFIG_DIRS[@]}"; do
    config_path="${ROOT_DIR}/${dir}/args.yaml"
    if [[ ! -f "${config_path}" ]]; then
        echo "Skipping ${dir}: config not found at ${config_path}" >&2
        failures+=("${dir} (missing config)")
        continue
    fi

    echo "=== Running ${dir} using ${config_path} ==="
    if ! "${PYTHON_BIN}" "${ROOT_DIR}/main.py" --config "${config_path}" "$@"; then
        echo "Experiment for ${dir} failed" >&2
        failures+=("${dir}")
    fi
done

if ((${#failures[@]})); then
    echo "Completed with failures:" >&2
    printf ' - %s\n' "${failures[@]}" >&2
    exit 1
fi

echo "All experiments completed successfully."
