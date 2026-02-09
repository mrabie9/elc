import torch
import pickle
import os


class args_class:
    def __init__(self, base_path=None):
        self.current_task = 0
        self.tasks = 2
        self.arch = "bayes_rfnet"
        self.disjoint_classifier = False

dataset = "radar_no-noise-bndo"
save_figs = True
datasets = {
    "DRC" : {"models" : "dronerc/exp-drc-1024sl-redo-ln-025prune/", "tasks" : "dronerc/tasks-1024sl/", "num_classes":17, "offset": [0, 5, 11]},
    "USRP"    : {"models" : "usrp/exp-usrp-DS-lower-v-adam-layernorm/", "tasks" : "usrp/tasks - 1t-1024slices-norm-3tasks/", "num_classes":18,  "offset": [0, 6, 12]},
    "LoRa"    : {"models" : "rfmls/exp-lora-redo-nonorm/", "tasks" : "rfmls/tasks_lp_downsampled/", "num_classes":10, "offset": [0, 5]},
    "radar"    : {"models" : "radar/exp-radar-mixed-3-nodyn/", "tasks" : "radar/tasks-mixed/", "num_classes":11, "offset": [0, 5]},
    "mixed"    : {"models" : "mixed/exp-mixed-2/", "tasks" : "mixed/tasks/", "num_classes":15, "offset": [0, 5, 10]},
    "DRC_B" : {"models" : "dronerc/drc-bresnet-sm/", "tasks" : "dronerc/tasks-sm/", "num_classes":15, "offset": [0, 0, 0]},
    "USRP_B"    : {"models" : "usrp/usrp-bresnet-sm/", "tasks" : "usrp/tasks-sm/", "num_classes":18,  "offset": [0, 6, 12]},
    "LoRa_B"    : {"models" : "rfmls/rfmls-bresnet-sm/", "tasks" : "rfmls/tasks-sm/", "num_classes":10, "offset": [0, 5]},
    "radar_B"    : {"models" : "radar/radar-bresnet-tasks/", "tasks" : "radar/tasks-mixed/", "num_classes":11, "offset": [0, 5]},
    "radar_redo"    : {"models" : "radar/exp-noisy-radchar-dcheck-redo-fulldata/", "tasks" : "radar/tasks-noisy-radchar/", "num_classes":12, "offset": [0, 6]},
    "radarsm_B"    : {"models" : "radar/radar-bresnet-sm/", "tasks" : "radar/tasks-sm/", "num_classes":12, "offset": [0, 5]},
    "mixed_B"    : {"models" : "mixed/mixed-bresnet-tasks/", "tasks" : "mixed/tasks/", "num_classes":15, "offset": [0, 5,10]},
    "radar_no-noise"    : {"models" : "radar/exp-bresnet-no-noise/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0, 5]},
    "radar_no-noise-bndo"    : {"models" : "radar/exp-blc-no-noise-bn-do/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0, 5]},
}

args = args_class("")
args.multi_head = True
args.pruned_layer = None
# args.base_path = "radar"
# args.current_task = 0
model_path = datasets[dataset]['models']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_configs = [
    {"file": "task0/cumu_model.pt", "task": "task0", "offset": 0},
    {"file": "task1/cumu_model.pt", "task": "task1", "offset": 5},
    # {"file": "task2/rfnet16.pt", "task": "task2", "offset": 11},
]

dataset_paths = ["/home/lunet/wsmr11/repos/radar/snr_splits_radnist_no-noise", "/home/lunet/wsmr11/repos/radar/snr_splits_radchar", ]
snr_range = list(range(-20, 20, 2))

def to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value

def save_test_outputs(output_dir, all_labels, all_preds, unique_labels, omega, pred_class, util, inv_utils,filename="test_outputs.pkl"):
    """Persist the key tensors/lists returned by test_model so they can be reused later."""
    os.makedirs(output_dir, exist_ok=True)

    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value
# mega, pred_class, util, inv_util
    payload = {
        "all_labels": _to_serializable(all_labels),
        "all_preds": _to_serializable(all_preds),
        "unique_labels": _to_serializable(unique_labels),
        "omega": _to_serializable(omega),
        "pred_class": _to_serializable(pred_class),
        "util": _to_serializable(util),
        "inv_utils": _to_serializable(inv_utils),
    }

    with open(filename, "wb") as handle:
        pickle.dump(payload, handle)

    return os.path.join(output_dir, filename)

def load_test_outputs(file_path):
    """Restore the cached outputs produced by save_test_outputs."""
    if os.path.exists(file_path) is False:
        raise FileNotFoundError(f"No saved outputs found at {file_path}")

    with open(file_path, "rb") as handle:
        payload = pickle.load(handle)

    return (
        payload["all_labels"],
        payload["all_preds"],
        payload["unique_labels"],
        payload["omega"],
        payload["pred_class"],
        payload["util"],
        payload["inv_utils"],
    )
