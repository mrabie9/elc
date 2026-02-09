import torch
import pickle
import os


class args_class:
    def __init__(self, base_path=None):
        self.arch = "evidential"
        self.base_path = base_path
        self.multi_head = True

dataset = "Radar_no-noise_kll_udist-10"
save_figs = True
datasets = {
    "DRC" : {"models" : "dronerc/exp-drc-1024sl-5/", "tasks" : "dronerc/tasks-1024sl/", "num_classes":17},
    "USRP"    : {"models" : "usrp/exp-usrp-DS-highlr-2/", "tasks" : "usrp/tasks - 1t-1024slices-norm-3tasks/", "num_classes":18},
    "LoRa"    : {"models" : "rfmls/exp-lora-2/", "tasks" : "rfmls/tasks_lp_downsampled/", "num_classes":10},
    "mixed-comms"    : {"models" : "mixed/exp-mixed-sm/", "tasks" : "mixed/tasks-sm/", "num_classes":15, "offset":[0]},
    "Radar"    : {"models" : "radar/exp-radar-mixed-3-nodyn/", "tasks" : "radar/tasks-mixed/", "num_classes":11, "offset": [0,6,11], "cpt":None},
    "Radar-2"    : {"models" : "radar/exp-radar-elc/", "tasks" : "radar/tasks-mixed/", "num_classes":11, "offset": [0,6]},
    "Radar_noisy"    : {"models" : "radar/exp-radar-elc-noisy-radhcar/", "tasks" : "radar/tasks-noisy-radchar/", "num_classes":12, "offset": [0,6]},
    "Radar_noisy_dcheck"    : {"models" : "radar/exp-radar-elc-noisy-radhcar-nobn/", "tasks" : "radar/tasks-noisy-radchar/", "num_classes":12, "offset": [0,6], "cpt":[6,6]},
    "Radar_no-noise"    : {"models" : "radar/exp-radar-elc-no-noise/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll"    : {"models" : "radar/exp-radar-elc-no-noise-kll/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-05"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-05/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-1"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-1/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-5"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-5/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10-c"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10 copy/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-15"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-20"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-50"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10-005g"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10-005g/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10-02g"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10-02g/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10-10p"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10-10p/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
    "Radar_no-noise_kll_udist-10-40p"    : {"models" : "radar/exp-radar-elc-no-noise-kll-udist-10-40p/", "tasks" : "radar/tasks-no-noise/", "num_classes":10, "offset": [0,5], "cpt":[5,5]},
}
args = args_class("")
args.multi_head = True
args.disjoint_classifier = not args.multi_head
args.pruned_layer = None
# args.base_path = "radar"
# args.current_task = 0
model_path = datasets[dataset]['models']
args.classes_per_task = datasets[dataset].get("cpt")
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
