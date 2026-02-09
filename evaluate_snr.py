import os
import numpy as np
import matplotlib.pyplot as plt
from models.bayes_resnet import Net as ResNet18_1d
from TrainValTest import CVTrainValTest
import torch
import pickle

def save_test_outputs(output_dir, results, aucs, filename="snr_results.pkl"):
    """Persist the key tensors/lists returned by test_model so they can be reused later."""
    os.makedirs(output_dir, exist_ok=True)

    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value
# mega, pred_class, util, inv_util
    payload = {
        "results": _to_serializable(results),
        "AUC": _to_serializable(aucs)
        }

    with open(os.path.join(output_dir, filename), "wb") as handle:
        pickle.dump(payload, handle)

def evaluate_model_on_snr_datasets(model, pipeline, args, folder, snr_range, mask, offset):
    accs, uncerts = [], []
    for snr in snr_range:
        filename = f"{snr}db.npz" if snr < 0 else f"{snr}db.npz"
        path = os.path.join(folder, filename)
        print(path)
        if not os.path.exists(path):
            accs.append(np.nan)
            uncerts.append(np.nan)
            continue
        # data = np.load(path)
        # x = data['xte']
        # y = data['yte']

        train_loader = pipeline.load_data_dronerc(256, offset=offset, data = path, mixed_snrs=True, args=args)
        # loader = pipeline.convert_to_loader(x, y, batch_size=256)

        acc= pipeline.test_model(
                args, model, mask, eval_entropy=True,cm=False, enable_diagnostics=False)
        accs.append(acc)
        uncerts.append([])
    return accs, uncerts

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
snr_range = list(range(-20, 20, 2))

# Paths to .npz datasets
dataset_paths = [ "/home/lunet/wsmr11/repos/radar/snr_splits_radnist","/home/lunet/wsmr11/repos/radar/snr_splits_radchar_noisy",]
labels = ["RadNIST", "RadChar"]

# Args and pipeline
class args_class():
    def __init__(self):
        self.multi_head = False
        self.current_task = 0
        self.base_path = ""
        self.offset = 0
        self.arch = "bayes_rfnet"
        self.disjoint_classifier = False
args=args_class()
args.disjoint_classifier = False#True#False#True#False#True#False#True#False
args.tasks = 2
args.multi_head = True
# --- Plot ---
plt.figure(figsize=(10, 6), dpi=200)
j=0
colors = ['#1f77b4', '#5fa2ce']
for i,model_path in enumerate(["radar/exp-noisy-radchar/task0/retrained.pt", "radar/exp-noisy-radchar/task1/cumu_model.pt",]):
    if i==0: continue
    print(model_path)
    path = dataset_paths[0] if "task0" in model_path else dataset_paths[1]
    mask = "radar/radar-bresnet-tasks/task0/cumu_mask.pkl" if "task0" in model_path else None
    task = "task0" if "task0" in model_path else "task1"
    args.current_task = 1 if "task1" in model_path else 0
    offset = 0 if "task0" in model_path else 0
    model = ResNet18_1d(input_size=1024, num_classes=12, args=args)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    pipeline = CVTrainValTest(base_path="", save_path="")

    # --- Evaluate ---
    results, aucs = [], []
    # for path in dataset_paths:
    accs, uncerts = evaluate_model_on_snr_datasets(model, pipeline, args, path, snr_range, mask, offset)
    results.append((accs, uncerts))

    for i, (accs, uncerts) in enumerate(results):
        plt.plot(snr_range, accs, label=f"{labels[j]} Accuracy", color=colors[j])
        plt.plot(snr_range, uncerts, linestyle='-.', label=f"{labels[j]} Uncertainty", color=colors[j])
    j+=1
    save_test_outputs("eu_comparison/BLC", results, aucs, filename=f"{task}_snr_results_noisy.pkl")

# plt.legend(["RadNIST Accuracy","RadNIST Uncertainty","RadChar Accuracy","RadChar Uncertainty"])
plt.xlabel("SNR (dB)")
plt.ylabel("Performance (%)")
# plt.title("Model Accuracy and Uncertainty vs SNR")
plt.grid(True)
plt.legend(fontsize=7)
plt.tight_layout()
plt.show()
