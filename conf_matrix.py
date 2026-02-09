from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for prettier heatmaps
import torch
# from models.masknet import ResNet50_1d, ResNet18_1d
from models.bayes_resnet import Net as ResNet18_1d
from TrainValTest import CVTrainValTest, accuracy
import pickle
import numpy as np
import pandas as pd
import os
from testers import test_sparsity_mask
from plots import plot_hexbin, plot_kde2d, plot_multiclass_roc_auc, get_auc, plot_risk_cov, plot_acc_cov
from sweep_eu import sweep_thresholds, choose_tau, top2_acc_rejected

import warnings
warnings.filterwarnings("ignore")

def to_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value

def save_test_outputs(output_dir, all_labels, all_preds, unique_labels, eu, h_pred, f_eu, filename="test_outputs.pkl"):
    """Persist the key tensors/lists returned by test_model so they can be reused later."""
    os.makedirs(output_dir, exist_ok=True)

    def _to_serializable(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    payload = {
        "all_labels": _to_serializable(all_labels),
        "all_preds": _to_serializable(all_preds),
        "unique_labels": _to_serializable(unique_labels),
        "eu": _to_serializable(eu),
        "h_pred": _to_serializable(h_pred),
        "f_eu": _to_serializable(f_eu),
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
        payload["eu"],
        payload["h_pred"],
        payload["f_eu"],
    )

class empty():
    def __init__(self):
        self.current_task = 0
        self.tasks = 3
        self.arch = "bayes_rfnet"
        self.disjoint_classifier = False

args = empty()
args.multi_head = False
file = "task0/cumu_model.pt"
task = "task0"
offset = 0
dataset = "radar_redo"
save_figs = False
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
    "radar_redo"    : {"models" : "radar/exp-noisy-radchar-dcheck/", "tasks" : "radar/tasks-noisy-radchar/", "num_classes":12, "offset": [0, 5]},
    "radarsm_B"    : {"models" : "radar/radar-bresnet-sm/", "tasks" : "radar/tasks-sm/", "num_classes":11, "offset": [0, 5]},
    "mixed_B"    : {"models" : "mixed/mixed-bresnet-tasks/", "tasks" : "mixed/tasks/", "num_classes":15, "offset": [0, 5,10]},
    # "mixed_B"    : {"models" : "mixed/exp-mixed-2/", "tasks" : "mixed/tasks/", "num_classes":15, "offset": [0, 5, 10]},
}
model_path = datasets[dataset]['models']
if os.path.exists(model_path) is False: raise ValueError("Model path does not exist:", model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_configs = [
    {"file": "task0/retrained.pt", "task": "task0"},
    {"file": "task1/cumu_model.pt", "task": "task1"},
    # {"file": "task2/bayes_rfnet16.pt", "task": "task2"},
]
args.disjoint_classifier = False#True#False#True#False#True#False#True#False
args.tasks = len(task_configs)
# print(len(task_configs))

#snr
plt.figure(figsize=(10, 6), dpi=200)
j=0
colors = ['#1f77b4', 'maroon']
labels = ["RadNIST", "RadChar"]
dataset_paths = ["/home/lunet/wsmr11/repos/radar/snr_splits_radnist","/home/lunet/wsmr11/repos/radar/snr_splits_radchar_noisy", ]
snr_range = list(range(-20, 20, 2))

fig, axs = plt.subplots(3, 2, dpi=200, figsize=(12, 10))  # 3 tasks × (CM + uncertainty)
for i, config in enumerate(task_configs):
    # if i == 0: continue
    file = config["file"]
    task = config["task"]
    offset = datasets[dataset]["offset"][i]
    args.current_task = i
    print(model_path + file)
    # Load model and mask
    model = ResNet18_1d(1024, datasets[dataset]['num_classes'], args)
    model.load_state_dict(torch.load(model_path + file))
    
    if os.path.exists(datasets[dataset]['models'] + f"{task}/cumu_mask.pkl"):
        trained_mask = pickle.load(open(datasets[dataset]['models'] + f"{task}/cumu_mask.pkl", 'rb'))
        # test_sparsity_mask(model, trained_mask)
    else:
        trained_mask = None
    model.eval()
    model.to(device)

    # Load data
    base_path = save_path = datasets[dataset]['tasks'] + task
    pipeline = CVTrainValTest(base_path=base_path, save_path=save_path)
    train_loader = pipeline.load_data_dronerc(256, offset=offset, args=args)

    eval_snr = True
    if eval_snr:
        folder = dataset_paths[i]
        accs, uncerts, results, aucs, covs = [], [], [], [], []
        n_rejected = 0
        rejected_top2_acc = [] #metrics['top2_accuracy_rejected']
        all_correct = []
        all_eu = []
        for snr in snr_range:
            # if snr >-10: continue
            filename = f"{snr}db.npz" if snr < 0 else f"{snr}db.npz"
            path = os.path.join(folder, filename)
            
            # data = np.load(path)
            # x = data['xte']
            # y = data['yte']
            print(datasets[dataset]['models'])
            save_file = f"snr{snr}_outputs_base_noisy.pkl"
            save_dir = datasets[dataset]['models'] + f"{task}/" + save_file
            train_loader = pipeline.load_data_dronerc(256, offset=offset, data = path, mixed_snrs=True, args=args)
            recollect_data = True
            if os.path.exists(save_file) and not recollect_data:
                all_labels, all_preds, unique_labels, eu, h_pred, f_eu  = load_test_outputs(save_file)
                all_labels = np.asarray(all_labels) - offset
                all_preds = np.asarray(all_preds) - offset
                correct = (np.array(all_preds) == np.array(all_labels))
            else:
                all_labels, all_preds, unique_labels, eu, h_pred, f_eu, y_proba = pipeline.test_model(
                args, model, trained_mask, eval_entropy=True,cm=True, enable_diagnostics=False, return_logits=True)
                # save_test_outputs(os.path.join(model_path, task), all_labels, all_preds, unique_labels, eu, h_pred, f_eu, filename=save_file)
            # all_correct.extend(correct)
            # all_eu.extend(h_pred)
            # continue
            # acc=100*accuracy(all_labels, all_preds)
            # plot_multiclass_roc_auc(all_labels, 
            #                         np.eye(len(unique_labels))[all_preds],
            #                         class_names=[str(lbl) for lbl in unique_labels],
            #                         title=f"ROC AUC - {dataset} - {task}"
            #                     )
            sweep_results = sweep_thresholds(all_labels, all_preds, u=list(eu))
            best = choose_tau(sweep_results, target_coverage=0.9)  # accept ~80% most-certain

            # # metrics = top2_acc_rejected(all_labels, eu, best['tau'], y_proba)

            accept_mask = eu <= best["tau"]

            all_labels_accepted = np.array(all_labels)[accept_mask]
            all_preds_accepted = np.array(all_preds)[accept_mask]
            eu_accepted = np.array(eu)[accept_mask]

            print("Chosen τ:", best["tau"])
            print("Coverage:", best["coverage"])
            print("Selective accuracy:", best["sel_acc"])
            print("Selective risk:", best["sel_risk"])
            print("Mean uncertainty: ", np.mean(eu_accepted))
            # # print("top2_accuracy_rejected", metrics['top2_accuracy_rejected'])
            
            auc = get_auc(all_labels_accepted, 
                                    np.eye(len(unique_labels))[all_preds_accepted],
                                    class_names=[str(lbl) for lbl in unique_labels])
            # # continue
            # plot_multiclass_roc_auc(all_labels_accepted, 
            #                         np.eye(len(unique_labels))[all_preds_accepted],
            #                         class_names=[str(lbl) for lbl in unique_labels],
            #                         title=f"ROC AUC - {dataset} - {task}"
            #                     )
            accs.append(best["sel_acc"]*100)
            aucs.append(auc)
            uncerts.append(eu_accepted)
            covs.append(best['coverage'])
            # if metrics['n_rejected'] != 0: rejected_top2_acc.append(metrics['top2_accuracy_rejected'])
            # n_rejected += metrics['n_rejected']
                # --- Evaluate ---

        results.append((accs, uncerts))
        print("========= Stats =========")
        print("Accs", np.mean(accs))
        print("Covs", np.mean(covs))
        # print("Top2 Acc", np.mean(rejected_top2_acc))
        # print("n_rejected", n_rejected)

        # for i, (accs, uncerts) in enumerate(results):
            # plt.plot(snr_range, accs, label=f"{labels[j]} Accuracy", color=colors[j])
            # plt.plot(snr_range, uncerts, linestyle='-.', label=f"{labels[j]} Uncertainty", color=colors[j])
        j+=1
        payload = {
            "results": to_serializable(results),
            "AUC": to_serializable(aucs)
            # "all_correct": to_serializable(all_correct),
            # "all_eu": to_serializable(all_eu)
        }
        # with open(f"eu_comparison/BLC/{task}_snr_results_selective_noisy.pkl", "wb") as handle:
        with open(f"eu_comparison/BLC/distrib/{task}_snr_results_test.pkl", "wb") as handle:
            pickle.dump(payload, handle)
    else:
    
        if args.arch != "bayes_rfnet":
            if os.path.exists(model_path + task + "/test_outputs.pkl"):
                all_labels, all_preds, unique_labels, eu, h_pred, f_eu = load_test_outputs(model_path + task + "/test_outputs.pkl")
            else:
                # Evaluate model
                all_labels, all_preds, unique_labels, omega, pred_class, util = pipeline.test_model(
                    args, model, trained_mask, cm=True, enable_diagnostics=False
                )
                correct = (np.array(all_preds) == np.array(all_labels))
                util_max, _ = torch.max(util, dim=1)
                inv_util = (1 - util_max) * 100
                save_test_outputs(os.path.join(model_path, task), all_labels, all_preds, unique_labels, omega, pred_class, util, inv_util)

        else:
            save_file = "/test_outputs_noisy.pkl"
            save_dir = datasets[dataset]['models'] + f"{task}" + save_file
            recollect_data = True
            if os.path.exists(save_dir) and not recollect_data:
                all_labels, all_preds, unique_labels, eu, h_pred, f_eu = load_test_outputs(save_dir)
            else:
                all_labels, all_preds, unique_labels, eu, h_pred, f_eu = pipeline.test_model(args, model, trained_mask, cm=True, eval_entropy=True)
                # save_test_outputs(os.path.join(model_path, task), all_labels, all_preds, unique_labels, eu, h_pred, f_eu, filename=save_dir)
                # continue
        alea = [total-eu for total, eu in zip(h_pred, eu)]
        results_epi = sweep_thresholds(all_labels, all_preds, u=eu)
        results_ale = sweep_thresholds(all_labels, all_preds, u=alea)
        results_tot = sweep_thresholds(all_labels, all_preds, u=h_pred)
        # best_epi = choose_tau(results_epi, target_coverage=0.75)
        # plot_acc_cov([results_epi, results_ale, results_tot],
        #              labels=["Epistemic", "Aleatoric", "Total"],
        #              mark_idx=[np.argmin(np.abs(results_epi["coverage"]-best_epi["coverage"])),
        #                        None, None])
        # plot_risk_cov([results_epi, results_ale, results_tot],
        #               labels=["Epistemic", "Aleatoric", "Total"])
        
        payload = {
            "results": to_serializable([results_epi, results_ale, results_tot]),
        }
        with open(f"eu_comparison/BLC/coverage/{task}_coverage_data_noisy.pkl", "wb") as handle:
            pickle.dump(payload, handle)

        # plot_multiclass_roc_auc(all_labels, 
        #                             np.eye(len(unique_labels))[all_preds],
        #                             class_names=[str(lbl) for lbl in unique_labels],
        #                             title=f"ROC AUC - {dataset} - {task}"
        #                         )
        # results = sweep_thresholds(all_labels, all_preds, u=f_eu)
        # best = choose_tau(results, target_coverage=0.95)  # accept ~80% most-certain

        # accept_mask = f_eu <= best["tau"]

        # all_labels_accepted = np.array(all_labels)[accept_mask]
        # all_preds_accepted = np.array(all_preds)[accept_mask]
        # eu_accepted = np.array(f_eu)[accept_mask]

        # print("Chosen τ:", best["tau"])
        # print("Coverage:", best["coverage"])
        # print("Selective accuracy:", best["sel_acc"])
        # print("Selective risk:", best["sel_risk"])

        # plot_multiclass_roc_auc(all_labels_accepted, 
        #                         np.eye(len(unique_labels))[all_preds_accepted],
        #                         class_names=[str(lbl) for lbl in unique_labels],
        #                         title=f"ROC AUC - {dataset} - {task}")
    continue
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels, normalize="true") * 100
    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=np.array(unique_labels),
                yticklabels=np.array(unique_labels),
                cbar=False, ax=axs[i, 0])
    axs[i, 0].set_xlabel("Predicted Label")
    axs[i, 0].set_ylabel("True Label")
    if i == 0:
        axs[i, 0].set_title(f"Confusion Matrices ({dataset})")

    # Uncertainty boxplot
    df = pd.DataFrame({
        "Predicted Label": all_preds,
        "Uncertainty": inv_util if args.arch != "bayes_rfnet" else np.array(eu),
        "Correctness": np.where(np.array(all_preds) == np.array(all_labels), "Correct", "Incorrect")
    })
    hue_order = ["Correct", "Incorrect"]

    sns.stripplot(x="Predicted Label", y="Uncertainty", hue="Correctness", hue_order=hue_order, data=df,
                  color='black', alpha=0.1, size=2, dodge=True, ax=axs[i, 1], legend="")
    axs[i, 1].grid(True)
    axs[i, 1].set_ylabel("Uncertainty (%)")
    if i == 0: axs[i, 1].set_title(f"Uncertainty per Prediction ({dataset})")
    if i == 0:
        sns.boxplot(x="Predicted Label", y="Uncertainty", hue="Correctness", hue_order=hue_order, data=df,
                whis=1.5, width=0.6, palette="Blues", fliersize=0, ax=axs[i, 1])
        plt.legend(loc="upper left", title=None, prop={'size': 4})
    else:
        sns.boxplot(x="Predicted Label", y="Uncertainty", hue="Correctness", hue_order=hue_order, data=df,
                whis=1.5, width=0.6, palette="Blues", fliersize=0, ax=axs[i, 1], legend="")
    # plt.legend(title=None)
    
    # Clean up duplicate legends from stripplot
    handles, labels = axs[i, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs[i, 1].legend(by_label.values(), by_label.keys(), title="", prop={'size': 8}, loc="upper right") # loc="upper right"
    
    axs[i, 0].tick_params(axis='x', labelsize=8)  # x-ticks
    axs[i, 0].tick_params(axis='y', labelsize=8)  # y-ticks
    axs[i, 1].tick_params(axis='x', labelsize=8)  # x-ticks
    axs[i, 1].tick_params(axis='y', labelsize=8)  # y-ticks
    # Find the highest top whisker line

    whisker_tops = [
        line.get_ydata()[1] for line in axs[i, 1].lines
        if len(line.get_ydata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]  # vertical line
    ]

    # Set dynamic y-limit slightly above the max whisker
    if whisker_tops:
        max_whisker = max(whisker_tops)
        axs[i, 1].set_ylim(-2, max_whisker * 1.1)  # add 5% headroom


# fig.tight_layout()
fig.subplots_adjust(
    left=0.352,   # space from the left edge
    bottom=0.09, # space from the bottom edge
    right=0.9,  # space from the right edge
    top=0.938,    # space from the top edge
    wspace=0.27,  # width spacing between columns
    hspace=0.474   # height spacing between rows
)

# plt.rcParams.update({
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'axes.labelsize': 12,
#     'axes.titlesize': 14
# })
os.makedirs(model_path + "figs/", exist_ok=True)
if save_figs:
    fig.savefig(datasets[dataset]['models'] + f"figs/cm-pred_uncertainty.png", bbox_inches='tight')
plt.show()
