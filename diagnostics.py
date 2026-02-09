# Dempster-Shafer Layer Diagnostics: Class-specific uncertainty and prototype behavior
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_per_class_belief(model, loader, num_classes):
    """
    Visualize per-class belief and omega.
    """
    model.eval()
    class_beliefs = [[] for _ in range(num_classes)]
    class_omegas = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.cuda().float()
            targets = targets.cuda().long()
            output = model(inputs)

            beliefs = output[:, :-1]
            omega = output[:, -1]

            for i in range(len(targets)):
                label = targets[i].item()
                class_beliefs[label].append(beliefs[i].cpu().numpy())
                class_omegas[label].append(omega[i].item())

    # Plot omega per class
    avg_omegas = [np.mean(omegas) if omegas else 0 for omegas in class_omegas]
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(range(num_classes)), y=avg_omegas)
    plt.title("Average Uncertainty (Omega) per Class")
    plt.xlabel("Class")
    plt.ylabel("Omega")
    plt.show()

    return class_beliefs, class_omegas


def visualize_prototype_distances(ds_layer, features, labels, class_names=None):
    """
    Visualize the distances from input samples to each prototype.
    """
    distances = ds_layer[1].ds1(features.cuda())  # shape: [batch_size, n_prototypes]
    distances = distances.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    print("Visualise Protoype Distances")
    plt.figure(figsize=(12, 6))
    sns.heatmap(distances[:50], cmap="viridis")
    plt.title("Prototype Distances for First 50 Samples")
    plt.xlabel("Prototype Index")
    plt.ylabel("Sample Index")
    plt.show()

    # Optionally per-class
    if class_names is not None:
        for class_idx in range(len(class_names)):
            class_mask = labels == class_idx
            if np.sum(class_mask) == 0:
                continue
            class_dist = distances[class_mask]
            avg_dist = np.mean(class_dist, axis=0)
            plt.plot(avg_dist, label=class_names[class_idx])
        plt.legend()
        plt.title("Average Prototype Distances per Class")
        plt.xlabel("Prototype Index")
        plt.ylabel("Avg Distance")
        plt.show()


def inspect_beta_matrix(ds_layer):
    """
    Print and optionally plot the beta matrix from Belief layer (DS2).
    """
    beta = ds_layer[1].ds2.beta.detach().cpu().numpy()
    beta_sq = np.square(beta)
    beta_sum = np.sum(beta_sq, axis=0, keepdims=True)
    u = beta_sq / (beta_sum + 1e-8)  # Normalized belief strength

    plt.figure(figsize=(10, 6))
    sns.heatmap(u, cmap="mako", cbar=True)
    plt.title("Belief Assignment Matrix (u)")
    plt.xlabel("Prototype")
    plt.ylabel("Class")
    plt.show()
    return u


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for prettier heatmaps
import torch
from models.masknet import ResNet50_1d, ResNet18_1d
from TrainValTest import CVTrainValTest


# Assuming this is your model class
model = ResNet18_1d(slice_size=1024, num_classes=10)
model.load_state_dict(torch.load("rfmls/exp-lora-1-test/task0/rfnet16.pt"))
trained_mask = None #pickle.load(open("usrp/exp-usrp-BD1E-576sl-ResNet50-1d/task0/cumu_mask.pkl",'rb'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

base_path = save_path = "rfmls/tasks_lp_downsampled/task0"#radar_dataset.npz"
pipeline = CVTrainValTest(base_path=base_path, save_path=save_path)
train_loader, _ = pipeline.load_data_dronerc(256, offset=0)
# test_loader = pipeline.test_loader()

ds_layer = model.ds_module

all_labels, all_preds, unique_labels, omega, pred_class, _ = pipeline.test_model(None, model, trained_mask, cm=False, ds_layer=ds_layer, enable_diagnostics=True)

