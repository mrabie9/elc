import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # optional, for prettier heatmaps
import torch
from models.masknet import ResNet50_1d, ResNet18_1d
from TrainValTest import CVTrainValTest
import pickle


file = "rfnet16"
# Assuming this is your model class
model = ResNet18_1d(slice_size=1024, num_classes=17)
model.load_state_dict(torch.load(f"dronerc/exp-drc-1024sl-3-p20-3t-batchnorm-fixedloss-1/task0/{file}.pt"))
trained_mask = None #pickle.load(open("dronerc/exp-drc-1024sl-3-p20-3t-batchnorm-fixedlabels-fullepochs/task1/cumu_mask.pkl",'rb'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


base_path = save_path = "dronerc/tasks-1024sl/task1"#radar_dataset.npz"
pipeline = CVTrainValTest(base_path=base_path, save_path=save_path)
train_loader = pipeline.load_data_dronerc(256, offset=5)
test_loader = pipeline.test_loader()

all_labels, all_preds, unique_labels, omega, pred_class, util = pipeline.test_model(None, model, trained_mask, cm=True, enable_diagnostics=False)

all_omegas = torch.cat(omega)   # shape [N]
util_max, _ = torch.max(util, dim=1)
inv_util = 1- util_max
print(inv_util.shape)

df = pd.DataFrame({
    "Predicted Class": all_preds,
    "Uncertainty": inv_util
})

plt.figure(figsize=(8, 6))
sns.boxplot(x="Predicted Class", y="Uncertainty", data=df, whis=[5, 95], width=0.6, palette="Blues", fliersize=0)
sns.stripplot(x="Predicted Class", y="Uncertainty", data=df, color='black', alpha=0.3, size=3)

plt.title("Drone RC - Uncertainty per Prediction")
plt.ylabel("Uncertainty (Omega)")
plt.grid(True)
plt.show()