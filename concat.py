import os
import numpy as np
from glob import glob

def split_features_into_samples(data, labels, sample_size=2048):
        labels = labels # Extract labels
        features = data # Extract feature columns
        
        num_features = features.shape[1]
        num_splits = num_features // sample_size  # Ensure it's divisible
        
        # print(features[:6,:6])
        # print("features", features[:6])

        assert num_features % sample_size == 0, "Number of features must be divisible by sample size" + str(num_features) + str(sample_size)
        
        # Reshape features into smaller chunks
        reshaped_features = features.reshape(-1, sample_size)  # Reshape while keeping all rows
        
        # Repeat labels to match the new rows
        repeated_labels = np.repeat(labels, num_splits).reshape(-1, 1)
        repeated_labels = np.hstack(repeated_labels)
        
        return repeated_labels, reshaped_features

def load_normalised_data(filepaths, sample_size=1024):
    X_all, y_all = [], []
    for path in filepaths:
        data = np.load(path)
        X, y = data['X'], data['y']
        
        # If too long, split into 1024-length chunks
        if X.shape[1] > sample_size:
            y, X = split_features_into_samples(X, y, sample_size)
        elif X.shape[1] < sample_size:
            print(f"Skipping {path} — feature length {X.shape[1]} not divisible by {sample_size}")
            continue
        
        # Final shape check
        assert X.shape[1] == sample_size, f"{path} has shape {X.shape}"

        # Normalise per-task
        X = (X - np.mean(X)) / (np.std(X) + 1e-8)

        X_all.append(X)
        y_all.append(y)
    
    return np.vstack(X_all), np.hstack(y_all)
# Find all train/test files under tasks/
train_files = sorted(glob('dronerc/tasks-1024sl/task*/train.npz'))
test_files = sorted(glob('dronerc/tasks-1024sl/task*/test.npz'))

# Process and normalise per task
X_train, y_train = load_normalised_data(train_files)
X_test, y_test = load_normalised_data(test_files)

# Save combined datasets
np.savez('dronerc/tasks-sm/task0/train.npz', X=X_train, y=y_train)
np.savez('dronerc/tasks-sm/task0/test.npz', X=X_test, y=y_test)
print("Saved 'train.npz' and 'test.npz'")