import torch
import os
import numpy as np

def filter_1_class(
    parent_dir,
    class_filter = 1,
    limit_samples = None
):
    synthetic_data_path = os.path.join(parent_dir, 'y_train.npy')
    labels = np.load(synthetic_data_path, allow_pickle=True)
    indices = np.where(labels == class_filter)[0]
    if len(indices) == 0:
        print(f"No samples of class {class_filter} found.")
        return
    if limit_samples is not None and len(indices) > limit_samples:
        indices = indices[:limit_samples]

    cat_path = os.path.join(parent_dir, 'X_cat_train.npy')
    if os.path.exists(cat_path):
        X_cat = np.load(cat_path, allow_pickle=True)
        X_cat = X_cat[indices]
        if X_cat.shape[1] > 0:
            np.save(cat_path, X_cat)
        else:
            print("No categorical features found")
    num_path = os.path.join(parent_dir, 'X_num_train.npy')
    if os.path.exists(num_path):
        X_num = np.load(num_path, allow_pickle=True)
        X_num = X_num[indices] 
        if X_num.shape[1] > 0:
            np.save(num_path, X_num)
        else:
            print("No numerical features found")

    labels = labels[indices]
    np.save(synthetic_data_path, labels)
    print(f"Filtered to {len(labels)} samples of class {class_filter}")

    return labels