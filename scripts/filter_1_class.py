import torch
import os
import argparse
import numpy as np

def filter_1_class(
    parent_dir,
    class_filter = 1,
    limit_samples = None,
    target_dir = None,
    val_split = False,
    test_split = False
):
    data_splits = ['train']
    if val_split:
        data_splits.append('val')
    if test_split:
        data_splits.append('test')

    for split in data_splits:
        data_path = os.path.join(parent_dir, f'y_{split}.npy')
        labels = np.load(data_path, allow_pickle=True)
        indices = np.where(labels == class_filter)[0]
        if len(indices) == 0:
            print(f"No samples of class {class_filter} found.")
            return
        if limit_samples is not None and len(indices) > limit_samples:
            indices = indices[:limit_samples]

        cat_path = os.path.join(parent_dir, f'X_cat_{split}.npy')
        if os.path.exists(cat_path):
            X_cat = np.load(cat_path, allow_pickle=True)
            X_cat = X_cat[indices]
            if X_cat.shape[1] > 0:
                if target_dir:
                    np.save(os.path.join(target_dir, f'X_cat_{split}.npy'), X_cat)
                else:
                    np.save(cat_path, X_cat)
            else:
                print("No categorical features found")

        num_path = os.path.join(parent_dir, f'X_num_{split}.npy')
        if os.path.exists(num_path):
            X_num = np.load(num_path, allow_pickle=True)
            X_num = X_num[indices] 
            if X_num.shape[1] > 0:
                if target_dir:
                    np.save(os.path.join(target_dir, f'X_num_{split}.npy'), X_num)
                else:
                    np.save(num_path, X_num)
            else:
                print("No numerical features found")

        labels = labels[indices]
        if target_dir:
            print("Saving data to", target_dir)
            np.save(os.path.join(target_dir, f'y_{split}.npy'), labels)
        else:
            print("Saving data to", data_path)
            np.save(data_path, labels)
        print(f"Filtered to {len(labels)} samples of class {class_filter} in {split} split")

    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('class_filter', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--target_dir', type=str, default=None)
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    filter_1_class(args.parent_dir, args.class_filter, args.limit, args.target_dir, args.val, args.test)


if __name__ == '__main__':
    main()