import torch
import os
import argparse
import numpy as np

def limit(
    parent_dir,
    limit_samples
):
    data_path = os.path.join(parent_dir, f'y_train.npy')
    labels = np.load(data_path, allow_pickle=True)
    if len(labels) > limit_samples:
        labels = labels[:limit_samples]
        print(f'Saving {limit_samples} data samples to {data_path}')
        np.save(data_path, labels)

        cat_path = os.path.join(parent_dir, f'X_cat_train.npy')
        if os.path.exists(cat_path):
            X_cat = np.load(cat_path, allow_pickle=True)
            X_cat = X_cat[:limit_samples]
            if X_cat.shape[1] > 0:
                np.save(cat_path, X_cat)
                print(f'Saving {limit_samples} samples of categorical features to {cat_path}')
            else:
                print("No categorical features found")

        num_path = os.path.join(parent_dir, f'X_num_train.npy')
        if os.path.exists(num_path):
            X_num = np.load(num_path, allow_pickle=True)
            X_num = X_num[:limit_samples]
            if X_num.shape[1] > 0:
                np.save(num_path, X_num)
                print(f'Saving {limit_samples} samples of numerical features to {num_path}')
            else:
                print("No numerical features found")
    else:
        print("Limit exceeds number of samples found")
    
    return limit_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('limit', type=int, default=None)

    args = parser.parse_args()

    limit(args.parent_dir, args.limit)


if __name__ == '__main__':
    main()