import numpy as np
import pandas as pd
import argparse
from pathlib import Path

def add_risk(ds_path, output_dir=None, placeholder=False):

    if output_dir:
        r_train_path = output_dir / 'r_train.npy'
        r_val_path = output_dir / 'r_val.npy'
        r_test_path = output_dir / 'r_test.npy'
    else:
        r_train_path = ds_path / 'r_train.npy'
        r_val_path = ds_path / 'r_val.npy'
        r_test_path = ds_path / 'r_test.npy'
    data_splits = ['train', 'val', 'test']

    y_train = ds_path / 'y_train.npy'
    y_train_data = np.load(y_train, allow_pickle=True)
    if y_train_data is None:
        print(f"Failed to find y_train.npy from {y_train}")
        return
    y_val = ds_path / 'y_val.npy'
    y_val_data = np.load(y_val, allow_pickle=True)
    if y_val_data is None:
        print(f"Failed to find y_val.npy from {y_val}")
        return
    y_test = ds_path / 'y_test.npy'
    y_test_data = np.load(y_test, allow_pickle=True)
    if y_test_data is None:
        print(f"Failed to find y_test.npy from {y_test}")
        return

    if placeholder:
        r_train_data = np.random.rand(len(y_train_data))
        r_val_data = np.random.rand(len(y_val_data))
        r_test_data = np.random.rand(len(y_test_data))

        np.save(r_train_path, r_train_data)
        np.save(r_val_path, r_val_data)
        np.save(r_test_path, r_test_data)

        return len(r_train_data) + len(r_val_data) + len(r_test_data)
    else:

        return 0

def main():
    # credit card dataset was split into 65/15/20 for train/val/test
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_path', type=str, default="data/credit-card")
    parser.add_argument('--output_dir', type=str, default="data/credit-card")
    parser.add_argument('--placeholder', action='store_true', default=False, help="Generate placeholder risk values rather than calculating")

    args = parser.parse_args()
    num_added =add_risk(
        ds_path = Path(args.ds_path),
        output_dir = Path(args.output_dir),
        placeholder = args.placeholder
    )
    print(f"Added {num_added} risk values to {args.output_dir}")


if __name__ == '__main__':
    main()