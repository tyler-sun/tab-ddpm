import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path


def get_info(ds_path, labels_column, output_dir):

    parent_path = Path(ds_path).parent
    name = Path(parent_path).name
    id = name.lower().replace(" ", "_") + "--id"
    
    train_npy = Path(parent_path) / "X_num_train.npy"
    val_npy = Path(parent_path) / "X_num_val.npy"
    test_npy = Path(parent_path) / "X_num_test.npy"
    sizes = []

    for npy in [train_npy, val_npy, test_npy]:
        arr = np.load(npy)
        if arr is None:
            print(f"Failed to find numpy file from {ds_path}")
            sizes.append(None)
        else:
            sizes.append(arr.shape[0])
    
    df = pd.read_csv(ds_path)
    if df is None:
        print(f"Failed to find CSV file from {ds_path}")
        return
    num_classes = df[labels_column].nunique()
    if num_classes != 2:
        task_type = "regression"
    else:
        task_type = "binclass"
    df.drop(columns=[labels_column], inplace=True)

    n_num_features = df.select_dtypes(include=['float64', 'int64']).shape[1]
    n_cat_features = df.select_dtypes(include=['object', 'category']).shape[1]

    info = {
        "name": name,
        "id": id,
        "task_type": task_type,
        "n_num_features": n_num_features,
        "n_cat_features": n_cat_features,
        "train_size": sizes[0],
        "val_size": sizes[1],
        "test_size": sizes[2]
    }
    with open(Path(output_dir)/"info.json", "w") as f:
        json.dump(info, f, indent=2)
        print(f"Saved info to {Path(output_dir)/'info.json'}")


def main():
    # need CSV and NPY files in same directory to get info
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_path', type=str, default="data/credit-card/creditcard.csv")
    parser.add_argument('labels_column', type=str, default='Class')
    parser.add_argument('output_dir', type=str, default="exp/credit-card/ddpm_cb_best")

    args = parser.parse_args()
    get_info(
        ds_path=args.ds_path,
        labels_column=args.labels_column,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()