import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


def get_risk_paths(ds_path, output_dir=None):
    if output_dir:
        output_dir = Path(output_dir)
        r_train_path = output_dir / 'r_train.npy'
        r_val_path = output_dir / 'r_val.npy'
        r_test_path = output_dir / 'r_test.npy'
    else:
        r_train_path = ds_path / 'r_train.npy'
        r_val_path = ds_path / 'r_val.npy'
        r_test_path = ds_path / 'r_test.npy'
    return r_train_path, r_val_path, r_test_path


def find_risk(ds_path, output_dir=None, placeholder=False):

    r_train_path, r_val_path, r_test_path = get_risk_paths(ds_path, output_dir)
    data_paths = [r_train_path, r_val_path, r_test_path]

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
    y_splits = [y_train_data, y_val_data, y_test_data]

    # generate random values between 0 and 1 as placeholder risk values
    if placeholder:
        r_train_data = np.random.rand(len(y_train_data))
        r_val_data = np.random.rand(len(y_val_data))
        r_test_data = np.random.rand(len(y_test_data))

        np.save(r_train_path, r_train_data)
        np.save(r_val_path, r_val_data)
        np.save(r_test_path, r_test_data)

        return len(r_train_data) + len(r_val_data) + len(r_test_data)
    # use logistic regression to calculate risk values
    else:
        splits = ['train', 'val', 'test']
        data_splits = []
        scaler = StandardScaler()
        encoder = OneHotEncoder(drop="first", handle_unknown="ignore")

        # if Path(ds_path / f'X_num_train.npy').exists():
        #     x_num_data = np.load(ds_path / f'X_num_train.npy', allow_pickle=True)
        #     data_split.append(x_num_data)
        # if Path(ds_path / f'X_cat_train.npy').exists():
        #     x_cat_data = np.load(ds_path / f'X_cat_train.npy', allow_pickle=True)
        #     data_split.append(x_cat_data)
        # if len(data_split) == 0:
        #     print(f"Failed to find either num or cat training data from {ds_path}")
        #     return
        # scaler.fit_transform(x_num_data)
        # encoder.fit_transform(x_cat_data)
        # x_train = np.hstack(data_split)

        # if Path(ds_path / f'X_num_val.npy').exists():
        #     x_num_data = np.load(ds_path / f'X_num_val.npy', allow_pickle=True)
        #     data_split.append(x_num_data)
        # if Path(ds_path / f'X_cat_val.npy').exists():
        #     x_cat_data = np.load(ds_path / f'X_cat_val.npy', allow_pickle=True)
        #     data_split.append(x_cat_data)
        # if len(data_split) == 0:
        #     print(f"Failed to find either num or cat validation data from {ds_path}")
        #     return
        # x_val = np.hstack(data_split)
        # scaler.transform(x_val[:, :x_num_data.shape[1]])

        # if Path(ds_path / f'X_num_test.npy').exists():
        #     x_num_data = np.load(ds_path / f'X_num_test.npy', allow_pickle=True)
        #     data_split.append(x_num_data)
        # if Path(ds_path / f'X_cat_test.npy').exists():
        #     x_cat_data = np.load(ds_path / f'X_cat_test.npy', allow_pickle=True)
        #     data_split.append(x_cat_data)
        # if len(data_split) == 0:
        #     print(f"Failed to find either num or cat test data from {ds_path}")
        #     return
        # x_test = np.hstack(data_split)
        # scaler.transform(x_test[:, :x_num_data.shape[1]])

        for split in splits:
            data_split = []
            
            if Path(ds_path / f'X_num_{split}.npy').exists():
                x_num_data = np.load(ds_path / f'X_num_{split}.npy', allow_pickle=True)
                if split == 'train':
                    x_num_data = scaler.fit_transform(x_num_data)
                else:
                    x_num_data = scaler.transform(x_num_data)
                data_split.append(x_num_data)
                print(type(x_num_data), data_split[-1].shape)
            if Path(ds_path / f'X_cat_{split}.npy').exists():
                x_cat_data = np.load(ds_path / f'X_cat_{split}.npy', allow_pickle=True)
                if split == 'train':
                    x_cat_data = encoder.fit_transform(x_cat_data).toarray()
                else:
                    x_cat_data = encoder.transform(x_cat_data).toarray()
                data_split.append(x_cat_data)
                print(type(x_cat_data), data_split[-1].shape)
            if len(data_split) == 0:
                print(f"Failed to find either num or cat data for {split} split from {ds_path}")
                return
            
            data_splits.append(np.concatenate(data_split, axis=1))

            if split == 'train':
                x_train_data = data_splits[-1]
        
        print("Logistic regression on", x_train_data.shape, y_train_data.shape)
        model = LogisticRegression()
        model.fit(x_train_data, y_train_data)
        total_r_data = 0
        for split, data_split, y_split, path in zip(splits, data_splits, y_splits, data_paths):
            r_data = model.predict_proba(data_split)[:, 1]
            np.save(path, r_data)
            total_r_data += len(r_data)
            risk0 = r_data[y_split == 0]
            risk1 = r_data[y_split == 1]
            print(f"Mean of risks in {split}, y = 0: {risk0.mean()}, y = 1: {risk1.mean()}")

        return total_r_data

def main():
    # credit card dataset was split into 65/15/20 for train/val/test
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--placeholder', action='store_true', default=False, help="Generate placeholder risk values rather than calculating")
    parser.add_argument('--plot', action='store_true', default=False, help='Plot risk values as histogram')

    args = parser.parse_args()
    num_added = find_risk(
        ds_path = Path(args.ds_path),
        output_dir = args.output_dir,
        placeholder = args.placeholder
    )
    if args.output_dir:
        print(f"Added {num_added} risk values to {args.output_dir}")
    else:
        print(f"Added {num_added} risk values to {args.ds_path}")

    if args.plot:
        r_train_path, r_val_path, r_test_path = get_risk_paths(Path(args.ds_path), args.output_dir)
        r_train_data = np.load(r_train_path, allow_pickle=True)
        r_val_data = np.load(r_val_path, allow_pickle=True)
        r_test_data = np.load(r_test_path, allow_pickle=True)

        plt.hist(r_train_data, alpha=0.5, label='Train')
        plt.hist(r_val_data, alpha=0.5, label='Validation')
        plt.hist(r_test_data, alpha=0.5, label='Test')
        plt.xlabel('Risk')
        plt.ylabel('Frequency')
        plt.title(f'{Path(args.ds_path).name} Dataset Risk Distribution')
        plt.legend()
        plot_path = Path(args.ds_path)/'risk_plot.png'
        plt.savefig(plot_path)
        print(f"Saved risk plot to {plot_path}")


if __name__ == '__main__':
    main()