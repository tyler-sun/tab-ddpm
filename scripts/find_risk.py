import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from catboost import CatBoostClassifier
from pathlib import Path
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, f1_score


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


def select_best_threshold(y_true, risk_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, risk_scores)
    if precisions.size == 0 or recalls.size == 0 or thresholds.size == 0:
        return 0.5
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    if len(f1_scores) != len(thresholds):
        f1_scores = f1_scores[:-1]
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx])


def find_risk(ds_path, output_dir=None, placeholder=False, model='xgb'):

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
    elif model.lower() == 'cb' or model.lower() == 'catboost':
        splits = ['train', 'val', 'test']
        data_splits = []

        for split in splits:
            num_path = ds_path / f'X_num_{split}.npy'
            cat_path = ds_path / f'X_cat_{split}.npy'

            df_num = pd.DataFrame()
            df_cat = pd.DataFrame()

            if num_path.exists():
                x_num = np.load(num_path, allow_pickle=True).astype(np.float32)
                df_num = pd.DataFrame(x_num)
                print("Numerical data:", split, df_num.shape)

            if cat_path.exists():
                x_cat = np.load(cat_path, allow_pickle=True)
                # Ensure categorical features are strings or integer categories
                df_cat = pd.DataFrame(x_cat).astype(str)
                # Avoid overlapping column names with numerical columns
                df_cat.columns = [c + df_num.shape[1] for c in df_cat.columns]
                print("Categorical data:", split, df_cat.shape)

            if df_num.empty and df_cat.empty:
                print(f"Failed to find data for {split} split")
                return

            # Combine numerical (floats) and categorical (strings) cleanly
            combined_df = pd.concat([df_num, df_cat], axis=1)
            data_splits.append(combined_df)

            if split == 'train':
                x_train_data = data_splits[-1]

        print("CatBoost on", x_train_data.shape, y_train_data.shape)
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.01, 
            depth=6,
            loss_function='Logloss',
            # auto_class_weights='Balanced',
            verbose=False)

        #cat_features = list(range(x_num_data.shape[1], x_num_data.shape[1] + x_cat_data.shape[1]))
        cat_features = list(df_cat.columns)
        model.fit(x_train_data, y_train_data, cat_features=cat_features)

        total_r_data = 0
        best_threshold = 0.5
        for split, data_split, y_split, path in zip(splits, data_splits, y_splits, data_paths):
            r_data = model.predict_proba(data_split)[:, 1]
            np.save(path, r_data)
            total_r_data += len(r_data)
            risk0 = r_data[y_split == 0]
            risk1 = r_data[y_split == 1]
            print(f"Mean of risks in {split}, y = 0: {risk0.mean():.2f}, y = 1: {risk1.mean():.2f}")
            print(f"Min/max of risks in {split}, y = 0: {risk0.min()}, {risk0.max()} y = 1: {risk1.min()}, {risk1.max()}")

            if split == 'val':
                best_threshold = select_best_threshold(y_val_data, r_data)
                print(f"Optimal validation threshold (Max F1): {best_threshold:.4f}")
                val_predictions = (r_data >= best_threshold).astype(int)
                val_f1 = f1_score(y_val_data, val_predictions)
                print(f"Validation F1 at selected threshold: {val_f1:.4f}")

        test_risk = np.load(r_test_path, allow_pickle=True)
        test_predictions = (test_risk >= best_threshold).astype(int)
        test_f1 = f1_score(y_test_data, test_predictions)
        print(f"Test F1 using validation-selected threshold: {test_f1:.4f}")

        print("CatBoost on", x_train_data.shape, y_train_data.shape)
    # default to logistic regression to calculate risk values
    else:
        splits = ['train', 'val', 'test']
        data_splits = []
        scaler = StandardScaler()
        try:
            encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True)
        except TypeError:
            encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=True)

        for split in splits:
            data_split = []

            if Path(ds_path / f'X_num_{split}.npy').exists():
                x_num_data = np.load(ds_path / f'X_num_{split}.npy', allow_pickle=True)
                if split == 'train':
                    x_num_data = scaler.fit_transform(x_num_data)
                else:
                    x_num_data = scaler.transform(x_num_data)
                x_num_data = sparse.csr_matrix(x_num_data)
                data_split.append(x_num_data)
                print("Numerical data:", split, type(x_num_data), data_split[-1].shape)
            if Path(ds_path / f'X_cat_{split}.npy').exists():
                x_cat_data = np.load(ds_path / f'X_cat_{split}.npy', allow_pickle=True)
                if split == 'train':
                    x_cat_data = encoder.fit_transform(x_cat_data)
                else:
                    x_cat_data = encoder.transform(x_cat_data)
                data_split.append(x_cat_data)
                print("Categorical data:", split, type(x_cat_data), data_split[-1].shape)
            if len(data_split) == 0:
                print(f"Failed to find either num or cat data for {split} split from {ds_path}")
                return

            data_splits.append(sparse.hstack(data_split, format='csr'))

            if split == 'train':
                x_train_data = data_splits[-1]

        print("Logistic regression on", x_train_data.shape, y_train_data.shape)
        model = LogisticRegression(max_iter=1000)
        model.fit(x_train_data, y_train_data)
        total_r_data = 0
        best_threshold = 0.5
        for split, data_split, y_split, path in zip(splits, data_splits, y_splits, data_paths):
            r_data = model.predict_proba(data_split)[:, 1]
            np.save(path, r_data)
            total_r_data += len(r_data)
            risk0 = r_data[y_split == 0]
            risk1 = r_data[y_split == 1]
            print(f"Mean of risks in {split}, y = 0: {risk0.mean()}, y = 1: {risk1.mean()}")
            print(f"Min/max of risks in {split}, y = 0: {risk0.min()}, {risk0.max()} y = 1: {risk1.min()}, {risk1.max()}")

            if split == 'val':
                best_threshold = select_best_threshold(y_val_data, r_data)
                print(f"Optimal validation threshold (Max F1): {best_threshold:.4f}")
                val_predictions = (r_data >= best_threshold).astype(int)
                val_f1 = f1_score(y_val_data, val_predictions)
                print(f"Validation F1 at selected threshold: {val_f1:.4f}")

        test_risk = np.load(r_test_path, allow_pickle=True)
        test_predictions = (test_risk >= best_threshold).astype(int)
        test_f1 = f1_score(y_test_data, test_predictions)
        print(f"Test F1 using validation-selected threshold: {test_f1:.4f}")

    return total_r_data

def main():
    # credit card dataset was split into 65/15/20 for train/val/test
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--placeholder', action='store_true', default=False, help="Generate placeholder risk values rather than calculating")
    parser.add_argument('--plot', action='store_true', default=False, help='Plot risk values as histogram')
    parser.add_argument('--plot_only', type=str, default=None, help="Plot risk values from file path (skips risk calculation)")
    parser.add_argument('--merged', action='store_true', default=False, help="Merge real data with synthetic data in exp, use with --plot_only")
    parser.add_argument('--model', type=str, default='logistic regression', help='cb or logistic regression')

    args = parser.parse_args()
    if not args.plot_only:
        num_added = find_risk(
            ds_path = Path(args.ds_path),
            output_dir = args.output_dir,
            placeholder = args.placeholder,
            model = args.model
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

            plt.hist(r_train_data, alpha=0.5, color='tab:blue', label='Train')
            plt.hist(r_val_data, alpha=0.5, color='tab:orange', label='Validation')
            plt.hist(r_test_data, alpha=0.5, color='tab:green', label='Test')
            plt.xlabel('Risk')
            plt.ylabel('Frequency')
            plt.title(f'{Path(args.ds_path).name} Dataset Risk Distribution')
            plt.legend()
            plot_path = Path(args.ds_path)/'risk_plot.png'
            plt.savefig(plot_path)
            print(f"Saved risk plot to {plot_path}")
    else:
        risk_data_path = Path(args.plot_only)
        risk_data = np.load(risk_data_path, allow_pickle=True)
        if risk_data is None:
            print(f"Failed to load data from provided path: {risk_data_path}")
            return
        plt.hist(risk_data, alpha=0.5, label='Synthetic')

        if args.merged:
            real_data_path = Path(args.ds_path) / 'r_train.npy'
            real_data = np.load(real_data_path, allow_pickle=True)
            if real_data is None:
                print(f"Failed to load real data from dataset path: {real_data_path}")
                return
            plt.hist(real_data, alpha=0.5, label='Real (Training)')
            plt.title(f'Merged {Path(args.ds_path).name} data: Risk Distribution')
        else:
            plt.title(f'Synthetic {Path(args.ds_path).name} data: Risk Distribution')
        
        plt.xlabel('Risk')
        plt.ylabel('Frequency')
        plt.legend()
        plot_path = risk_data_path.parent / 'risk_plot.png'
        plt.savefig(plot_path)
        print(f"Saved risk plot to {plot_path}")



if __name__ == '__main__':
    main()