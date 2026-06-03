import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def convert_to_df(ds_name="adult", sampling_method="ddpm", eval_type="synthetic", model_type="catboost", normalized=True, file_path=None):
    
    if file_path is not None:
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded CSV from {file_path}")
        except Exception as e:
            print(f"Failed to load CSV from {file_path}: {e}")
            return None
    else:
        # locate files for data
        if eval_type == 'real':
            parent_path = Path(f'data/{ds_name}/')
            X_cat_file = parent_path / 'X_cat_train.npy'
            X_num_file = parent_path / 'X_num_train.npy'
            y_file = parent_path / f'y_train.npy'
        else:
            parent_path = Path(f'exp/{ds_name}/')
            if normalized:
                file_details = "_train"
            else:
                file_details = "_unnorm"

            if sampling_method == "ddpm":
                if model_type == "catboost":
                    model_folder = "ddpm_cb_best"
                elif model_type == "xgboost":
                    model_folder = "ddpm_xg_best"
                else:
                    model_folder = "ddpm_mlp_best"
            elif sampling_method in ["ctabgan", "ctabgan-plus", "tvae", "smote"]:
                model_folder = sampling_method

            X_cat_file = parent_path / model_folder / f'X_cat{file_details}.npy'
            X_num_file = parent_path / model_folder / f'X_num{file_details}.npy'
            y_file = parent_path / model_folder / f'y{file_details}.npy'
        
        print(f"Loading data from: {X_cat_file}, {X_num_file}, {y_file}")
        cat_data, num_data, y_data = None, None, None
        df_array = []
        if X_cat_file.exists():
            cat_data = np.load(X_cat_file, allow_pickle=True)
            df_array.append(cat_data)
        else:
            print("Unable to find categorical data file.")
        if X_num_file.exists():
            num_data = np.load(X_num_file, allow_pickle=True)
            df_array.append(num_data)
        else:
            print("Unable to find numerical data file.")
        if y_file.exists():
            y_data = np.load(y_file, allow_pickle=True)
            df_array.append(y_data.reshape(-1, 1))
        else:
            print("Unable to find target data file.")
            return None
        df_stack = np.hstack(df_array)

        column_headers = []
        if cat_data is not None:
            column_headers += [f'cat_{i}' for i in range(cat_data.shape[1])]
        if num_data is not None:
            column_headers += [f'num_{i}' for i in range(num_data.shape[1])]
        if y_data is not None:
            column_headers += ['target']

        df = pd.DataFrame(df_stack, columns=column_headers)
    
    print("Created dataframe with shape", df.shape)
    print("Sample from dataframe:", df.head())
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_name', type=str, default="adult")
    parser.add_argument('sampling_method', type=str, default="ddpm")
    parser.add_argument('eval_type',  type=str, default='synthetic')
    parser.add_argument('model_type',  type=str, default='catboost')
    parser.add_argument('--normalized', action='store_true', default=True)
    parser.add_argument('--file_path', type=str, default=None)

    args = parser.parse_args()
    df = convert_to_df(
        ds_name=args.ds_name,
        sampling_method=args.sampling_method,
        eval_type=args.eval_type,
        model_type=args.model_type,
        normalized=args.normalized
    )
    if df is not None:
        df.to_csv(f'{args.ds_name}_{args.sampling_method}_{args.eval_type}_{args.model_type}_{"normalized" if args.normalized else "unnormalized"}.csv', index=False)
        print("Dataframe saved to CSV.")


if __name__ == '__main__':
    main()