import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def percent_range(min, max):
    def check_validity(value):
        try:
            value = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Value must be an integer between {min} and {max}")
        if value < min or value > max:
            raise argparse.ArgumentTypeError(f"Value must be between {min} and {max}")
        return value
    return check_validity


def csv_to_npy(csv_file, labels_column, output_dir, train_split=None, val_split=None):
    df = pd.read_csv(csv_file)
    if df is None:
        print(f"Failed to find CSV file from {csv_file}")
        return
    if df.empty:
        print("CSV file is empty.")
        return
    print(f"Loaded CSV with shape: {df.shape}")

    test_split = 100
    if train_split is not None:
        test_split -= train_split
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(len(df) * (train_split / 100))
        if val_split is not None:
            test_split -= val_split
            val_size = int(len(df) * ((train_split + val_split) / 100))

    name_splits = []
    df_splits = []
    name_splits.append('train')
    if train_split is not None:
        df_splits.append(df_shuffled.iloc[:train_size])
        if val_split is not None:
            name_splits.append('val')
            df_splits.append(df_shuffled.iloc[train_size:val_size])
            df_splits.append(df_shuffled.iloc[val_size:])
        else:
            df_splits.append(df_shuffled.iloc[train_size:])
        name_splits.append('test')
    else:
        df_splits.append(df)

    print(f"Saving files for {len(name_splits)} split(s).")

    for name_split, df in zip(name_splits, df_splits):
        y = df[labels_column].values
        if labels_column not in df.columns:
            print(f"Column '{labels_column}' not found")
            return
        # Remove labels column from dataframe
        df = df.drop(columns=[labels_column])
        
        # Identify categorical and numerical columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        num_cols = df.select_dtypes(include=['number']).columns
        
        print(f"Found {len(cat_cols)} categorical columns")
        print(f"Found {len(num_cols)} numerical columns")
        
        # Extract data
        X_cat = df[cat_cols].values if not cat_cols.empty else None
        X_num = df[num_cols].values if not num_cols.empty else None
        
        # Save to .npy files
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        if X_cat is not None:
            X_cat_path = output_dir / f'X_cat_{name_split}.npy'
            np.save(X_cat_path, X_cat)
            print(f"Saved categorical data to {X_cat_path}")
        
        if X_num is not None:
            X_num_path = output_dir / f'X_num_{name_split}.npy'
            np.save(X_num_path, X_num)
            print(f"Saved numerical data to {X_num_path}")
        
        y_path = output_dir / f'y_{name_split}.npy'
        np.save(y_path, y)
        print(f"Saved target data to {y_path}")


def main():
    # credit card dataset was split into 65/15/20 for train/val/test
    parser = argparse.ArgumentParser()
    parser.add_argument('ds_path', type=str, default="data/credit-card/creditcard.csv")
    parser.add_argument('labels_column', type=str, default='class')
    parser.add_argument('output_dir', type=str, default="data/credit-card")
    parser.add_argument('--train_split', type=percent_range(1, 99), help="Percentage of data to use for training")
    parser.add_argument('--val_split', type=percent_range(1, 99), help="Percentage of data to use for validation, test split takes remaining percentage after training and validation splits")

    args = parser.parse_args()
    csv_to_npy(
        csv_file=args.ds_path,
        labels_column=args.labels_column,
        output_dir=args.output_dir,
        train_split=args.train_split,
        val_split=args.val_split
    )


if __name__ == '__main__':
    main()