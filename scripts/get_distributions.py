import torch
import numpy as np
import pandas as pd
import argparse

def get_risk_distribution(file_path, labels_col=None):
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
        else:
            print("Expected CSV or NPY data file")
            return None
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None

    if len(data.shape) > 1:
        if labels_col and file_path.endswith('.csv'):
            labels = df[labels_col]
        else:
            labels = data[:, -1]
    else:
        labels = data
    # group risk labels into bins of 0.1 and report quantities in each bin
    df = pd.DataFrame({'risk': labels})
    df['risk_bin'] = pd.cut(df['risk'], bins=np.arange(0, 1.1, 0.1), right=False)

    return df['risk_bin'].value_counts().sort_index().to_dict()

def get_class_distribution(file_path, labels_col=None):
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
        else:
            print("Expected CSV or NPY data file")
            return None
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None

    if len(data.shape) > 1:
        if labels_col and file_path.endswith('.csv'):
            labels = df[labels_col]
        else:
            labels = data[:, -1]
    else:
        labels = data
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.astype(int), counts))
    percents = {k: f"{(v / len(labels) * 100):.2f}%" for k, v in distribution.items()}
    return distribution, percents, data.shape


def get_stats(file_path, labels_col=None):
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
            if labels_col and len(data) > 1:
                data = df[labels_col]
        else:
            print("Expected CSV or NPY data file")
            return None
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None
    
    min, max = np.min(data, axis=0), np.max(data, axis=0)
    range = max - min
    mean, std = np.mean(data, axis=0), np.std(data, axis=0)
    stats = {
        'min': min,
        'max': max,
        'range': range,
        'mean': mean,
        'std': std
    }
    return stats


def check_for_null(file_path):
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path, allow_pickle=True)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            data = df.values
        else:
            print("Expected CSV or NPY data file")
            return None
        print(f"Loaded data from {file_path}")
    except Exception as e:
        print(f"Failed to load data from {file_path}: {e}")
        return None
    has_null = pd.isna(data).any()
    if 'num' in file_path:
        print(np.isnan(data).any())
        print(np.isinf(data).any())
        print(np.nanmax(data))
        print(np.nanmin(data))
    elif 'cat' in file_path:
        print((data == None).any())
    print(f"Data in {file_path} contains NaN values: {has_null}")
    return has_null


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    parser.add_argument('--no_distribution', action='store_true', default=False)
    parser.add_argument('--labels_col', type=str, default=None)
    parser.add_argument('--stats', action='store_true', default=False)
    parser.add_argument('--nan', action='store_true', default=False)
    parser.add_argument('--risk', action='store_true', default=False)
    args = parser.parse_args()

    if not args.no_distribution:
        distribution, percents, shape = get_class_distribution(args.file_path, labels_col=args.labels_col)
        print(f"Class distribution in {args.file_path}: {distribution}, {percents}")
        print(f"Total samples:", shape[0])
    if args.stats:
        stats = get_stats(args.file_path, labels_col=args.labels_col)
        print(f"Dataset stats:")
        for key, value in stats.items():
            print(f"{key}: {value}")
    if args.nan:
        check_for_null(args.file_path)
    if args.risk:
        risk_bins = get_risk_distribution(args.file_path, labels_col=None)
        print(f"Risk distribution in {args.file_path}:\n{risk_bins}")


if __name__ == '__main__':
    main()