import torch
import numpy as np
import pandas as pd
import argparse

def get_distributions(diffusion, D, batch_size, num_samples, disbalance, fixed_class=None):
    all_samples = []
    all_labels = []
    num_generated = 0
    while num_generated < num_samples:
        current_batch_size = min(batch_size, num_samples - num_generated)
        samples = diffusion.sample(current_batch_size, device=diffusion.device)
        samples_np = samples.cpu().numpy()
        if fixed_class is not None:
            class_indices = np.where(samples_np[:, -1] == fixed_class)[0]
            samples_np = samples_np[class_indices]
        all_samples.append(samples_np[:, :-1])
        all_labels.append(samples_np[:, -1])
        num_generated += len(samples_np)
    
    all_samples = np.vstack(all_samples)
    all_labels = np.hstack(all_labels)

    if disbalance is not None and fixed_class is None:
        class_counts = np.bincount(all_labels.astype(int))
        total_count = len(all_labels)
        class_ratios = class_counts / total_count
        desired_ratios = disbalance
        weights = desired_ratios / class_ratios
        sample_weights = weights[all_labels.astype(int)]
        sample_weights /= sample_weights.sum()
        indices_to_sample = np.random.choice(len(all_labels), size=num_samples, replace=True, p=sample_weights)
        all_samples = all_samples[indices_to_sample]
        all_labels = all_labels[indices_to_sample]

    return all_samples, all_labels

def get_class_distribution(file_path):
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
        labels = data[:, -1]
    else:
        labels = data
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique.astype(int), counts))
    percents = {k: f"{(v / len(labels) * 100):.2f}%" for k, v in distribution.items()}
    return distribution, percents, data.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()

    distribution, percents, shape = get_class_distribution(args.file_path)
    print(f"Class distribution in {args.file_path}: {distribution}, {percents}")
    print(f"Total samples:", shape[0])


if __name__ == '__main__':
    main()