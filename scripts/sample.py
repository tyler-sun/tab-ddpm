import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError
from utils_train import get_model, make_dataset
from lib import round_columns
import lib

def process_risk_condition(y_conditional, y_range, y_values, num_samples, seed=0):

    if y_conditional is None:
        return np.asarray(y_values, dtype=float).reshape(-1)
    y_conditional = y_conditional.to(device='cpu')
    values = (
        [float(y_conditional)]
        if isinstance(y_conditional, (int, float))
        else np.asarray(y_conditional, dtype=float).reshape(-1)
    )

    if not y_range or len(values) == 1:
        return np.asarray(values, dtype=float)

    lo = float(np.min(values))
    hi = float(np.max(values))
    subset = np.asarray(y_values, dtype=float)
    mask = (subset >= lo) & (subset <= hi)
    subset = subset[mask] if np.any(mask) else subset

    rng = np.random.default_rng(seed)
    return rng.choice(subset, size=num_samples, replace=True).astype(float)

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)


def _sample_y_values_in_range(y_values, target_values, num_samples, seed=0):
    target_values = np.asarray(target_values).reshape(-1)
    if target_values.size == 0:
        return np.array([], dtype=float)

    if isinstance(y_values, (int, float)):
        y_values = [float(y_values)]
    y_values = np.asarray(y_values, dtype=float).reshape(-1)
    if y_values.size == 0:
        return np.array([], dtype=float)

    lo = float(np.min(y_values))
    hi = float(np.max(y_values))
    mask = (target_values >= lo) & (target_values <= hi)
    subset = target_values[mask]
    if subset.size == 0:
        subset = target_values

    rng = np.random.default_rng(seed)
    sampled = rng.choice(subset, size=num_samples, replace=True)
    return sampled.astype(float)


def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cuda:1'),
    seed = 0,
    change_val = False,
    append = False,
    use_risk_variable = False,
    y_conditional = None,
    y_cond_weights = None,
    y_range = False,
    guidance_scale = 0.0
):
    zero.improve_reproducibility(seed)
    print("Sampling with seed", seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val,
        use_risk_variable=use_risk_variable
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device,
        cfg_rate=guidance_scale
    )
    if guidance_scale > 0.0:
        print("Sampling with guidance scale", guidance_scale)
    diffusion.to(device)
    diffusion.eval()
    
    # sampling with continuous risk conditioning
    # sample from actual continuous values from training set/config
    if model_params['is_y_cond'] and (D.is_regression or use_risk_variable):
        print("Sampling on continuous variable.")
        if y_conditional is not None:
            y_values = [y_conditional] if isinstance(y_conditional, (int, float)) else y_conditional
            y_values = torch.tensor(y_values).float().to(device)
            if y_values.dim() == 1:
                y_values = y_values.unsqueeze(-1)
            if y_range:
                if y_values.numel() == 0:
                    print("y_range requested but no y values were provided")
                    return
                y_values_np = process_risk_condition(
                    y_conditional=y_values,
                    y_range=y_range,
                    y_values=D.y['train'],
                    num_samples=num_samples,
                    seed=seed
                )
                y_values = torch.from_numpy(y_values_np).float().to(device)
                if y_values.dim() == 1:
                    y_values = y_values.unsqueeze(-1)
                x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, y_values=y_values, ddim=False, guidance_scale=guidance_scale)
            elif y_cond_weights is not None:
                if len(y_cond_weights) != len(y_values):
                    print("y_cond_weights must be the same length as conditioning y values")
                    return
                x_gen, y_gen = [], []
                for weight, y_value in zip(y_cond_weights, y_values):
                    weighted_samples = round(num_samples*weight)
                    print(f"Generating {weighted_samples} samples conditioned on y={y_value}")
                    x_temp, y_temp = diffusion.sample_all(round(num_samples*weight), batch_size, y_values=y_value, ddim=False, guidance_scale=guidance_scale)
                    x_gen.append(x_temp)
                    y_gen.append(y_temp)

                x_gen = torch.cat(x_gen, dim=0)
                y_gen = torch.cat(y_gen, dim=0)
            else:
                x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, y_values=y_values, ddim=False, guidance_scale=guidance_scale)
        else:
            # find and sample from the training dataset's risk distribution
            y_values = torch.from_numpy(D.y['train']).float()
            if y_values.dim() == 1:
                y_values = y_values.unsqueeze(-1)
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, y_values=y_values, ddim=False, guidance_scale=guidance_scale)
    # discrete class sampling
    # find and sample from the training dataset's class distribution
    else:
        labels, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
        if disbalance == 'fix':
            empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
            if y_conditional is not None:
                if y_conditional is not int or y_conditional not in labels:
                    print(f"{y_conditional} not found in training labels")
                    return
                empirical_class_dist = torch.zeros_like(empirical_class_dist)
                empirical_class_dist[y_conditional] = 1
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, y_dist=empirical_class_dist.float(), ddim=False, guidance_scale=guidance_scale)

        elif disbalance == 'fill':
            ix_major = empirical_class_dist.argmax().item()
            val_major = empirical_class_dist[ix_major].item()
            x_gen, y_gen = [], []
            # generate minority class samples for the specified class only
            if y_conditional is not None:
                # if y_conditional is not int or y_conditional not in labels:
                #     print(f"{y_conditional} not found in training labels")
                #     return
                if y_conditional != ix_major:
                    distrib = torch.zeros_like(empirical_class_dist)
                    distrib[i] = 1
                    num_samples = val_major - empirical_class_dist[y_conditional].item()
                    x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, y_dist=distrib.float(), ddim=False, guidance_scale=guidance_scale)
                    x_gen.append(x_temp)
                    y_gen.append(y_temp)
            # generate minority class samples for all minority classes until they have the same number of samples as the majority class
            else:
                for i in range(empirical_class_dist.shape[0]):
                    if i == ix_major:
                        continue
                    distrib = torch.zeros_like(empirical_class_dist)
                    distrib[i] = 1
                    num_samples = val_major - empirical_class_dist[i].item()
                    x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, y_dist=distrib.float(), ddim=False, guidance_scale=guidance_scale)
                    x_gen.append(x_temp)
                    y_gen.append(y_temp)
            
            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)

        else:
            if y_conditional is not None:
                # if y_conditional is not int:
                #     print(f"{y_conditional} not found in training labels")
                #     return
                empirical_class_dist = torch.zeros_like(empirical_class_dist)
                empirical_class_dist[y_conditional] = 1
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, y_dist=empirical_class_dist.float(), ddim=False, guidance_scale=guidance_scale)

    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )
    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    # Map sampled class indices back to original label values (handles datasets
    # where labels aren't 0..K-1 or when only a subset of classes is present).
    try:
        labels_np = labels.numpy()
        y_gen = labels_np[y_gen]
    except Exception:
        pass

    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    data_new = []
    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

        # if use_risk_variable:
        #     y_gen = X_cat[:, -1].astype(int)
        #     X_cat = X_cat[:, :-1]
        data_new.append(X_cat)

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0 and not model_params['is_y_cond']:
            if not use_risk_variable:
                y_gen = X_num[:, 0]
                X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)
        data_new.append(X_num)

    print("Appending to existing files:", append)
    if append:
        # iterate through and append to corresponding files in parent_dir
        files = []
        if num_numerical_features < X_gen.shape[1]:
            files.append('X_cat_train.npy')
        if num_numerical_features != 0:
            files.append('X_num_train.npy')
        files.append('y_train.npy')
        for file, data in zip(files, data_new):
            data_path = os.path.join(parent_dir, file)
            if os.path.exists(data_path):
                data_old = np.load(data_path, allow_pickle=True)
                data_combined = np.vstack([data_old, data])
                np.save(data_path, data_combined)
                print(f"Appended {data.shape[0]} samples to {data_path} for {data_combined.shape[0]} total samples.")
    else:
        print("Saving synthetic data to", parent_dir)
        if num_numerical_features != 0:
            print("Numerical features shape: ", X_num.shape)
            np.save(os.path.join(parent_dir, 'X_num_train'), X_num)

        if num_numerical_features < X_gen.shape[1]:
            print("Categorical features shape: ", X_cat.shape)
            np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)

        np.save(os.path.join(parent_dir, 'y_train.npy'), y_gen)
        data_new.append(y_gen)
