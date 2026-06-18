import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset
from lib import round_columns
import lib
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from skorch.classifier import NeuralNetClassifier
from lib import concat_features, get_catboost_config, concat_to_df

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample_1_class(
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
    fixed_class = 1,
    boundary_filter = False,
    classifier = "catboost"
):
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
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
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )

    diffusion.to(device)
    diffusion.eval()

    print("Generating samples of class", fixed_class)
    
    labels, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()
    if disbalance == 'fix':
        empirical_class_dist[0], empirical_class_dist[1] = empirical_class_dist[1], empirical_class_dist[0]
        distribution = torch.zeros_like(empirical_class_dist)
        distribution[fixed_class] = 1
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, distribution.float(), ddim=False)

    elif disbalance == 'fill':
        ix_major = empirical_class_dist.argmax().item()
        val_major = empirical_class_dist[ix_major].item()
        x_gen, y_gen = [], []
        # If fixed_class is set, only generate for that class
        if 'fixed_class' in locals() or 'fixed_class' in globals():
            i = fixed_class
            if i != ix_major:
                distrib = torch.zeros_like(empirical_class_dist)
                distrib[i] = 1
                num_samples = val_major - empirical_class_dist[i].item()
                x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
                x_gen.append(x_temp)
                y_gen.append(y_temp)
            # If fixed_class is the majority, nothing to fill
        else:
            for i in range(empirical_class_dist.shape[0]):
                if i == ix_major:
                    continue
                distrib = torch.zeros_like(empirical_class_dist)
                distrib[i] = 1
                num_samples = val_major - empirical_class_dist[i].item()
                x_temp, y_temp = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
                x_gen.append(x_temp)
                y_gen.append(y_temp)
        if x_gen:
            x_gen = torch.cat(x_gen, dim=0)
            y_gen = torch.cat(y_gen, dim=0)

    else:
        # Generate only the class specified by 'fixed_class'
        # 'fixed_class' should be passed as an argument to sample()
        if 'fixed_class' in locals() or 'fixed_class' in globals():
            distrib = torch.zeros_like(empirical_class_dist)
            distrib[fixed_class] = 1
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, distrib.float(), ddim=False)
        else:
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


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

    # use xgboost classifier as catboost and mlp are already used for the final evaluation?
    # config_path = lib.load_config(parent_dir / "config.toml")
    #classifier = XGBClassifier(
    # classifier = CatBoostClassifier(
    #     loss_function="MultiClass" if D.is_multiclass else "Logloss",
    #     **config['eval']['model_params'],
    #     eval_metric='TotalF1',
    #     random_seed=seed,
    #     class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
    # )
    # if classifier == "xgboost":
    #     boundary_clf = XGBClassifier(
    #         **config['eval']['model_params'],
    #         eval_metric='logloss',
    #         random_seed=seed,
    #         use_label_encoder=False
    #     )
    # elif
    if boundary_filter:
        if classifier == "catboost":
            catboost_config = get_catboost_config(real_data_path, is_cv=True)
            boundary_clf = CatBoostClassifier(
                loss_function="MultiClass" if D.is_multiclass else "Logloss",
                **catboost_config,
                eval_metric='TotalF1',
                random_seed=seed,
                class_names=[str(i) for i in range(D.n_classes)] if D.is_multiclass else ["0", "1"]
            )
    # elif classifier == "mlp":
    #     boundary_clf = NeuralNetClassifier(
    #         model,
    #         criterion=BCEWithLogitsLoss if D.is_binclass else CrossEntropyLoss,
    #         optimizer=AdamW,
    #         lr=params["lr"],
    #         optimizer__weight_decay=params["weight_decay"],
    #         batch_size=128 if len(D.y["train"]) < 10_000 else 256,
    #         max_epochs=1000,
    #         train_split=predefined_split(val_ds),
    #         iterator_train__shuffle=True,
    #         device=device,
    #         callbacks=[es, EpochScoring(f1, lower_is_better=False)],
    #     )

            # filter for samples close to the decision boundary
            X = concat_features(D)
            boundary_clf.fit(X['train'], D.y['train'])

            X_gen_catboost = concat_to_df(D, X_gen)
            probs = boundary_clf.predict_proba(X_gen_catboost)[:, 1]

            threshold = 0.5
            margin = np.abs(probs - threshold)
            boundary_indices = margin < 0.25
            X_gen = X_gen[boundary_indices]
            y_gen = y_gen[boundary_indices]
            print(f"Filtered to {X_gen.shape[0]} samples close to the decision boundary")

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

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
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    if num_numerical_features != 0:
        print("Num shape: ", X_num.shape)
        np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
    np.save(os.path.join(parent_dir, 'y_train'), y_gen)