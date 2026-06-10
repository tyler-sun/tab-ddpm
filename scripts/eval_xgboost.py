from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, r2_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
from sklearn.utils import shuffle
import zero
import matplotlib.pyplot as plt
from pathlib import Path
import lib
from pprint import pprint
from lib import concat_features, read_pure_data, get_xgboost_config, read_changed_val

def train_xgboost(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    params = None,
    change_val = True,
    device = None # dummy
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = lib.load_json(os.path.join(real_data_path, 'info.json'))
    T = lib.Transformations(**T_dict)
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2, random_state=seed)

    X = None
    print("running training of xgboost classifier...")
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
            X_num_real, X_cat_real, y_real = read_pure_data(real_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)

        y = np.concatenate([y_real, y_fake], axis=0)

        X_num = None
        if X_num_real is not None:
            print("merging numerical features...")
            X_num = np.concatenate([X_num_real, X_num_fake], axis=0)

        X_cat = None
        if X_cat_real is not None:
            print("merging categorical features...")
            X_cat = np.concatenate([X_cat_real, X_cat_fake], axis=0)

    elif eval_type == 'synthetic':
        print(f'loading synthetic data: {parent_dir}')
        X_num, X_cat, y = read_pure_data(synthetic_data_path)

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path)
        else:
            X_num, X_cat, y = X_num_real, X_cat_real, y_real
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val')
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test')

    D = lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        lib.TaskType(info['task_type']),
        info.get('n_classes')
    )

    D = lib.transform_dataset(D, T, None)
    X = concat_features(D)
    print(f'Train size: {X["train"].shape}, Val size {X["val"].shape}')

    # set is_cv to False for fraudDiffuse replication and True for tuned parameters from Optuna
    if params is None:
        xgboost_config = get_xgboost_config(real_data_path, is_cv=False)
    else:
        xgboost_config = params

    # if 'cat_features' not in xgboost_config:
    #     xgboost_config['cat_features'] = list(range(D.n_num_features, D.n_features))

    # for col in range(D.n_features):
    #     for split in X.keys():
    #         if col in xgboost_config['cat_features']:
    #             X[split][col] = X[split][col].astype(str)
    #         else:
    #             X[split][col] = X[split][col].astype(float)
    if 'cat_features' in xgboost_config:
        for col in xgboost_config['cat_features']:
            encoder = LabelEncoder()
            X['train'].iloc[:, col] = encoder.fit_transform(X['train'].iloc[:, col].astype(str))
            X['val'].iloc[:, col] = encoder.transform(X['val'].iloc[:, col].astype(str))
            X['test'].iloc[:, col] = encoder.transform(X['test'].iloc[:, col].astype(str))

    for col in range(D.n_num_features):
        for split in X:
            X[split].iloc[:, col] = X[split].iloc[:, col].astype(float)

    print(T_dict)
    pprint(xgboost_config, width=100)
    print('-'*100)
    
    if D.is_regression:
        model = XGBRegressor(
            **xgboost_config,
            eval_metric='rmse',
            random_state=seed
        )
        predict = model.predict
    else:
        objective = "multi:softprob" if D.is_multiclass else "binary:logistic"
        # for binary classification, can set as error, logloss (probabilistic measure) or auc (for ranking performance)
        eval_metric = "mlogloss" if D.is_multiclass else "aucpr"
        model = XGBClassifier(
            objective=objective,
            **xgboost_config,
            eval_metric=eval_metric,
            random_state=seed
        )
        predict = (
            model.predict_proba
            if D.is_multiclass
            else lambda x: model.predict_proba(x)[:, 1]
        )

    model.fit(
        X['train'], D.y['train'],
        eval_set=[(X['train'], D.y['train']), (X['val'], D.y['val'])],
        verbose=100
    )
    predictions = {k: predict(v) for k, v in X.items()}
    print(predictions['train'].shape)

    report = {}
    report['eval_type'] = eval_type
    report['dataset'] = real_data_path
    report['metrics'] = D.calculate_metrics(predictions,  None if D.is_regression else 'probs')

    metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_xgboost.json"))

        # add plot for training progression
        # epochs = len(model.evals_result()['validation_0'][eval_metric])
        # iterations = range(1, epochs+1)
        # plt.plot(iterations, model.evals_result()['validation_0'][eval_metric], label='Validation')
        # plt.xlabel("Boosting Round")
        # plt.ylabel(eval_metric)
        # plt.title("XGBoost Classifier Training Curve")
        # fig_path = os.path.join(parent_dir, "xgboost_training.png")
        # plt.savefig(fig_path)
        # print(f"Saved training plot to {fig_path}")

    return metrics_report, model.evals_result(), eval_metric

    