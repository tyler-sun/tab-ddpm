from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report,
    r2_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
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
from get_distributions import get_class_distribution


def _select_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if precision.size == 0 or recall.size == 0 or thresholds.size == 0:
        return 0.5

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    if len(thresholds) == len(f1_scores) - 1:
        f1_scores = f1_scores[:-1]
        threshold_candidates = thresholds
    else:
        threshold_candidates = thresholds

    best_idx = int(np.nanargmax(f1_scores))
    if best_idx >= len(threshold_candidates):
        return 0.5
    return float(threshold_candidates[best_idx])


def _build_thresholded_metrics(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(np.int64)
    report = classification_report(y_true, y_pred, output_dict=True)
    report['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    report['roc_auc'] = roc_auc_score(y_true, probs)
    report['pr_auc'] = average_precision_score(y_true, probs)
    return report

def train_xgboost(
    parent_dir,
    real_data_path,
    eval_type,
    T_dict,
    seed = 0,
    params = None,
    change_val = True,
    device = None, # dummy
    risk = False,
    convert_to_class = False
):
    zero.improve_reproducibility(seed)
    if eval_type != "real":
        synthetic_data_path = os.path.join(parent_dir)
    info = lib.load_json(os.path.join(real_data_path, 'info.json'))
    T = lib.Transformations(**T_dict)
    
    if risk:
        target = 'r'
    else:
        target = 'y'
    # target = 'y'
    print(risk, target)
    
    if change_val:
        X_num_real, X_cat_real, y_real, X_num_val, X_cat_val, y_val = read_changed_val(real_data_path, val_size=0.2, random_state=seed, target_prefix=target)

    X = None
    print('-'*100)
    if eval_type == 'merged':
        print('loading merged data...')
        if not change_val:
             X_num_real, X_cat_real, y_real = read_pure_data(real_data_path, target_prefix=target)
        print("synthetic_data_path =", synthetic_data_path)
        X_num_fake, X_cat_fake, y_fake = read_pure_data(synthetic_data_path)
        print("Merging", y_real.shape, y_fake.shape)
        #print("Synthetic data:", X_cat_fake.dtype, type(X_cat_fake[0,0]))
        # print("Real data:", X_cat_real.dtype, type(X_cat_real[0,0]))
        if X_cat_fake is not None:
            X_cat_fake = np.asarray(X_cat_fake, dtype=np.int64)
            print(X_cat_fake.dtype)
        if y_fake.ndim == 2:
            y_fake= y_fake.squeeze(axis=1)

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
        X_cat = np.asarray(X_cat, dtype=np.int64)

    elif eval_type == 'real':
        print('loading real data...')
        if not change_val:
            X_num, X_cat, y = read_pure_data(real_data_path, target_prefix=target)
        else:
            X_num, X_cat, y = X_num_real, X_cat_real, y_real
    else:
        raise "Choose eval method"

    if not change_val:
        X_num_val, X_cat_val, y_val = read_pure_data(real_data_path, 'val', target_prefix=target)
    X_num_test, X_cat_test, y_test = read_pure_data(real_data_path, 'test', target_prefix=target)

    # convert risk probabilities to binary classification
    if risk and convert_to_class:
        # print(Path(real_data_path) / "y_train.npy")
        # _, percents, _ = get_class_distribution(Path(real_data_path) / "y_train.npy")

        # positive_ratio = float(percents['1.0'].strip('%')) / 100.0
        positive_ratio = 0.035
        target_quantile = (1.0 - positive_ratio) * 100
        r_train_path = Path(real_data_path) / "r_train.npy"
        r_train = np.load(r_train_path, allow_pickle=True)

        # Compute threshold from the risk scores (r)
        threshold = np.percentile(r_train, target_quantile)

        print(f"Quantile-Aligned Threshold: {threshold:.4f}")
        # Binarize risks using the quantile cutoff
        y = [0 if x < threshold else 1 for x in y]
        y_val = [0 if x < threshold else 1 for x in y_val]
        y_test = [0 if x < threshold else 1 for x in y_test]

        # y = [0 if x < 0.5 else 1 for x in y]
        # y_val = [0 if x < 0.5 else 1 for x in y_val]
        # y_test = [0 if x < 0.5 else 1 for x in y_test]

    D = lib.Dataset(
        {'train': X_num, 'val': X_num_val, 'test': X_num_test} if X_num is not None else None,
        {'train': X_cat, 'val': X_cat_val, 'test': X_cat_test} if X_cat is not None else None,
        {'train': y, 'val': y_val, 'test': y_test},
        {},
        lib.TaskType(info['task_type']),
        info.get('n_classes')
    )
    if convert_to_class:
        D.task_type = lib.TaskType('binclass')

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
        eval_metric = 'rmse'
        model = XGBRegressor(
            **xgboost_config,
            eval_metric=eval_metric,
            random_state=seed
        )
        predict = model.predict
        print("running training of xgboost regressor...")
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
        print("running training of xgboost classifier...")

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

    if risk and convert_to_class and not D.is_regression:
        # threshold = _select_best_threshold(D.y['val'], predictions['val'])
        print(f"Selected validation threshold for binary risk classification: {threshold:.4f}")
        report['threshold'] = threshold
        report['metrics'] = {
            split: _build_thresholded_metrics(
                D.y[split],
                predictions[split],
                threshold,
            )
            for split in predictions
        }
        # report['threshold'] = 0.5
        # report['metrics'] = {
        #     split: _build_thresholded_metrics(
        #         D.y[split],
        #         predictions[split],
        #         0.5,
        #     )
        #     for split in predictions
        # }
    else:
        report['metrics'] = D.calculate_metrics(predictions, None if D.is_regression else 'probs')

    metrics_report = lib.MetricsReport(report['metrics'], D.task_type)
    metrics_report.print_metrics()

    if parent_dir is not None:
        lib.dump_json(report, os.path.join(parent_dir, "results_xgboost.json"))

    return metrics_report, model.evals_result(), eval_metric

    