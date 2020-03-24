import catboost
import lightgbm as lgb
import numpy as np
from sklearn.metrics import auc, roc_auc_score


def cross_validation(cv, model, features, target, metrics=(roc_auc_score,), verbose=True, train_params={}):
    """

    :param cv: sklearn.model_selection class object
    :param model: sklearn interface-like classifier model
    :param features: 'list or ndarray'
    :param target: 'list or 1darray'
    :param metrics: 'list or tuple or set of functions'
    :param verbose: bool
    :param train_params: dict - e.g. {'modeltype': either 'lgb' or 'catboost' or None(default),
                                      'catboost_cat_features': list of indexes or None(default),
                                      'fit_params': dict}
    :return: dict - e.g. {metric.__name__: {'train': list(len=cv.n_splits),
                                            'val': list,
                                            'train_mean': float,
                                            'val_mean': float},
                         ...}
            model fitted on the last train_fold split
    """
    scores = {}
    for metric in metrics:
        scores[metric.__name__] = {'train': [], 'val': []}
    modeltype = train_params.pop('modeltype', None)
    catboost_cat_features = train_params.pop('catboost_cat_features', None)

    for train_index, val_index in cv.split(features, target):
        train_features, val_features = features[train_index], features[val_index]
        train_target, val_target =  target[train_index], target[val_index]

        if modeltype in ['lgb', 'catboost']:
            if modeltype == 'lgb':
                train_dataset = lgb.Dataset(train_features, train_target, free_raw_data=False)
                val_dataset = lgb.Dataset(val_features, val_target, free_raw_data=False)

            elif modeltype == 'catboost':
                train_dataset = catboost.Pool(train_features, train_target, cat_features=catboost_cat_features,
                                              feature_names=list(train_features.columns), thread_count=1)
                val_dataset = catboost.Pool(val_features, val_target, cat_features=catboost_cat_features,
                                            feature_names=list(train_features.columns), thread_count=1)
            model.fit(train_dataset, eval_set=val_dataset, **train_params['fit_params'])

        else:  # any other sklearn-like interface model
            if 'fit_params' in train_params:
                model.fit(train_features, train_target, **train_params['fit_params'])
            else:
                model.fit(train_features, train_target)

        train_predictions_proba = model.predict_proba(train_features).T[1]
        val_predictions_proba = model.predict_proba(val_features).T[1]

        train_predictions = np.round(train_predictions_proba)
        val_predictions = np.round(val_predictions_proba)

        # metric calculation
        for index, metric in enumerate(metrics):
            if metric.__name__ in ['precision_recall_curve', 'roc_curve']:
                train_score = auc(*metric(train_target, train_predictions_proba)[:2][::-1])
                val_score = auc(*metric(val_target, val_predictions_proba)[:2][::-1])
            elif metric.__name__ == 'roc_auc_score':
                train_score = metric(train_target, train_predictions_proba)
                val_score = metric(val_target, val_predictions_proba)
            else:
                train_score = metric(train_target, train_predictions)
                val_score = metric(val_target, val_predictions)

            scores[metric.__name__]['train'].append(train_score)
            scores[metric.__name__]['val'].append(val_score)

    for metric in metrics:
        if verbose:
            print(metric.__name__)
        for key in ['train', 'val']:
            scores[metric.__name__][key] = np.round(scores[metric.__name__][key], 5)
            scores[metric.__name__][f'{key}_mean'] = round(np.mean(scores[metric.__name__][key]), 5)
            if verbose:
                print(f"{key.upper()}: {scores[metric.__name__][key]} ({scores[metric.__name__][key + '_mean']})")

    return scores, model

if __name__ == '__main__':
    print(cross_validation.__doc__)