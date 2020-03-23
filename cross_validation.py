import numpy as np
import catboost
from sklearn.metrics import auc, roc_auc_score
import lightgbm as lgb

def cross_validation(cv, model, X, y, metrics=[roc_auc_score], verbose=True, train_params={}):
    scores = {}
    for metric in metrics:
        scores[metric.__name__] = {'train': [], 'val': []}
    modeltype = train_params.pop('modeltype', None)
    cat_features = train_params.pop('cat_features', None)

    for train_index, val_index in cv.split(X, y):
        X_train, X_val, y_train, y_val = X.loc[train_index], X.loc[val_index], y.loc[train_index], y.loc[val_index]

        if modeltype == 'lgb':
            train_dataset = lgb.Dataset(X_train, y_train, free_raw_data=False)
            val_dataset = lgb.Dataset(X_val, y_val, free_raw_data=False)

            model = lgb.train(train_set=train_dataset, valid_sets=[val_dataset], **train_params)

            train_predictions_proba = model.predict(X_train)
            val_predictions_proba = model.predict(X_val)

        elif modeltype == 'catboost':
            train_dataset = catboost.Pool(X_train, y_train, cat_features=cat_features,
                                          feature_names=list(X_train.columns), thread_count=1)
            val_dataset = catboost.Pool(X_val, y_val, cat_features=cat_features, feature_names=list(X_train.columns),
                                        thread_count=1)

            model = catboost.CatBoostClassifier(**train_params['params'])
            model.fit(train_dataset, eval_set=val_dataset, **train_params['fit_params'])

            train_predictions_proba = model.predict_proba(X_train).T[1]
            val_predictions_proba = model.predict_proba(X_val).T[1]
        else: # any other sklearn-like interface model
            model.fit(X_train, y_train)

            train_predictions_proba = model.predict_proba(X_train).T[1]
            val_predictions_proba = model.predict_proba(X_val).T[1]

        train_predictions = np.round(train_predictions_proba)
        val_predictions = np.round(val_predictions_proba)

        # metric calculation
        for index, metric in enumerate(metrics):
            if metric.__name__ in ['precision_recall_curve', 'roc_curve']:
                train_score = auc(*metric(y_train, train_predictions_proba)[:2][::-1])
                val_score = auc(*metric(y_val, val_predictions_proba)[:2][::-1])
            elif metric.__name__ == 'roc_auc_score':
                train_score = metric(y_train, train_predictions_proba)
                val_score = metric(y_val, val_predictions_proba)
            else:
                train_score = metric(y_train, train_predictions)
                val_score = metric(y_val, val_predictions)

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
