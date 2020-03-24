from hyperopt import fmin, tpe, STATUS_OK, Trials

from cross_validation import cross_validation


def hyperparameters_optimization(cv, X, y, model, space_search, max_evals: int, base_params: dict = {}, mode: str = ''):
    """Bayes optimization with hyperopt"""

    # modeltype = base_params.get('modeltype', None)

    def objective(space_search):
        model.set_params(**space_search)
        # if model is not None:
        #     model.set_params(**space_search)
        # else:
        #     if 'params' in base_params:
        #         base_params['params'].update(space_search)
        #     else:
        #         base_params.update(space_search)
        #     base_params['modeltype'] = modeltype
        #         print(space_search, model)
        scores = cross_validation(cv, model, X, y, verbose=True, train_params=base_params)[0]
        if mode == 'overfit':  # consider difference between train and val metrics
            return {'loss': -scores['roc_auc_score']['val_mean'] +
                            max(0, (scores['roc_auc_score']['train_mean'] - scores['roc_auc_score']['val_mean'])),
                    'status': STATUS_OK, 'scores': scores, 'params': space_search}
        return {'loss': -scores['roc_auc_score']['val_mean'], 'status': STATUS_OK, 'scores': scores,
                'params': space_search}

    trials = Trials()
    best = fmin(fn=objective,
                space=space_search,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    return best, sorted(trials.results, key=lambda x: x['loss'])
