"""
Module containing various utility methods.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from scipy import sparse


def get_relational_entities(y_hat, target_ids, relations, data_dir='', logger=None):
    """
    Generates pseudo-relational features using given predictions and relations.
    """

    # classification nodes
    target_col = 'com_id'
    target_priors = list(zip(target_ids, y_hat))

    # relational nodes
    relations_dict = {}

    for relation in relations:
        if logger:
            logger.info('creating predicates for {}...'.format(relation))

        # read connections data
        relation_id = '{}_id'.format(relation)
        fp = os.path.join(data_dir, 'relation_{}.csv'.format(relation))
        df = pd.read_csv(fp)

        # filter out connections not in the target_ids list
        df = df[df[target_col].isin(target_ids)]
        connections = list(zip(df[relation_id], df[target_col]))
        relations_dict[relation_id] = connections

    return target_priors, relations_dict, target_col


def pseudo_relational_features(y_hat, target_ids, relations, data_dir='',
                               return_sparse=True, verbose=0):
    """
    Generates pseudo-relational features using given predictions and relations.
    """
    Xr = []
    Xr_cols = []

    for relation in relations:
        fp = os.path.join(data_dir, 'relation_{}.csv'.format(relation))
        x, name = _create_feature(y_hat, target_ids, fp, relation,
                                  return_sparse=return_sparse)
        Xr.append(x)
        Xr_cols.append(name)

    Xr = hstack(Xr)
    return Xr, Xr_cols


def cross_val_predict(X,
                      y,
                      estimator,
                      param_grid=None,
                      num_cvfolds=5,
                      num_tunefolds=3,
                      logger=None,
                      random_state=None):
    """
    Generates predictions for all instances in X using cross-validation.
    """

    # create folds
    np.random.seed(random_state)
    cv_folds = np.random.randint(num_cvfolds, size=X.shape[0])

    # store predictions
    p = y.copy().astype(float)

    # make predictions on each fold
    for i, (train_index, test_index) in enumerate(PredefinedSplit(cv_folds).split()):
        start = time.time()

        X_train, y_train = X[train_index], y[train_index]
        X_test = X[test_index]

        # tune the hyperparameters on this training fold
        if param_grid is not None:
            np.random.seed(random_state)
            tune_folds = np.random.randint(num_tunefolds, size=X_train.shape[0])

            model = GridSearchCV(clone(estimator), cv=PredefinedSplit(tune_folds),
                                 param_grid=param_grid)
            model = clone(model).fit(X_train, y_train)

        else:
            model = clone(estimator).fit(X_train, y_train)

        # make predictions on this test set
        y_score = model.predict_proba(X_test)[:, 1]
        np.put(p, test_index, y_score)

        if logger:
            logger.info('[CV] fold {}: {:.3f}s'.format(i, time.time() - start))

    assert len(p) == len(y)
    return p


def hstack(blocks):
    """
    Horizontally stacks sparse or non-sparse blocks.
    """

    # checks if all blocks are sparse
    sparse_blocks = 0
    for block in blocks:
        if sparse.issparse(block):
            sparse_blocks += 1

    # stacks the blocks together
    if sparse_blocks == len(blocks):
        result = sparse.hstack(blocks).tocsr()
    elif sparse_blocks == 0:
        result = np.hstack(blocks)
    else:
        raise ValueError('Sparse and non-sparse blocks present!')

    return result


# private
def _create_feature(y_hat, target_ids, fp, relation, return_sparse=True):
    """
    Generates pseudo-relational feature values for a given relation.
    """

    relation_id = '{}_id'.format(relation)
    feature_name = relation_id.split('_')[0] + '_relation'

    # get relational info
    rel_df = pd.read_csv(fp)[['com_id', relation_id]]

    # merge predictions with relational information
    data_df = pd.DataFrame(list(zip(y_hat, target_ids)), columns=['y_hat', 'com_id'])

    # get comments, hub ids, and predictions in one dataframe
    temp_df = rel_df.merge(data_df, on='com_id')

    # compute mean predictions for each hub
    hub_df = temp_df.groupby(relation_id)['y_hat'].mean().reset_index()\
                    .rename(columns={'y_hat': 'mean_y_hat'})

    # merge hub predictions to individual comments
    temp_df = temp_df.merge(hub_df, on=relation_id)[['com_id', 'mean_y_hat']]

    # merge comment hub predictions back onto all comments
    data_df = data_df.merge(temp_df, on='com_id', how='left')
    data_df = data_df.rename(columns={'mean_y_hat': feature_name})
    data_df[feature_name] = data_df[feature_name].fillna(data_df['y_hat'])

    # sparsify if specified
    x = data_df[feature_name].to_numpy().reshape(-1, 1)
    if return_sparse:
        x = sparse.csr_matrix(x)

    return x, feature_name
