"""
This script uses EGGS to model a spam dataset.
"""
import os
import sys
import time
import argparse
from datetime import datetime
here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, here + '/../../')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz

import eggs
import util


def _get_model(args, data_dir, logger):
    """
    Initialize model.
    """
    if args.base_estimator == 'lr':
        base_estimator = LogisticRegression(solver='liblinear')

    elif args.base_estimator == 'lgb':
        base_estimator = lgb.LGBMClassifier(n_estimators=args.n_estimators,
                                            max_depth=args.max_depth,
                                            random_state=args.rs)
    else:
        raise ValueError('base_estimator {} unknown!'.format(args.base_estimator))

    if args.eggs:
        assert args.sgl_stacks > 0 or args.pgm in ['mrf', 'psl']
        model = eggs.EGGSClassifier(estimator=base_estimator,
                                    relations=args.relations,
                                    sgl_method=args.sgl_method,
                                    sgl_stacks=args.sgl_stacks,
                                    pgm=args.pgm,
                                    psl_learner=args.psl_learner,
                                    data_dir=data_dir,
                                    logger=logger,
                                    random_state=args.rs)

    else:
        model = base_estimator

    return model


def experiment(args, logger, out_dir):

    logger.info('\nDATA')
    in_dir = os.path.join(args.data_dir, args.dataset, 'fold_{}'.format(args.fold))

    # read in feature data
    logger.info('reading in data...')
    X_train = load_npz(os.path.join(in_dir, '{}_train.npz'.format(args.feature_type))).tocsr()
    X_val = load_npz(os.path.join(in_dir, '{}_val.npz'.format(args.feature_type))).tocsr()
    X_test = load_npz(os.path.join(in_dir, '{}_test.npz'.format(args.feature_type))).tocsr()

    # read in label data
    train_df = pd.read_csv(os.path.join(in_dir, 'y_train.csv'))
    val_df = pd.read_csv(os.path.join(in_dir, 'y_val.csv'))
    test_df = pd.read_csv(os.path.join(in_dir, 'y_test.csv'))

    # filter out transductive test indices
    if args.test_type == 'inductive':
        indices = np.load(os.path.join(in_dir, 'inductive_indices.npz'))

        val_df = val_df[val_df['com_id'].isin(indices['val'])]
        test_df = test_df[test_df['com_id'].isin(indices['test'])]

        X_val = X_val[val_df.index]
        X_test = X_test[test_df.index]

    # extract label data
    y_train = train_df['label'].to_numpy()
    y_val = val_df['label'].to_numpy()
    y_test = test_df['label'].to_numpy()

    # extract identifier data
    target_ids_train = train_df['com_id'].to_numpy()
    target_ids_val = val_df['com_id'].to_numpy()
    target_ids_test = test_df['com_id'].to_numpy()

    logger.info('\ntrain instances: X: {}, y: {}'.format(X_train.shape, y_train.shape))
    logger.info('val   instances: X: {}, y: {}'.format(X_val.shape, y_val.shape))
    logger.info('test  instances: X: {}, y: {}'.format(X_test.shape, y_test.shape))

    # setup models
    model = _get_model(args, data_dir=in_dir, logger=logger)

    # train and predict
    logger.info('\nTRAIN')
    start = time.time()
    if args.eggs:
        model = model.fit(X_train, y_train, target_ids_train, X_val, y_val, target_ids_val)
    else:
        model = model.fit(X_train, y_train)
    logger.info('total time: {:.3f}s'.format(time.time() - start))

    logger.info('\nPREDICT')
    if args.eggs:
        proba = model.predict_proba(X_test, target_ids_test)[:, 1]
    else:
        proba = model.predict_proba(X_test)[:, 1]
    auc, ap = util.performance(y_test, proba, logger=logger, name='model')
    logger.info('total time: {:.3f}s'.format(time.time() - start))

    # save results
    result = {'auc': auc, 'ap': ap}
    np.save(os.path.join(out_dir, 'result.npy'), result)


def main(args):

    # create output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           'fold_{}'.format(args.fold),
                           'rs_{}'.format(args.rs),
                           'base_{}'.format(args.base_estimator),
                           'feature_{}'.format(args.feature_type),
                           'test_{}'.format(args.test_type),
                           'sgl_{}'.format(args.sgl_method),
                           'stacks_{}'.format(args.sgl_stacks),
                           'pgm_{}'.format(args.pgm))
    if args.pgm == 'psl':
        out_dir = os.path.join(out_dir, 'psl_{}'.format(args.psl_learner))

    log_fp = os.path.join(out_dir, 'log.txt')
    os.makedirs(out_dir, exist_ok=True)

    # skip experiment if results already exist
    if args.append_results and os.path.exists(log_fp.replace('log.txt', 'result.npy')):
        return

    # create logger
    logger = util.get_logger(log_fp)
    logger.info(args)
    logger.info(datetime.now())

    # run experiment
    experiment(args, logger, out_dir)

    # remove logger
    util.remove_logger(logger)


if __name__ == '__main__':

    # read in commandline args
    parser = argparse.ArgumentParser(description='EGGS: Extended Group-based Graphical models for Spam.')

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/performance/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory.')
    parser.add_argument('--dataset', type=str, default='youtube', help='dataset.')

    # experiment settings
    parser.add_argument('--feature_type', type=str, default='full', help='limited or full.')
    parser.add_argument('--test_type', type=str, default='full', help='inductive or full.')
    parser.add_argument('--fold', type=int, default=0, help='dataset fold to use.')
    parser.add_argument('--base_estimator', type=str, default='lr', help='base estimator: lr or lgb.')
    parser.add_argument('--n_estimators', type=int, default=100, help='no. trees.')
    parser.add_argument('--max_depth', type=int, default=None, help='max depth of each tree.')

    # EGGS settings
    parser.add_argument('--eggs', action='store_true', default=False, help='use EGGS.')
    parser.add_argument('--relations', type=str, nargs='+', default=['user', 'text'], help='relations.')
    parser.add_argument('--sgl_method', type=str, default='None', help='training method.')
    parser.add_argument('--sgl_stacks', type=int, default=0, help='number of SGL stacks.')
    parser.add_argument('--pgm', type=str, default='None', help='joint inference model (MRF or PSL).')
    parser.add_argument('--psl_learner', type=str, default='mle', help='PSL weight learning.')

    # extra settings
    parser.add_argument('--rs', type=int, default=1, help='random state.')
    parser.add_argument('--verbose', type=int, default=0, help='verbosity level.')
    parser.add_argument('--append_results', action='store_true', default=False, help='skip finished results.')

    args = parser.parse_args()
    main(args)
