"""
Organize the results into a single csv.
"""
import os
import argparse
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import sem
from tqdm import tqdm

import print_util


def _get_result(r, experiment_dir):
    """
    Obtain the results for this baseline method.
    """
    result = r.copy()

    fp = os.path.join(experiment_dir, 'result.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]

        result['auc'] = d['auc']
        result['ap'] = d['ap']

    return result


def process_results(df):
    """
    Compute performance differences and average among all folds.
    """
    dataset_list = []
    for dataset in args.dataset:
        temp_df = df[df['dataset'] == dataset]

        if len(temp_df) == 0:
            continue

        # compute performance difference from baseline
        fold_list = []
        for tup, gf in temp_df.groupby(['fold', 'feature_type', 'test_type', 'rs', 'base_estimator']):
            base_df = gf[gf['sgl_method'] == 'None']
            base_df = gf[gf['sgl_stacks'] == 0]
            base_df = gf[gf['pgm'] == 'None']

            base_auc = base_df['auc'].values[0]
            base_ap = base_df['ap'].values[0]

            new_gf = gf.copy()
            new_gf['auc_diff'] = new_gf['auc'] - base_auc
            new_gf['ap_diff'] = new_gf['ap'] - base_ap

            fold_list.append(new_gf)

        temp_df = pd.concat(fold_list)

        # average methods over all folds
        settings = ['rs', 'base_estimator', 'feature_type', 'test_type',
                    'sgl_method', 'sgl_stacks', 'pgm']

        results = []
        for tup, gf in temp_df.groupby(settings):
            result = {k: v for k, v in zip(settings, tup)}
            result['auc_mean'] = gf['auc'].mean()
            result['ap_mean'] = gf['ap'].mean()
            result['auc_std'] = gf['auc'].std()
            result['ap_std'] = gf['ap'].std()

            result['auc_diff_mean'] = gf['auc_diff'].mean()
            result['ap_diff_mean'] = gf['ap_diff'].mean()
            result['auc_diff_std'] = sem(gf['auc_diff'])
            result['ap_diff_std'] = sem(gf['ap_diff'])
            result['dataset'] = dataset
            results.append(result)

        dataset_df = pd.DataFrame(results)
        dataset_list.append(dataset_df)

    result_df = pd.concat(dataset_list)
    return result_df


def create_csv(args, logger):

    experiment_settings = list(product(*[args.dataset, args.fold, args.rs, args.base_estimator,
                                         args.feature_type, args.test_type,
                                         args.sgl_method, args.sgl_stacks, args.pgm]))

    results = []
    for experiment_tuple in tqdm(experiment_settings):
        dataset, fold, rs, base_estimator, feature_type, test_type, sgl_method, sgl_stacks, pgm = experiment_tuple

        result = {'dataset': dataset, 'fold': fold, 'rs': rs, 'base_estimator': base_estimator,
                  'feature_type': feature_type, 'test_type': test_type,
                  'sgl_method': sgl_method, 'sgl_stacks': sgl_stacks, 'pgm': pgm}

        experiment_dir = os.path.join(args.in_dir,
                                      dataset,
                                      'fold_{}'.format(fold),
                                      'rs_{}'.format(rs),
                                      'base_{}'.format(base_estimator),
                                      'feature_{}'.format(feature_type),
                                      'test_{}'.format(test_type),
                                      'sgl_{}'.format(sgl_method),
                                      'stacks_{}'.format(sgl_stacks),
                                      'pgm_{}'.format(pgm))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        cedar_result = _get_result(result, experiment_dir)

        if cedar_result:
            results.append(cedar_result)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)

    df = pd.DataFrame(results)
    logger.info('\nRaw:\n{}'.format(df))

    res_df = process_results(df)
    logger.info('\nProcessed:\n{}'.format(res_df))

    fp = os.path.join(args.out_dir, 'results.csv')
    res_df.to_csv(fp, index=None)


def main(args):

    # create logger
    os.makedirs(args.out_dir, exist_ok=True)
    logger = print_util.get_logger(os.path.join(args.out_dir, 'log.txt'))
    logger.info(args)
    logger.info(datetime.now())

    create_csv(args, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # I/O settings
    parser.add_argument('--in_dir', type=str, default='output/performance/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/csv/performance/', help='output directory.')

    # experiment settings
    parser.add_argument('--dataset', type=str, nargs='+', default=['youtube', 'twitter', 'soundcloud'], help='data.')
    parser.add_argument('--fold', type=str, nargs='+', default=list(range(185)), help='folds.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1], help='random state.')
    parser.add_argument('--base_estimator', type=str, nargs='+', default=['lr', 'lgb'], help='baseline model.')
    parser.add_argument('--feature_type', type=str, nargs='+', default=['limited', 'full'], help='features to use.')
    parser.add_argument('--test_type', type=str, nargs='+', default=['inductive', 'full'], help='test instances.')

    # EGGS settings
    parser.add_argument('--sgl_method', type=str, nargs='+', default=['None', 'holdout', 'cv'], help='train type.')
    parser.add_argument('--sgl_stacks', type=int, nargs='+', default=[0, 1, 2], help='number of SGL stacks.')
    parser.add_argument('--pgm', type=str, nargs='+', default=['None', 'psl', 'mrf'], help='joint model.')

    args = parser.parse_args()
    main(args)
