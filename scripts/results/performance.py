"""
Organize the deletion results into a single csv.
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


def get_avg_retrain_depth(d):
    """
    Return the average depth of retrains, if any.
    """
    avg_retrain_depth = -1

    total_sum = 0
    total_retrains = 0

    if 2 in d:
        for depth, num in d[2].items():
            total_sum += depth * num
            total_retrains += num

        if total_retrains > 0:
            avg_retrain_depth = total_sum / total_retrains

    return total_retrains, avg_retrain_depth


def get_baseline_result(method, r, baseline_dir):
    """
    Obtain the results for this baseline method.
    """
    result = r.copy()

    if method == 'naive':
        fp = os.path.join(baseline_dir, 'naive.npy')
        result['lmbda'] = -1
    elif method == 'exact':
        fp = os.path.join(baseline_dir, 'exact.npy')
        result['lmbda'] = -1
    else:
        raise ValueError('unknown method {}'.format(method))

    if not os.path.exists(fp):
        result = None, None

    else:
        d = np.load(fp, allow_pickle=True)[()]

        result['topd'] = -1
        result['min_support'] = -1
        result['epsilon'] = 0
        result['method'] = method
        result['train_time'] = d['train_time']

        if method == 'naive':
            result['amortized'] = np.mean(d['time'])
            result['amortized_worst_case'] = -1
            result['speedup_vs_naive'] = -1
            result['auc'] = -1
            result['acc'] = -1
            result['bacc'] = -1
            result['ap'] = -1
            result['auc_diff_avg'] = result['auc_diff_std'] = -1
            result['acc_diff_avg'] = result['acc_diff_std'] = -1
            result['bacc_diff_avg'] = result['bacc_diff_std'] = -1
            result['ap_diff_avg'] = result['ap_diff_std'] = -1
            result['num_retrains'] = result['avg_retrain_depth'] = -1
            result['percent_complete'] = 1.0
            result['n_nodes_avg'] = -1
            result['n_exact_avg'] = -1
            result['n_semi_avg'] = -1

        elif method == 'exact':
            result['amortized'] = d['amortized']
            result['amortized_worst_case'] = d['amortized_worst_case']
            result['speedup_vs_naive'] = -1
            result['auc'] = d['auc'][0]
            result['acc'] = d['acc'][0]
            result['bacc'] = d['bacc'][0]
            if 'ap' in d and d['ap'].size > 1:
                result['ap'] = d['ap'][0]
                result['ap_diff_avg'] = result['ap_diff_std'] = 0
            else:
                result['ap'] = -1
                result['ap_diff_avg'] = 0
            result['auc_diff_avg'] = result['auc_diff_std'] = 0
            result['acc_diff_avg'] = result['acc_diff_std'] = 0
            result['bacc_diff_avg'] = result['bacc_diff_std'] = 0
            result['num_retrains'], result['avg_retrain_depth'] = get_avg_retrain_depth(d['retrains'])
            result['percent_complete'] = d['percent_complete']
            result['n_nodes_avg'] = -1
            result['n_exact_avg'] = -1
            result['n_semi_avg'] = -1

        result = result, d

    return result


def update_baseline_results(naive_result, exact_result, exact_raw):
    """
    Updates results of methods based on results of other methods.
    """
    if naive_result and exact_result:
        exact_result['speedup_vs_naive'] = naive_result['amortized'] / exact_result['amortized']


def get_cedar_result(r, cedar_dir, naive_result, exact_raw):
    """
    Obtain the results for this baseline method.
    """
    result = r.copy()

    fp = os.path.join(cedar_dir, 'cedar.npy')

    if not os.path.exists(fp):
        result = None

    else:
        d = np.load(fp, allow_pickle=True)[()]

        result['method'] = 'cedar'
        result['train_time'] = d['train_time']
        result['amortized'] = d['amortized']
        result['amortized_worst_case'] = d['amortized_worst_case']
        result['num_retrains'], result['avg_retrain_depth'] = get_avg_retrain_depth(d['retrains'])
        result['percent_complete'] = d['percent_complete']
        result['auc'] = d['auc'][0]
        result['acc'] = d['acc'][0]
        result['bacc'] = d['bacc'][0]
        if 'ap' in d and d['ap'].size > 1:
            result['ap'] = d['ap'][0]
        else:
            result['ap'] = -1
        if 'n_nodes_avg' in d:
            result['n_nodes_avg'] = d['n_nodes_avg']
            result['n_exact_avg'] = d['n_exact_avg']
            result['n_semi_avg'] = d['n_semi_avg']

        if naive_result:
            naive_result['amortized']
            result['speedup_vs_naive'] = naive_result['amortized'] / result['amortized']

        if exact_raw:
            n = min(len(exact_raw['auc']), len(d['auc']))
            auc_diff = exact_raw['auc'][:n] - d['auc'][:n]
            acc_diff = exact_raw['acc'][:n] - d['acc'][:n]
            bacc_diff = exact_raw['bacc'][:n] - d['bacc'][:n]
            if 'ap' in d and d['ap'].size > 1:
                ap_diff = exact_raw['ap'][:n] - d['ap'][:n]

            result['auc_diff_avg'] = np.mean(auc_diff)
            result['auc_diff_std'] = sem(auc_diff)
            result['acc_diff_avg'] = np.mean(acc_diff)
            result['acc_diff_std'] = sem(acc_diff)

            result['bacc_diff_avg'] = np.mean(bacc_diff)
            result['bacc_diff_std'] = sem(bacc_diff)
            if 'ap' in d and d['ap'].size > 1:
                result['ap_diff_avg'] = np.mean(ap_diff)
                result['ap_diff_std'] = sem(ap_diff)
            else:
                result['ap_diff_avg'] = 0
                result['ap_diff_std'] = 0

    return result


def create_csv(args, logger):

    experiment_settings = list(product(*[args.dataset, args.model_type, args.criterion, args.adversary, args.rs,
                                         args.n_estimators, args.max_depth]))

    additional_settings = list(product(*[args.topd, args.min_support, args.epsilon, args.lmbda]))

    results = []
    for dataset, model_type, criterion, adversary, rs, n_estimators, max_depth in tqdm(experiment_settings):

        result = {'dataset': dataset, 'model_type': model_type, 'criterion': criterion,
                  'adversary': adversary, 'rs': rs, 'n_estimators': n_estimators,
                  'max_depth': max_depth}

        experiment_dir = os.path.join(args.in_dir, dataset, model_type, criterion, adversary,
                                      'rs_{}'.format(rs),
                                      'trees_{}'.format(n_estimators),
                                      'depth_{}'.format(max_depth))

        # skip empty experiments
        if not os.path.exists(experiment_dir):
            continue

        # contains naive and exact unlearners
        naive_result, naive_raw = get_baseline_result('naive', result, experiment_dir)
        exact_result, exact_raw = get_baseline_result('exact', result, experiment_dir)

        update_baseline_results(naive_result, exact_result, exact_raw)

        # baseline results are complete, save them
        if naive_result:
            results.append(naive_result)

        if exact_result:
            results.append(exact_result)

        # load in CeDAR results
        for topd, min_support, epsilon, lmbda in additional_settings:

            result['topd'] = topd
            result['min_support'] = min_support
            result['epsilon'] = epsilon
            result['lmbda'] = lmbda

            cedar_dir = os.path.join(experiment_dir,
                                     'topd_{}'.format(topd),
                                     'support_{}'.format(min_support),
                                     'epsilon_{}'.format(epsilon),
                                     'lmbda_{}'.format(lmbda))

            cedar_result = get_cedar_result(result, cedar_dir, naive_result, exact_raw)

            if cedar_result:
                results.append(cedar_result)

    df = pd.DataFrame(results)
    fp = os.path.join(args.out_dir, 'results.csv')
    df.to_csv(fp, index=None)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 180)
    print(df)


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
    parser.add_argument('--in_dir', type=str, default='output/deletion/', help='input directory.')
    parser.add_argument('--out_dir', type=str, default='output/csv/deletion/', help='output directory.')

    # experiment settings
    parser.add_argument('--model_type', type=str, nargs='+', default=['forest'], help='stump, tree, or forest.')
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=['surgical', 'adult', 'bank_marketing', 'flight_delays', 'diabetes',
                                 'olympics', 'census', 'credit_card', 'synthetic', 'higgs'], help='dataset.')
    parser.add_argument('--adversary', type=str, nargs='+', default=['random', 'root'], help='adversary.')
    parser.add_argument('--criterion', type=str, nargs='+', default=['gini', 'entropy'], help='criterion.')
    parser.add_argument('--rs', type=int, nargs='+', default=[1], help='random state.')

    # hyperparameter settings
    parser.add_argument('--n_estimators', type=int, nargs='+', default=[10, 50, 100, 250, 500], help='no. trees.')
    parser.add_argument('--max_depth', type=int, nargs='+', default=[1, 3, 5, 10, 20], help='max depth.')
    parser.add_argument('--topd', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='top d.')
    parser.add_argument('--min_support', type=int, nargs='+', default=[2500], help='minimum support.')
    parser.add_argument('--epsilon', type=float, nargs='+', default=[0.01, 0.1, 1.0], help='epsilon.')
    parser.add_argument('--lmbda', type=float, nargs='+', default=[1e-8, 1e-4, 1e-1], help='lmbda.')

    args = parser.parse_args()
    main(args)
