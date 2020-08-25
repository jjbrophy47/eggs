"""
Preprocesses the data.
"""
import os
import time
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

import util
from relations import create_text_relation
from relations import create_user_relation
from relations import create_link_relation
from relations import create_hashuser_relation
from induction import get_inductive_indices

MIN_POS = 2


def _compute_features(args, logger, df, cv=None):
    """
    SoundCloud.
    """
    features_df = df.copy()

    logger.info('adding graph features...')
    if args.dataset in ['soundcloud', 'twitter']:
        graph = ['pagerank', 'triangle_count', 'core_id', 'out_degree', 'in_degree']

    logger.info('building content features...')
    features_df['num_chs'] = df['text'].str.len()
    features_df['num_lnk'] = df['text'].str.count('http')

    if args.dataset == 'soundcloud':
        content = ['num_chs', 'num_lnk', 'polarity', 'subjectivity']

    elif args.dataset == 'twitter':
        features_df['num_hsh'] = df['text'].str.count('#')
        features_df['num_men'] = df['text'].str.count('@')
        features_df['num_rtw'] = df['text'].str.count('RT')
        features_df['num_uni'] = df['text'].str.count(r'(\\u\S\S\S\S)')
        content = ['num_chs', 'num_hsh', 'num_men', 'num_lnk', 'num_rtw',
                   'num_uni', 'polarity', 'subjectivity']

    elif args.dataset == 'youtube':
        features_df['wday'] = df['timestamp'].dt.dayofweek
        features_df['hour'] = df['timestamp'].dt.hour
        content = ['num_chs', 'wday', 'hour', 'polarity', 'subjectivity']

    logger.info('building sequential features...')
    features_df['usr_msg_cnt'] = df.groupby('user_id').cumcount()

    if args.dataset == 'soundcloud':
        features_df['has_lnk'] = df['text'].str.contains('http').astype(int)
        lnk_cnt = features_df.groupby('user_id')['has_lnk'].cumsum() - features_df['has_lnk']
        features_df['num_trk_msgs'] = df.groupby('track_id').cumcount()
        features_df['usr_lnk_rto'] = lnk_cnt.divide(features_df['usr_msg_cnt']).fillna(0)
        sequential = ['num_trk_msgs', 'usr_msg_cnt', 'usr_lnk_rto']

    elif args.dataset == 'twitter':
        features_df['has_lnk'] = df['text'].str.contains('http').astype(int)
        features_df['has_hsh'] = df['text'].str.contains('#').astype(int)
        features_df['has_men'] = df['text'].str.contains('@').astype(int)
        lnk_cnt = features_df.groupby('user_id')['has_lnk'].cumsum() - features_df['has_lnk']
        hsh_cnt = features_df.groupby('user_id')['has_hsh'].cumsum() - features_df['has_hsh']
        men_cnt = features_df.groupby('user_id')['has_men'].cumsum() - features_df['has_men']

        features_df['usr_lnk_rto'] = lnk_cnt.divide(features_df['usr_msg_cnt']).fillna(0)
        features_df['usr_hsh_rto'] = hsh_cnt.divide(features_df['usr_msg_cnt']).fillna(0)
        features_df['usr_men_rto'] = men_cnt.divide(features_df['usr_msg_cnt']).fillna(0)
        sequential = ['usr_msg_cnt', 'usr_lnk_rto', 'usr_hsh_rto', 'usr_men_rto']

    elif args.dataset == 'youtube':
        features_df['len'] = features_df['text'].str.len()
        features_df['usr_msg_max'] = features_df.groupby('user_id')['len'].cummax()
        features_df['usr_msg_min'] = features_df.groupby('user_id')['len'].cummin()
        features_df['usr_msg_mean'] = list(features_df.groupby('user_id')['len']
                                                      .expanding().mean().reset_index()
                                                      .sort_values('level_1')['len'])
        sequential = ['usr_msg_cnt', 'usr_msg_max', 'usr_msg_min', 'usr_msg_mean']

    # choose features based on dataset and either `limited' or `full` features
    ngrams = None

    limited = {'soundcloud': ['content', 'graph'],
               'twitter': ['content', 'graph', 'sequential'],
               'youtube': ['content', 'sequential']}

    full = {'soundcloud': ['content', 'graph', 'sequential', 'ngrams'],
            'twitter': ['content', 'graph', 'sequential', 'ngrams'],
            'youtube': ['content', 'sequential', 'ngrams']}

    feature_dict = {'limited': limited, 'full': full}

    features = []
    for feature_group in feature_dict[args.feature_type][args.dataset]:
        if feature_group == 'content':
            features += content

        if feature_group == 'graph':
            features += graph

        if feature_group == 'sequential':
            features += sequential

        if feature_group == 'ngrams':
            logger.info('building ngram features...')
            ngrams, cv = _ngrams(df, count_vectorizer=cv)

    # convert features to sparse numpy matrix
    features_df = features_df[features]
    X = csr_matrix(features_df.astype(float).values)
    if ngrams is not None:
        X = hstack([X, ngrams])

    logger.info(X.shape)

    return X, cv


def _ngrams(df, count_vectorizer=None, max_features=10000, ngram_range=(3, 3),
            stop_words='english', min_df=1, max_df=1.0, analyzer='char_wb',
            binary=True, vocabulary=None, dtype=np.int32):
    """
    Generate ngram features based on the text.
    """
    str_list = df['text'].tolist()

    if count_vectorizer is None:
        count_vectorizer = CountVectorizer(stop_words=stop_words, min_df=min_df,
                                           ngram_range=ngram_range, max_df=max_df,
                                           max_features=max_features, analyzer=analyzer,
                                           binary=binary, vocabulary=vocabulary,
                                           dtype=dtype)
        ngram_matrix = count_vectorizer.fit_transform(str_list)

    else:
        ngram_matrix = count_vectorizer.transform(str_list)

    id_matrix = ss.lil_matrix((len(df), 1))
    ngram_matrix = ss.hstack([id_matrix, ngram_matrix]).tocsr()

    return ngram_matrix, count_vectorizer


def _generate_relations(args, train_df, val_df, test_df):
    """
    Create relation files.
    SoundCloud: user, text, link.
    Twitter: user, text, hashuser.
    YouTube: user, text.
    """
    result = {}

    result['user'] = create_user_relation(train_df, val_df, test_df)
    result['text'] = create_text_relation(train_df, val_df, test_df)

    if args.dataset == 'soundcloud':
        result['link'] = create_link_relation(train_df, val_df, test_df)

    elif args.dataset == 'twitter':
        result['hashuser'] = create_hashuser_relation(train_df, val_df, test_df)

    return result


def make_dataset(args, df, fold, out_dir, logger):

    # split the data into train, val, and test
    n_val = int(len(df) * args.val_frac)
    n_test = int(len(df) * args.test_frac)

    val_df = df[len(df) - n_test - n_val: len(df) - n_test]
    test_df = df[len(df) - n_test:]
    train_df = df[:len(df) - n_test - n_val]

    print(train_df.shape, val_df.shape, val_df['label'].sum(), test_df.shape, test_df['label'].sum())

    # skip this fold, no positive samples
    if val_df['label'].sum() < MIN_POS or test_df['label'].sum() < MIN_POS:
        return -1

    # generate independent features
    logger.info('\nTRAIN')
    start = time.time()
    train, cv = _compute_features(args, logger, train_df)
    logger.info('total time: {:3f}s'.format(time.time() - start))

    logger.info('\nVAL')
    start = time.time()
    val, _ = _compute_features(args, logger, val_df, cv=cv)
    logger.info('total time: {:3f}s'.format(time.time() - start))

    logger.info('\nTEST')
    start = time.time()
    test, _ = _compute_features(args, logger, test_df, cv=cv)
    logger.info('total time: {:3f}s'.format(time.time() - start))

    logger.info('\n[train] pos labels: {}'.format(train_df['label'].sum()))
    logger.info('[val] pos labels: {}'.format(val_df['label'].sum()))
    logger.info('[test] pos labels: {}'.format(test_df['label'].sum()))

    # save engineered features
    logger.info('\nsaving engineered features...')
    os.makedirs(os.path.join(out_dir), exist_ok=True)
    save_npz(os.path.join(out_dir, '{}_train.npz'.format(args.feature_type)), train)
    save_npz(os.path.join(out_dir, '{}_val.npz'.format(args.feature_type)), val)
    save_npz(os.path.join(out_dir, '{}_test.npz'.format(args.feature_type)), test)

    # saving labels
    train_df[['com_id', 'label']].to_csv(os.path.join(out_dir, 'y_train.csv'), index=None)
    val_df[['com_id', 'label']].to_csv(os.path.join(out_dir, 'y_val.csv'), index=None)
    test_df[['com_id', 'label']].to_csv(os.path.join(out_dir, 'y_test.csv'), index=None)

    # generate relational features
    if args.relations:

        logger.info('\nRELATIONS')
        start = time.time()
        relations = _generate_relations(args, train_df, val_df, test_df)

        # save relation files
        logger.info('saving relations....')
        for relation, rf in relations.items():
            rf.to_csv(os.path.join(out_dir, 'relation_{}.csv'.format(relation)), index=None)

        # save inductive and inductive + transductive val / test instances
        logger.info('filtering out transductive indices...')
        val_inductive_indices = get_inductive_indices(relations, train_df, val_df)
        test_inductive_indices = get_inductive_indices(relations, train_df, test_df)

        new_val_df = val_df[val_df['com_id'].isin(val_inductive_indices)]
        new_test_df = test_df[test_df['com_id'].isin(test_inductive_indices)]

        # skip this fold, no positive samples
        if new_val_df['label'].sum() < MIN_POS or new_test_df['label'].sum() < MIN_POS:
            return -1

        logger.info('\n[val] {}, pos labels: {}'.format(new_val_df.shape, new_val_df['label'].sum()))
        logger.info('[test] {}, pos labels: {}'.format(new_test_df.shape, new_test_df['label'].sum()))

        logger.info('\nsaving inductive indices...')
        np.savez_compressed(os.path.join(out_dir, 'inductive_indices.npz'),
                            val=val_inductive_indices, test=test_inductive_indices)
        logger.info('total time: {:3f}s'.format(time.time() - start))


def main(args):

    # display settings
    pd.set_option('display.width', 181)
    pd.set_option('display.max_columns', 100)

    # setup input directory
    in_dir = os.path.join(args.data_dir, args.dataset, 'raw')

    # read in data
    df = pd.read_csv(os.path.join(in_dir, 'comments.csv'), nrows=args.nrows)
    df['text'] = df['text'].fillna('')
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # create logger
    log_dir = os.path.join(args.log_dir, args.dataset, args.feature_type)
    os.makedirs(log_dir, exist_ok=True)

    logger_fp = os.path.join(log_dir, 'log.txt')
    logger = util.get_logger(logger_fp)
    logger.info('\n{}'.format(args))

    # split data into folds
    logger.info('\nno. folds: {:,}'.format(args.n_folds))

    fold = 0
    for fold_df in np.array_split(df, args.n_folds):
        logger.info('\n\nFold {}: {:,}'.format(fold, len(fold_df)))

        # setup output directory
        out_dir = os.path.join(args.data_dir, args.dataset, 'fold_{}'.format(fold))
        result = make_dataset(args, fold_df, fold, out_dir, logger)

        if result == -1:
            exit('Not enough positive samples!')

        fold += 1

    logger.info('total no. valid folds: {}'.format(fold))
    util.remove_logger(logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='output/preprocess', help='logging directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='data directory.')
    parser.add_argument('--dataset', type=str, default='youtube', help='dataset.')

    parser.add_argument('--n_folds', type=int, default=10, help='number of train/val/test splits.')

    parser.add_argument('--val_frac', type=float, default=0.025, help='fraction of validation data.')
    parser.add_argument('--test_frac', type=float, default=0.285, help='fraction of test data.')

    parser.add_argument('--feature_type', type=str, default='limited', help='limited or full.')
    parser.add_argument('--relations', action='store_true', default=False, help='create relational features.')

    parser.add_argument('--nrows', type=int, default=None, help='number of rows to use.')

    args = parser.parse_args()
    main(args)
