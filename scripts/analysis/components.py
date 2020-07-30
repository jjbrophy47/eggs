"""
Module to find subnetwork of a data points based on its relationships.
"""
import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx

import util


def main(args):
    """
    Find and report statistics on the connected components.
    """
    # create output directory
    out_dir = os.path.join(args.out_dir,
                           args.dataset,
                           'feature_{}'.format(args.feature_type),
                           'test_{}'.format(args.test_type))

    log_fp = os.path.join(out_dir, 'log.txt')
    os.makedirs(out_dir, exist_ok=True)

    # create logger
    logger = util.get_logger(log_fp)
    logger.info(args)
    logger.info(datetime.now())

    logger.info('\nDATA')
    in_dir = os.path.join(args.data_dir, args.dataset)

    # read in label data
    val_df = pd.read_csv(os.path.join(in_dir, 'y_val.csv'))
    test_df = pd.read_csv(os.path.join(in_dir, 'y_test.csv'))

    # filter out transductive test indices
    if args.test_type == 'inductive':
        indices = np.load(os.path.join(in_dir, 'inductive_indices.npz'))

        val_df = val_df[val_df['com_id'].isin(indices['val'])]
        test_df = test_df[test_df['com_id'].isin(indices['test'])]

    # extract label data
    y_test = test_df['label'].to_numpy()

    # extract identifier data
    target_ids_test = test_df['com_id'].to_numpy()

    # create msg -> label mapping
    msg_label_map = {target_id: label for target_id, label in zip(target_ids_test, y_test)}

    target_col = 'com_id'
    g = nx.Graph()

    for relation in args.relations:
        relation_col = '{}_id'.format(relation)
        relation_df = pd.read_csv(os.path.join(in_dir, 'relation_{}.csv'.format(relation)))
        relation_df[relation_col] = relation_df[relation_col].astype(int)

        # filter messages not in this test set
        relation_df = relation_df[relation_df['com_id'].isin(target_ids_test)]

        msg_relation_list = list(zip(relation_df[target_col], relation_df[relation_col]))
        for target_id, relation_id in msg_relation_list:
            g.add_edge('{}_{}'.format(target_col, target_id), '{}_{}'.format(relation_col, relation_id))

    # compute connected components
    components = list(nx.connected_components(g))

    # aggregate statistics about each component
    results = []
    for component in components:

        if len(component) <= 2:
            continue

        result = {}
        msg_nodes = {x for x in component if target_col in str(x)}
        hub_nodes = {x for x in component if target_col not in str(x)}

        # count number of spam
        n_spam = 0
        for msg_str in msg_nodes:
            msg_id = int(msg_str.split('_')[-1])
            n_spam += msg_label_map[msg_id]

        # count number of edges and number of relations
        relations = set()
        edges = 0

        for hub_node in hub_nodes:
            relations.add(hub_node.split('_')[0])
            edges += g.degree(hub_node)

        result['num_nodes'] = len(msg_nodes) + len(hub_nodes)
        result['num_msg_nodes'] = len(msg_nodes)
        result['num_hub_nodes'] = len(hub_nodes)
        result['num_spam_msg_nodes'] = n_spam
        result['num_relations'] = len(relations)
        result['num_edges'] = edges
        results.append(result)

    df = pd.DataFrame(results)
    logger.info(df)

    # global statistics
    num_components = len(df)
    num_edges = df['num_edges'].sum()
    logger.info('total no. components: {}, no. edges: {}'.format(num_components, num_edges))

    # compute aggregate statistics
    for n_spam in range(df['num_spam_msg_nodes'].max()):

        temp = df[df['num_spam_msg_nodes'] == n_spam].copy()
        if len(temp) == 0:
            continue

        if n_spam == 0:
            temp['spam_fraction'] = 0
        else:
            temp['spam_fraction'] = temp['num_spam_msg_nodes'] / temp['num_nodes']

        num_nodes_avg = temp['num_nodes'].mean()
        spam_fraction_avg = temp['spam_fraction'].mean()

        logger.info('\n{} spam messages ({})'.format(n_spam, len(temp)))
        logger.info('  component size (average): {:.3f}'.format(num_nodes_avg))
        logger.info('  spam fraction (average): {:.3f}'.format(spam_fraction_avg))

    # remove logger
    util.remove_logger(logger)


if __name__ == '__main__':

    # read in commandline args
    parser = argparse.ArgumentParser(description='EGGS: Extended Group-based Graphical models for Spam.')

    # I/O settings
    parser.add_argument('--out_dir', type=str, default='output/analysis/', help='output directory.')
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory.')
    parser.add_argument('--dataset', type=str, default='youtube', help='dataset.')

    # experiment settings
    parser.add_argument('--feature_type', type=str, default='full', help='limited or full.')
    parser.add_argument('--test_type', type=str, default='full', help='inductive or full.')

    # EGGS settings
    parser.add_argument('--relations', type=str, nargs='+', default=['user', 'text'], help='relations.')

    args = parser.parse_args()
    main(args)
