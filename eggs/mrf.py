"""
This class handles loopy belief propagation using Libra.
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.utils.validation import check_is_fitted

from . import connections
from . import util


MAX_CLUSTER_SIZE = 40000


class MRF:
    """
    Class that performs loopy belief propagation using Libra.
    """

    def __init__(self,
                 relations,
                 epsilon=[0.1, 0.2, 0.3, 0.4, 0.5],
                 scoring='auc',
                 working_dir='.temp',
                 data_dir='data',
                 logger=None):
        """
        Initialization of the MRF model.

        Parameters
        ----------
        relations : list (default=None)
            Relations to use for relational modeling.
        epsilon : list (default=[0.1, 0.2, 0.3, 0.4])
            Epsilon values to try for each relation during training.
        scoring : str (default='auc')
            Method of scoring to use for model selection during training.
        working_dir : str (default='.temp')
            Temporary directory to store intermediate files.
        data_dir : str (default='data')
            Directory with relational data files.
        logger : object (default=None)
            Save output.
        """
        self.relations = relations
        self.epsilon = epsilon
        self.scoring = scoring
        self.working_dir = working_dir
        self.data_dir = data_dir
        self.logger = logger

    # public
    def fit(self, y, y_hat, target_ids):
        """
        Train an MRF model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        # set metric
        if self.scoring == 'auc':
            self.scoring_ = roc_auc_score

        elif self.scoring == 'ap':
            self.scoring_ = average_precision_score

        else:
            raise ValueError('scoring {} unknown!'.format(self.scoring))

        # test each relation individually
        result = util.get_relational_entities(y_hat=y_hat,
                                              target_ids=target_ids,
                                              relations=self.relations,
                                              data_dir=self.data_dir,
                                              logger=self.logger)
        target_priors, relations_dict, target_col = result

        self.relation_epsilons_ = {}
        for relation_id, relation_list in relations_dict.items():
            relation = relation_id.split('_')[0]

            if self.logger:
                self.logger.info('\n[MRF] tuning relation: {}'.format(relation))

            start = time.time()
            relation_dict = {relation_id: relation_list}

            # test different epsilon values for this relation
            scores = []
            for epsilon in self.epsilon:

                # run inference over subgraph clusters
                y_score = self._group_inference(target_priors, relation_dict, epsilon, target_col)

                metric_score = self.scoring_(y, y_score[:, 1])
                scores.append((metric_score, epsilon))

                if self.logger:
                    s = '\n[MRF]: epsilon={}, {}={:.3f}, time={:.3f}s'
                    self.logger.info(s.format(epsilon, self.scoring, metric_score, time.time() - start))

            self.relation_epsilons_[relation_id] = sorted(scores, reverse=True)[0][1]

        if self.logger:
            self.logger.info('\ntuned hyperparameters: {}'.format(self.relation_epsilons_))

        return self

    def inference(self, y_hat, target_ids):
        """
        Joint inference using PSL.
            y_hat: priors for target nodes.
            target_col: list of target_ids.
        """
        check_is_fitted(self, 'relation_epsilons_')

        target_priors, relations_dict, target_col = util.get_relational_entities(y_hat=y_hat,
                                                                                 target_ids=target_ids,
                                                                                 relations=self.relations,
                                                                                 data_dir=self.data_dir,
                                                                                 logger=self.logger)

        y_score = self._group_inference(target_priors, relations_dict, target_col=target_col)

        return y_score

    # private
    def _group_inference(self, target_priors, relations_dict, epsilon=None, target_col='com_id',
                         max_size=7500, max_edges=40000):
        """
        Run inference over clusters of connected components to reduce
        memory and runtime.
        """
        result_df = pd.DataFrame(target_priors, columns=[target_col, 'ind_yhat'])

        clusters = connections.create_clusters(target_priors, relations_dict,
                                               max_size=MAX_CLUSTER_SIZE)

        # Run inference over each cluster
        results = []
        for i, (msg_nodes, hub_nodes, relations, n_edges) in enumerate(clusters):
            start = time.time()

            # filter target IDs
            cluster_target_ids = [int(x.split('-')[1]) for x in msg_nodes]
            temp_df = result_df[result_df[target_col].isin(cluster_target_ids)]
            cluster_target_priors = list(zip(temp_df[target_col], temp_df['ind_yhat']))

            # create libra files for this cluster
            targets_dict, relation_dict_list = self._create_mrf(target_priors=cluster_target_priors,
                                                                relations_dict=relations_dict,
                                                                target_col=target_col,
                                                                epsilon=epsilon)

            # run inference for this cluster and save the results
            y_score = self._libra_inference(targets_dict, relation_dict_list)[:, 1]
            yhat_df = pd.DataFrame(zip(targets_dict.keys(), y_score), columns=[target_col, 'pgm_yhat'])
            results.append(yhat_df)

            if self.logger:
                s = '[CLUSTER {} / {}] msgs: {}, hubs: {}, edges: {}, time: {:.3f}s'
                self.logger.info(s.format(i, len(clusters), len(msg_nodes),
                                 len(hub_nodes), n_edges, time.time() - start))

        # put updated scores in order of target IDs
        yhat_df = pd.concat(results)
        result_df = result_df.merge(yhat_df, on=target_col, how='left')

        # fill independent target ID nodes with independent predictions
        result_df['pgm_yhat'] = result_df['pgm_yhat'].fillna(result_df['ind_yhat'])

        # put scores into sklearn format
        result_df['pgm_yhat_neg'] = 1 - result_df['pgm_yhat']
        y_score = np.hstack([result_df['pgm_yhat_neg'].values.reshape(-1, 1),
                             result_df['pgm_yhat'].values.reshape(-1, 1)])

        return y_score

    def _libra_inference(self, targets_dict, relation_dicts):
        """
        Loopy belief propagation using Libra.
        """
        in_fp = os.path.join(self.working_dir, 'model.mn')
        out_fp = os.path.join(self.working_dir, 'marginals.txt')

        cmd = 'libra bp -m {} -mo {}'.format(in_fp, out_fp)
        os.system(cmd)

        return self._get_marginals(targets_dict, relation_dicts)

    def _get_marginals(self, targets_dict, relation_dict_list):
        """
        Create target predicates.
        """
        posteriors = []

        # read marginals file
        with open(os.path.join(self.working_dir, 'marginals.txt'), 'r') as f:

            for i, line in enumerate(f.readlines()):
                for target_id, target_dict in targets_dict.items():

                    # makes sure targets are in the same order
                    if target_dict['ndx'] == i:
                        pred = [float(x) for x in line.split(' ')]
                        posteriors.append(pred)

        # assemble y_score
        y_score = np.array(posteriors)
        return y_score

    def _create_mrf(self, target_priors, relations_dict,
                    epsilon=None, target_col='com_id'):
        """
        Generates predicate files for Libra.
            target_priors: list of (target_id, prior) tuples.
            relations_dict: key=relation_id, value=list of (hub_id, target_id) tuples.
        """

        # create target nodes
        targets_dict = {}
        for i, (target_id, prior) in enumerate(target_priors):
            targets_dict[target_id] = {'ndx': i, 'prior': prior, 'card': 2}

        target_ids = [target_id for target_id, prior in target_priors]
        ndx = len(target_ids)

        # create relational hub nodes
        relation_dict_list = []

        # build a dict for each relation type
        for relation_id, relation_list in relations_dict.items():
            hubs_dict = {}

            # save connections from each hub node for this relation type
            df = pd.DataFrame(relation_list, columns=[relation_id, target_col])

            for hub_id, hub_df in df.groupby(relation_id):
                connection_ids = list(set(target_ids).intersection(set(list(hub_df[target_col]))))
                hubs_dict[hub_id] = {'ndx': ndx, relation_id: connection_ids}
                ndx += 1

            relation_dict_list.append((hubs_dict, relation_id))

        self._write_model_file(targets_dict=targets_dict,
                               relation_dict_list=relation_dict_list,
                               num_nodes=ndx,
                               epsilon=epsilon)

        return targets_dict, relation_dict_list

    def _write_model_file(self, targets_dict, relation_dict_list, num_nodes, epsilon=None):
        """
        Generates predicate files for Libra.
            targets_dict: key=target_id, value=dict(ndx: v1, prior: v2, card: v3)
            relations_dict: list of (hubs_dict, relation_type) tuples.
                            hubs_dict: key=relation_id, value=dict(ndx: v1, relation_id: list of target_ids)
        """

        # create temporary directory
        os.makedirs(self.working_dir, exist_ok=True)

        # write MRF file
        with open(os.path.join(self.working_dir, 'model.mn'), 'w') as f:

            # first line: comma separated cardinality for each node
            line = ''
            for i in range(num_nodes):
                line += '2\n' if i == num_nodes - 1 else '2,'
            f.write(line)

            # start network
            f.write('MN {\n')

            # write single node factors
            for i, (target_id, target_dict) in enumerate(targets_dict.items()):
                assert i == target_dict['ndx']

                prior = target_dict['prior']
                ndx = target_dict['ndx']

                factor = '{:.5f} +v{}_1\n'
                factor += '{:.5f} +v{}_0\n'
                f.write(factor.format(prior, ndx, 1.0 - prior, ndx))

            # retrieve each relation
            for hubs_dict, relation_id in relation_dict_list:

                # obtain epsilon value
                if epsilon is None:
                    assert self.relation_epsilons_ is not None
                    relation_epsilon = self.relation_epsilons_[relation_id]

                else:
                    relation_epsilon = epsilon

                # retrieve each hub_id
                for hub_id, hub_dict in hubs_dict.items():

                    hub_ndx = hub_dict['ndx']
                    target_ids = hub_dict[relation_id]

                    # write each hub target_id pair
                    for target_id in target_ids:
                        target_dict = targets_dict[target_id]
                        target_ndx = target_dict['ndx']

                        factor = '{:.5f} +v{}_0 +v{}_0\n'
                        factor += '{:.5f} +v{}_0 +v{}_1\n'
                        factor += '{:.5f} +v{}_1 +v{}_0\n'
                        factor += '{:.5f} +v{}_1 +v{}_1\n'

                        values = (1.0 - relation_epsilon, target_ndx, hub_ndx)
                        values += (relation_epsilon, target_ndx, hub_ndx)
                        values += (relation_epsilon, target_ndx, hub_ndx)
                        values += (1.0 - relation_epsilon, target_ndx, hub_ndx)

                        f.write(factor.format(*values))

            f.write('}\n')
