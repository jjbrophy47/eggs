"""
This class handles inference using PSL.
"""
import os
import time

import numpy as np
import pandas as pd
from pslpython.model import Model
from pslpython.partition import Partition
from pslpython.predicate import Predicate
from pslpython.rule import Rule

from . import util

JVM_OPTIONS = ['-Xmx60g']


class PSL:
    """
    Class that performs inference using PSL.
    """

    def __init__(self,
                 relations,
                 data_dir='data',
                 working_dir='.temp',
                 learner='mle',
                 logger=None):
        """
        Initialization of joint inference class.

        Parameters
        ----------
        relations : list
            Relations to use for relational modeling.
        data_dir : str (default='data')
            Temporary directory to store intermdiate files.
        working_dir : str (default='.temp')
            Temporary directory to store intermdiate files.
        logger : object (default=None)
            Logger for logging output.
        learner : str (default='mle')
            Weight learning optimizer, 'mle': Maximum Likelihood,
            'gpp': Gaussian Process Prior (uses the Ranking Estimator).
        """
        self.relations = relations
        self.data_dir = data_dir
        self.working_dir = working_dir
        self.learner = learner
        self.logger = logger

    # public
    def fit(self, y, y_hat, target_ids):
        """
        Train a PSL model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """

        # create model
        self.model_ = Model('spam')
        self._add_predicates(self.model_, self.relations)
        self._add_rules(self.model_, self.relations)

        # add data
        result = util.get_relational_entities(y_hat=y_hat,
                                              target_ids=target_ids,
                                              relations=self.relations,
                                              data_dir=self.data_dir,
                                              logger=self.logger)
        target_priors, relations_dict, target_col = result

        self._add_data(self.model_, target_col=target_col, target_priors=target_priors,
                       relations_dict=relations_dict, y=y)

        # start timing
        start = time.time()
        if self.logger:
            self.logger.info('[PSL] training...')

        # learning settings
        optimizer = ''
        additional_cli_options = []

        if self.learner == 'gpp':
            optimizer = 'GaussianProcessPrior'
            additional_cli_options = ['-e', 'RankingEvaluator']

        # train
        self.model_.learn(method=optimizer,
                          additional_cli_optons=additional_cli_options,
                          temp_dir=self.working_dir,
                          jvm_options=JVM_OPTIONS,
                          logger=self.logger)

        if self.logger:
            self.logger.info('[PSL] time: {:.3f}s'.format(time.time() - start))
            self.logger.info('[PSL] learned rules:')
            for rule in self.model_.get_rules():
                self.logger.info('   ' + str(rule))

        return self

    def inference(self, y_hat, target_ids):
        """
        Joint inference using PSL.
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).

        Returns predictions with shape=(len(y_hat),).
        """
        assert self.model_

        # add data
        result = util.get_relational_entities(y_hat=y_hat,
                                              target_ids=target_ids,
                                              relations=self.relations,
                                              data_dir=self.data_dir,
                                              logger=self.logger)
        target_priors, relations_dict, target_col = result

        self._add_data(self.model_, target_col=target_col, target_priors=target_priors,
                       relations_dict=relations_dict)

        # infer
        results = self.model_.infer(temp_dir=self.working_dir,
                                    logger=self.logger,
                                    jvm_options=JVM_OPTIONS)

        # exact udpdated scores
        y_hat_hat_df = results[self.model_.get_predicate('spam_msg')]
        y_hat_hat_df.columns = [target_col, 'y_hat_hat']
        target_df = pd.DataFrame(list(zip(target_ids, y_hat)), columns=[target_col, 'y_hat'])
        target_df = target_df.merge(y_hat_hat_df, on=target_col, how='left')

        # reshape scores
        y_hat_hat = target_df['y_hat_hat'].to_numpy()
        y_score = np.hstack([1 - y_hat_hat.reshape(-1, 1), y_hat_hat.reshape(-1, 1)])
        assert len(y_score) == len(target_ids)

        return y_score

    # private
    def _add_predicates(self, model, relations):
        """
        Add predicates based on the given relations.
        """
        model.add_predicate(Predicate('spam_msg', closed=False, size=1))
        model.add_predicate(Predicate('prior_msg', closed=True, size=1))

        for relation in relations:
            model.add_predicate(Predicate('spam_{}'.format(relation), closed=False, size=1))
            model.add_predicate(Predicate('has_{}'.format(relation), closed=True, size=2))

    def _add_rules(self, model, relations):
        """
        Add rules connecting entities together.
        """
        model.add_rule(Rule('1.0: ~spam_msg(M) ^2'))
        model.add_rule(Rule('1.0: prior_msg(M) -> spam_msg(M) ^2'))

        for relation in relations:
            var = relation[0].upper()
            r1 = '1.0: has_{}(M, {}) & spam_{}({}) -> spam_msg(M) ^2'
            r2 = '1.0: has_{}(M, {}) & spam_msg(M) -> spam_{}({}) ^2'

            model.add_rule(Rule('1.0: ~spam_{}({}) ^2'.format(relation, var)))
            model.add_rule(Rule(r1.format(relation, var, relation, var)))
            model.add_rule(Rule(r2.format(relation, var, relation, var)))

    def _add_data(self, model, target_col, target_priors, relations_dict, y=None, sep='\t'):
        """
        Add predicate data.
            Observations: observed values for closed predicates and open predicates.
            Targets: Predicate targets we want to infer values for.
            Truth: Labels of some target predicates for training.
        """

        # clear any data
        for predicate in model.get_predicates().values():
            predicate.clear_data()

        # organize targets
        target_df = pd.DataFrame(target_priors, columns=[target_col, 'y_hat'])

        # filepaths
        prior_msg_fp = os.path.join(self.working_dir, 'spam_msg.tsv')
        spam_msg_nolabel_fp = os.path.join(self.working_dir, 'spam_msg_nolabel.tsv')

        # create data files
        target_df.to_csv(prior_msg_fp, columns=[target_col, 'y_hat'], sep=sep, header=None, index=None)
        target_df.to_csv(spam_msg_nolabel_fp, columns=[target_col], sep=sep, header=None, index=None)

        # add data to the model
        model.get_predicate('prior_msg').add_data_file(Partition.OBSERVATIONS, prior_msg_fp)
        model.get_predicate('spam_msg').add_data_file(Partition.TARGETS, spam_msg_nolabel_fp)

        # add relational data to the model
        for relation_id, relation_list in relations_dict.items():
            relation = relation_id.split('_')[0]

            # filepaths
            relation_fp = os.path.join(self.working_dir, 'has_{}.tsv'.format(relation))
            hub_fp = os.path.join(self.working_dir, 'spam_{}.tsv'.format(relation))

            # organize data
            relation_df = pd.DataFrame(relation_list, columns=[relation_id, target_col], dtype=int)
            hub_df = relation_df.drop_duplicates(subset=[relation_id])

            # create files
            relation_df.to_csv(relation_fp, columns=[target_col, relation_id], sep=sep, header=None, index=None)
            hub_df.to_csv(hub_fp, columns=[relation_id], sep=sep, header=None, index=None)

            # add data
            model.get_predicate('has_{}'.format(relation)).add_data_file(Partition.OBSERVATIONS, relation_fp)
            model.get_predicate('spam_{}'.format(relation)).add_data_file(Partition.TARGETS, hub_fp)

        # add labeled data for weight learning
        if y is not None:
            spam_msg_label_fp = os.path.join(self.working_dir, 'spam_msg_label.tsv')
            label_df = pd.DataFrame(list(zip(target_df[target_col], y)), columns=[target_col, 'y'])
            label_df.to_csv(spam_msg_label_fp, columns=[target_col, 'y'], sep=sep, header=None, index=None)
            model.get_predicate('spam_msg').add_data_file(Partition.TRUTH, spam_msg_label_fp)
