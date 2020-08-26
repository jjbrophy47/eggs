"""
This module contains high-level APIs implementing joint inference using PGMs.
"""
import os
import uuid
import shutil

from sklearn.utils.validation import check_is_fitted

from .mrf import MRF
from .psl import PSL


class PGM:
    """
    High-level class with multiple pgm implementations.
    """

    def __init__(self,
                 relations,
                 pgm_type='psl',
                 scoring='auc',
                 psl_learner='mle',
                 data_dir='data',
                 working_dir='.temp',
                 logger=None):
        """
        Initialization of joint inference class.

        Parameters
        ----------
        relations : list
            Relations to use for relational modeling.
        pgm_type : str (default='psl') {'psl', 'mrf'}
            Type of PGM to use for joint inference.
        scoring : str (default='auc') {'auc', 'ap'}
        data_dir : str (default='data')
            Dataset directory containing relational information.
        working_dir : str (default='.temp/')
            Temporary directory to store intermediate files.
        logger : object (default=None)
            Saving output.
        """
        self.relations = relations
        self.pgm_type = pgm_type
        self.scoring = scoring
        self.psl_learner = psl_learner
        self.data_dir = data_dir
        self.working_dir = os.path.join(working_dir, str(uuid.uuid4()))
        self.logger = logger

        # clear the working directory
        if os.path.exists(self.working_dir):
            shutil.rmtree(self.working_dir)
        os.makedirs(self.working_dir)

    def __del__(self):
        """
        Clean up any temporary directories.
        """
        shutil.rmtree(self.working_dir)

    def fit(self, y, y_hat, target_col):
        """
        Trains a PGM model.
            y: true labels for target nodes. shape: (n_samples,).
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """
        if self.logger:
            self.logger.info('\nPGM: {}'.format(self.pgm_type))

        if self.pgm_type == 'psl':
            pgm = PSL(relations=self.relations,
                      data_dir=self.data_dir,
                      working_dir=self.working_dir,
                      learner=self.psl_learner,
                      logger=self.logger)

        elif self.pgm_type == 'mrf':
            pgm = MRF(relations=self.relations,
                      scoring=self.scoring,
                      data_dir=self.data_dir,
                      working_dir=self.working_dir,
                      logger=self.logger)

        else:
            raise ValueError('pgm_type {} unknown!'.format(self.pgm_type))

        self.pgm_ = pgm.fit(y, y_hat, target_col)

        return self

    def inference(self, y_hat, target_col):
        """
        Performs joint inference.
            y_hat: priors for target nodes. shape: (n_samples,).
            target_col: list of target_ids. shape: (n_samples,).
        """
        assert len(y_hat) == len(target_col)
        check_is_fitted(self, 'pgm_')

        y_hat = self.pgm_.inference(y_hat, target_col)
        return y_hat
