"""
This class uses a pipeline approach of a traditional classifier
in combination with two types of relational reasoning:
  1. Stacked Graphical Learning (SGL)
  2. Probabilistic Graphical Model (PGM)
"""
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted

from .sgl import SGL
from .pgm import PGM


class EGGSClassifier:
    """
    Extended Group-Based Graphical Models for Spam (EGGS).
    """

    def __init__(self,
                 estimator,
                 relations=None,
                 sgl_method=None,
                 sgl_stacks=2,
                 pgm=None,
                 data_dir='data',
                 logger=None,
                 random_state=None):
        """
        Initialization of EGGS classifier.

        Parameters
        ----------
        estimator : object
            Classifier with fit and predict methods.
        relations : list (default=None)
            Relations to use for relational modeling.
        sgl_method : str {'cv', 'holdout', None} (default=None)
            If not None, method for stacked graphical learning.
            Cross-validation (cv) or sequential (holdout).
        sgl_stacks : int (default=2)
            Number of stacks to use for SGL. Only relevant if sgl is not None.
        pgm : str {'psl', 'mrf', None}
            Probabilistic graphical model to use for joint inference.
        data_dir : str (default='data')
            Dataset directory containing relational information.
        logger : object (default=None)
            Saves output.
        random_state : int (default=None)
            Reprducibility parameter.
        """
        self.estimator = estimator
        self.relations = relations
        self.sgl_method = sgl_method
        self.sgl_stacks = sgl_stacks
        self.pgm = pgm
        self.data_dir = data_dir
        self.logger = logger
        self.random_state = random_state

    def __str__(self):
        return '%s' % self.get_params()

    def fit(self, X_train, y_train, target_ids_train, X_val, y_val, target_ids_val):
        """
        Train the base/SGL and PGM models.
        """
        X_train, y_train = check_X_y(X_train, y_train, accept_sparse=True)
        if y_train.dtype == np.float and not np.all(np.mod(y_train, 1) == 0):
            raise ValueError('Unknown label type: ')

        self.n_feats_ = X_train.shape[1]
        self.classes_ = np.unique(y_train)

        # train an SGL model
        if self.sgl_method in ['holdout', 'cv'] and self.sgl_stacks > 0:
            assert self.relations is not None

            # train
            sgl = SGL(estimator=self.estimator,
                      relations=self.relations,
                      method=self.sgl_method,
                      stacks=self.sgl_stacks,
                      data_dir=self.data_dir,
                      logger=self.logger,
                      random_state=self.random_state)

            self.sgl_ = sgl.fit(X_train, y_train, target_ids_train)

        else:
            self.clf_ = clone(self.estimator).fit(X_train, y_train)

        # train a joint inference model
        if self.pgm in ['psl', 'mrf']:
            assert self.relations is not None

            # compute priors
            if self.sgl_method in ['holdout', 'cv'] and self.sgl_stacks > 0:
                y_hat_val = self.sgl_.predict_proba(X_val, target_ids_val)

            else:
                y_hat_val = self.clf_.predict_proba(X_val)

            # train
            pgm = PGM(relations=self.relations,
                      pgm_type=self.pgm,
                      data_dir=self.data_dir,
                      logger=self.logger)

            self.pgm_ = pgm.fit(y_val, y_hat_val[:, 1], target_ids_val)

        return self

    def predict_proba(self, X, target_ids):
        """
        Get priors from base/SGL model, then perform joint inference.
        """
        X = check_array(X, accept_sparse=True)
        if X.shape[1] != self.n_feats_:
            raise ValueError('X does not have the same number of features!')

        # perform SGL inference
        if self.sgl_method in ['holdout', 'cv'] and self.sgl_stacks > 0:
            check_is_fitted(self, 'sgl_')
            y_hat = self.sgl_.predict_proba(X, target_ids)

        else:
            check_is_fitted(self, 'clf_')
            assert hasattr(self.clf_, 'predict_proba')
            y_hat = self.clf_.predict_proba(X)

        # perform joint inference
        if self.pgm in ['psl', 'mrf']:
            check_is_fitted(self, 'pgm_')
            y_hat = self.pgm_.inference(y_hat[:, 1], target_ids)

        return y_hat

    def predict(self, X, target_col):
        """
        Binarize probabilities.
        """
        y_score = self.predict_proba(X, target_col)
        return self.classes_[np.argmax(y_score, axis=1)]

    def set_params(self, **params):
        """
        Set the parameters of this model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self):
        """
        Get thr parameters of this model.
        """
        d = {}
        d['estimator'] = self.estimator
        d['relations'] = self.relations
        d['sgl_method'] = self.sgl_method
        d['sgl_stacks'] = self.sgl_stacks
        d['pgm'] = self.pgm
        d['logger'] = self.logger
        return d
