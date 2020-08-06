"""
This class implements stacked graphical learning (SGL).
"""
import time

import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from collections import defaultdict

from . import util


class SGL:
    """
    Stacked Graphical Learning (SGL).
    """

    def __init__(self,
                 estimator,
                 relations,
                 stacks=2,
                 method='cv',
                 cv_folds=5,
                 data_dir='data',
                 logger=None,
                 random_state=None):
        """
        Initialization of SGL classifier.

        Parameters
        ----------
        estimator : object
            Classifier object.
        relations : list (default=None)
            Relations to use for relational modeling.
        stacks : int (default=2)
            Number of stacks to use for SGL. Only relevant if sgl is not None.
        method : str (default='cv')
            Domain-dependent helper method to generate pseudo-relational features.
        cv_folds : int (default=5)
            Number of folds to use for the cross-validation method.
        data_dir : str (default='data')
            Dataset directory containing relational information.
        logger : object (default=None)
            Saves output.
        random_state : int (default=None)
            Reprducibility parameter.
        """
        self.estimator = estimator
        self.relations = relations
        self.method = method
        self.cv_folds = cv_folds
        self.stacks = stacks
        self.data_dir = data_dir
        self.logger = logger
        self.random_state = random_state

    def fit(self, X, y, target_col):
        """
        Performs stacked graphical learning using the
        cross-validation or holdout method.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        if self.method == 'cv':
            self._cross_validation(X, y, target_col)

        elif self.method == 'holdout':
            self._holdout(X, y, target_col)

        else:
            raise ValueError('SGL method {} unknown!'.format(self.method))

        return self

    def predict_proba(self, Xg, target_col):
        """
        Perform semi-joint prediction using a sequence
        of stacked models with semi-relational features.
        """
        Xg = check_array(Xg, accept_sparse=True)
        Xr = None

        check_is_fitted(self, 'stacked_models_')

        for i, model in enumerate(self.stacked_models_):
            X = Xg if i == 0 else util.hstack([Xg, Xr])
            y_hat = model.predict_proba(X)[:, 1]
            Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_col,
                                                          relations=self.relations,
                                                          data_dir=self.data_dir)

        y_hat = y_hat.reshape(-1, 1)
        y_score = np.hstack([1 - y_hat, y_hat])
        return y_score

    # private
    def _cross_validation(self, Xg, y, target_ids):
        """
        Trains stacked learners on the entire training set
        using cross-validation.
        """
        Xr = None
        base_model = clone(self.estimator).fit(Xg, y)
        self.stacked_models_ = [base_model]

        if self.logger:
            self.logger.info('\nSGL')

        for i in range(self.stacks):

            if self.logger:
                self.logger.info('\nStack {}'.format(i + 1))

            X = Xg if i == 0 else util.hstack([Xg, Xr])

            y_hat = util.cross_val_predict(X=X,
                                           y=y,
                                           estimator=self.estimator,
                                           num_cvfolds=self.cv_folds,
                                           logger=self.logger,
                                           random_state=self.random_state)

            Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_ids,
                                                          relations=self.relations,
                                                          data_dir=self.data_dir)

            X = util.hstack([Xg, Xr])
            clf = clone(self.estimator).fit(X, y)
            self.stacked_models_.append(clf)

        self.Xr_cols_ = Xr_cols

    def _holdout(self, Xg, y, target_ids):
        """
        Sequentailly trains stacked learners with pseudo-relational features.
        """

        self.stacked_models_ = []

        # note: pr_features[i][j] is X_r for data piece j using predictions from model i
        pr_features = defaultdict(dict)

        # split data into equal-sized pieces
        Xg_list = []
        y_list = []
        target_ids_list = []
        incrementer = int(Xg.shape[0] / (self.stacks + 1))

        for i in range(self.stacks + 1):
            start_ndx = i * incrementer if i > 0 else None
            end_ndx = (i + 1) * incrementer if i < self.stacks else None

            Xg_list.append(Xg[start_ndx:end_ndx])
            y_list.append(y[start_ndx:end_ndx])
            target_ids_list.append(target_ids[start_ndx:end_ndx])

        # fit a base model, and stacked models using pseudo-relational features
        for i in range(self.stacks + 1):
            start = time.time()

            X = Xg_list[i] if i == 0 else util.hstack([Xg_list[i], pr_features[i - 1][i]])
            fit_model = clone(self.estimator).fit(X, y_list[i])
            self.stacked_models_.append(fit_model)

            # generate predictions for all subsequent data pieces
            for j in range(i + 1, self.stacks + 1):
                X = Xg_list[j] if i == 0 else util.hstack([Xg_list[j], pr_features[i - 1][j]])
                y_hat = fit_model.predict_proba(X)[:, 1]
                Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_ids_list[j],
                                                              relations=self.relations,
                                                              data_dir=self.data_dir)
                pr_features[i][j] = Xr

            if self.logger:
                self.logger.info('stack {}: {:.3f}s'.format(i, time.time() - start))

        # save pseudo-relational features
        self.Xr_cols = Xr_cols
