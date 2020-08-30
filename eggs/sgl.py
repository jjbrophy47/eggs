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
        Sequentailly trains stacked learners with pseudo-relational features
        using all of the training data for each stack.
        """

        # note: pr_features[i][j] is X_r for data piece j using predictions from model i
        pr_features = defaultdict(dict)

        self.stacked_models_ = []

        # fit a base model, and stacked models using pseudo-relational features
        for i in range(self.stacks + 1):
            start = time.time()

            X = Xg if i == 0 else util.hstack([Xg, pr_features[i - 1]])
            fit_model = clone(self.estimator).fit(X, y)
            self.stacked_models_.append(fit_model)

            # generate predictions for all subsequent data pieces
            y_hat = fit_model.predict_proba(X)[:, 1]
            Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_ids,
                                                          relations=self.relations,
                                                          data_dir=self.data_dir)
            pr_features[i] = Xr

            if self.logger:
                self.logger.info('stack {}: {:.3f}s'.format(i, time.time() - start))

        # save pseudo-relational features
        self.Xr_cols = Xr_cols

    def _holdout_old(self, Xg, y, target_ids):
        """
        Sequentailly trains stacked learners with pseudo-relational features.
        Trains a base learner on all data, and subseqent learners on
        mutually exclusive pieces of the data.
        """

        # note: pr_features[i][j] is X_r for data piece j using predictions from model i
        pr_features = defaultdict(dict)

        self.base_model_ = clone(self.estimator).fit(Xg, y)
        self.stacked_models_ = []

        Xg_array = self._array_split(Xg, self.stacks + 1)
        y_array = self._array_split(y, self.stacks + 1)
        target_ids_array = self._array_split(target_ids, self.stacks + 1)

        # fit a base model, and stacked models using pseudo-relational features
        for i in range(self.stacks):
            start = time.time()

            # X = Xg_array[i] if i == 0 else util.hstack([Xg_array[i], pr_features[i - 1][i]])
            X = Xg if i == 0 else util.hstack([Xg, pr_features[i - 1][i]])
            # fit_model = clone(self.estimator).fit(X, y_array[i])
            fit_model = clone(self.estimator).fit(X, y)
            self.stacked_models_.append(fit_model)

            # generate predictions for all subsequent data pieces
            for j in range(i + 1, self.stacks):
                # X = Xg_array[j] if i == 0 else util.hstack([Xg_array[j], pr_features[i - 1][j]])
                X = Xg if i == 0 else util.hstack([Xg, pr_features[i - 1][j]])
                y_hat = fit_model.predict_proba(X)[:, 1]
                # Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_ids_array[j],
                #                                               relations=self.relations,
                #                                               data_dir=self.data_dir)
                Xr, Xr_cols = util.pseudo_relational_features(y_hat, target_ids,
                                                              relations=self.relations,
                                                              data_dir=self.data_dir)
                pr_features[i][j] = Xr

            if self.logger:
                self.logger.info('stack {}: {:.3f}s'.format(i, time.time() - start))

        # save pseudo-relational features
        self.Xr_cols = Xr_cols

    def _array_split(self, X, splits):
        """
        Splits sparse array into equal-sized pieces.
        """

        assert splits > 1

        array_list = []
        n = X.shape[0]
        incrementer = int(n / splits)

        for i in range(1, splits + 1):

            if i == splits:
                array_fragment = X[(i - 1) * incrementer:]
            else:
                array_fragment = X[(i - 1) * incrementer: i * incrementer]

            array_list.append(array_fragment)

        return np.array(array_list)
