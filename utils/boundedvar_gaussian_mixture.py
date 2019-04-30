from sklearn.mixture import GaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky, _estimate_gaussian_parameters
from scipy import linalg
import numpy as np


action_bound = 2
sigma_bound = 1

class BoundedVarGaussianMixture(GaussianMixture):

    # override _initialize
    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)

        ### Bound covariance
        means = np.clip(means, -action_bound, action_bound)
        covariances = np.clip(covariances, np.exp(-2 * sigma_bound), np.exp(2 * sigma_bound))
        ###
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        if self.precisions_init is None:
            self.covariances_ = covariances
            self.precisions_cholesky_ = _compute_precision_cholesky(
                covariances, self.covariance_type)
        elif self.covariance_type == 'full':
            self.precisions_cholesky_ = np.array(
                [linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.precisions_init])
            raise ValueError
        elif self.covariance_type == 'tied':

            self.precisions_cholesky_ = linalg.cholesky(self.precisions_init,
                                                        lower=True)
            raise ValueError
        else:
            self.precisions_cholesky_ = self.precisions_init
            raise ValueError

    def _m_step(self, X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))

        ### Bound covariance

        self.means_ = np.clip(self.means_, -action_bound, action_bound)
        self.covariances_ = np.clip(self.covariances_, np.exp(-2 * sigma_bound), np.exp(2 * sigma_bound))
        ###

        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)
