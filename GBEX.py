import numpy as np
from TreeGB import TreeGB
from GB_GPD import GB_GPD
import quantile_forest as qf


class GBEX():
    """ Gradient Boosting for Extreme Quantile Regression using Decision Trees. """
    def __init__(self, tau_0=0.8, B=100, D_sig=2, D_gam=2, lamb_sig=0.01, lamb_gam=0.001, s=0.5, L_sig=10, L_gam=10, l1_reg=0.0, l2_reg=0.0, clip=False):
        # Intermediate quantile
        self.tau_0 = tau_0

        # Quantile Forest for intermediate quantile estimation
        self.intermediate_quantile = qf.RandomForestQuantileRegressor(n_estimators=100, default_quantiles=tau_0, min_samples_leaf=20, max_depth=3)

        # GB-GPD model for tail estimation
        self.gb_gpd = GB_GPD(B=B, D_sig=D_sig, D_gam=D_gam, lamb_sig=lamb_sig, lamb_gam=lamb_gam, s=s, L_sig=L_sig, L_gam=L_gam, l1_reg=l1_reg, l2_reg=l2_reg, clip=clip)
    

    def fit(self, X, Y):
        # Fit the intermediate quantile model
        self.intermediate_quantile.fit(X, Y)

        # Compute excesses for tail modeling
        q_tau_0 = self.intermediate_quantile.predict(X, quantiles=self.tau_0)
        ind_exc = Y > q_tau_0
        X_exc = X[ind_exc]
        Z = Y[ind_exc] - q_tau_0[ind_exc]

        self.gb_gpd.fit(X_exc, Z, sigma_0=None, gamma_0=None)  
    

    def predict(self, X, tau, num_trees = None):
        # Estimate the intermediate quantile
        q_tau_0 = self.intermediate_quantile.predict(X, quantiles=self.tau_0)

        # Predict GPD parameters
        sigma_pred, gam_pred = self.gb_gpd.predict(X, num_trees)

        # Compute the final quantile prediction
        quantile_pred = q_tau_0 + (sigma_pred / gam_pred) * (((1 - self.tau_0) / (1 - tau))**gam_pred - 1)
        
        return quantile_pred

    
    def compute_deviance(self, X, Y, tau, num_trees = None):
        q_pred = self.predict(X, tau)
        ind_exc = Y > q_pred
        X_exc = X[ind_exc]
        Z = Y[ind_exc] - q_pred[ind_exc]
        sigma_pred, gam_pred = self.gb_gpd.predict(X_exc, num_trees)
        deviance = np.mean((1 + 1/gam_pred) * np.log(1 + gam_pred * Z / sigma_pred) + np.log(sigma_pred))
        return deviance
        

