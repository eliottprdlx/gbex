from scipy.stats import genpareto
import numpy as np
from TreeGB import TreeGB


class GB_GPD():
    """ Gradient Boosting for Generalized Pareto Distribution parameter estimation."""
    def __init__(self, B=100, D_sig=2, D_gam=2, lamb_sig=0.01, lamb_gam=0.001, s=0.5, L_sig=10, L_gam=10, l1_reg=0.0, l2_reg=0.1, clip=False):
        # Number of trees
        self.B = B

        # Max depths 
        self.D_sig = D_sig
        self.D_gam = D_gam

        # Regularization parameters
        self.lamb_sig = lamb_sig
        self.lamb_gam = lamb_gam

        # Bootstrap sampling rate 
        self.s = s

        # Minimum samples per leaf
        self.L_sig = L_sig
        self.L_gam = L_gam

        # Lists to hold the trees
        self.trees_sig = []
        self.trees_gam = []

        # Initial GPD parameters
        self.sigma_0 = None
        self.gamma_0 = None

        # Regularization for leaf value updates
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        # Clipping option
        self.clip = clip
    

    def _compute_gradients(self, X, Z, sig_pred, gam_pred):

        assert all(Z > 0) # Ensure all excesses are positive

        # Compute first and second derivatives of the negative log-likelihood
        sig_residuals= (1 - (1+gam_pred)*Z / (sig_pred+gam_pred*Z)) / sig_pred
        dl2_sig = (Z/sig_pred + (Z - sig_pred)/(sig_pred + gam_pred*Z)) / (sig_pred*(sig_pred + gam_pred*Z))

        arg = 1 + gam_pred * Z / sig_pred
        arg = np.maximum(arg, 1e-8)  # safeguard
        gam_residuals = -np.log(arg) / gam_pred**2 + (1 + 1/gam_pred) * Z / (sig_pred + gam_pred*Z)
        dl2_gam = 2 / gam_pred**3 * np.log(arg) - 2 * Z / (gam_pred**2 * (sig_pred + gam_pred*Z)) - (1 + 1/gam_pred) * Z**2 / (sig_pred + gam_pred*Z)**2

        return sig_residuals, dl2_sig, gam_residuals, dl2_gam


    def fit(self, X, Z, sigma_0=None, gamma_0=None):
        n = X.shape[0]

        # Initialize GPD parameters
        if sigma_0 is None or gamma_0 is None:
            self.gamma_0, _, self.sigma_0 = genpareto.fit(Z, floc=0)
        
        else:
            self.sigma_0, self.gamma_0 = sigma_0, gamma_0

        sigma_pred = np.ones(n) * self.sigma_0
        gamma_pred = np.ones(n) * self.gamma_0
        
        # Boosting iterations
        for b in range(self.B):
            
            # Bootstrap sampling
            subsample = np.random.choice(n, size=int(0.8*n), replace=False)
            X_b = X[subsample]
            Z_b = Z[subsample]

            # Compute gradients
            sig_residuals, dl2_sig, gam_residuals, dl2_gam = self._compute_gradients(X_b, Z_b, sigma_pred[subsample], gamma_pred[subsample])

            # Fit trees to gradients and leaves update
            tree_sig = TreeGB(max_depth=self.D_sig, min_samples_leaf=self.L_sig, l1_reg=self.l1_reg, l2_reg=self.l2_reg, clip=self.clip)
            tree_gam = TreeGB(max_depth=self.D_gam, min_samples_leaf=self.L_gam, l1_reg=self.l1_reg, l2_reg=self.l2_reg, clip=self.clip)
            tree_sig.fit(X_b, sig_residuals)
            tree_sig.update_leaf_values(X_b, sig_residuals, dl2_sig)
            tree_gam.fit(X_b, gam_residuals)
            tree_gam.update_leaf_values(X_b, gam_residuals, dl2_gam)
            self.trees_sig.append(tree_sig)
            self.trees_gam.append(tree_gam)

            # Update predictions
            sigma_pred += self.lamb_sig * tree_sig.predict(X)
            gamma_pred += self.lamb_gam * tree_gam.predict(X)


    def predict(self, X, num_trees = None):
        n = X.shape[0]
        sigma_pred = np.ones(n) * self.sigma_0
        gamma_pred = np.ones(n) * self.gamma_0
        B = self.B if num_trees is None else num_trees
        
        for b in range(B):
            sigma_pred += self.lamb_sig * self.trees_sig[b].predict(X)
            gamma_pred += self.lamb_gam * self.trees_gam[b].predict(X)
        
        return sigma_pred, gamma_pred