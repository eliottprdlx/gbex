from sklearn.tree import DecisionTreeRegressor
import numpy as np


class TreeGB():
    """ A simple decision tree regressor for gradient boosting."""
    def __init__(self, max_depth=2, min_samples_leaf=2, clip=False, l1_reg=0.0, l2_reg=0.0):
        self.model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.max_depth = max_depth
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.leaf_values = None
        self.clip = clip
    
    
    def _get_leaf_values(self, X, y):
        leaf_indices = self.model.apply(X)
        leaf_values = np.zeros(np.max(leaf_indices) + 1)
        for leaf in np.unique(leaf_indices):
            mask = (leaf_indices == leaf)
            leaf_values[leaf] = np.mean(y[mask])
        return leaf_values
    

    def fit(self, X, y):
        if self.max_depth > 0:
            self.model.fit(X, y)
            self.leaf_values = self._get_leaf_values(X, y)
    

    def update_leaf_values(self, X, dl1, dl2):
        if not self.max_depth:
            xi = -np.sum(dl1) / np.sum(dl2)
            self.leaf_values = np.array([np.clip(xi, -1, 1)])
        
        else:
            leaf_indices = self.model.apply(X)
            for leaf in np.unique(leaf_indices):
                mask = (leaf_indices == leaf)
                dl1_sum = np.sum(dl1[mask])
                dl2_sum = np.sum(dl2[mask])
                s = np.sign(dl1_sum) * max(abs(dl1_sum) - self.l1_reg, 0) # soft-thresholding step
                xi = - s / (dl2_sum + self.l2_reg)
                if self.clip:
                    self.leaf_values[leaf] = np.clip(xi, -1, 1)
                else:
                    self.leaf_values[leaf] = xi
    

    def predict(self, X):
        if not self.max_depth:
            return np.ones(X.shape[0]) * self.leaf_values[0]
        
        leaf_indices = self.model.apply(X)
        predictions = np.array([self.leaf_values[leaf] for leaf in leaf_indices])
        return predictions
