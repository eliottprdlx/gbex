import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genpareto, norm, qmc
from tqdm import tqdm
from sklearn.model_selection import KFold
import plotly.graph_objects as go
from GBEX import GBEX


def cv_mise_path(X, Y, tau, D_sig, D_gam, B_list, true_quantile_func, dim, s=0.75, l1_reg=0, l2_reg=0, L_sig=10, L_gam=10, K=5):

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_deviances = []
    fold_mises = []

    for (train_idx, val_idx) in tqdm(kf.split(X)):

        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        model = GBEX(
            tau_0=0.8,
            B=B_list[-1],
            D_sig=D_sig,
            D_gam=D_gam,
            lamb_sig=0.01,
            lamb_gam=0.001,
            s=0.75,
            L_sig=L_sig,
            L_gam=L_gam,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            clip=True
        )

        model.fit(X_train, Y_train)

        fold_mise = []
        for B in B_list:
            mise = compute_mise_quantile(model, X_val, true_quantile_func, dim, tau, num_trees=B)
            fold_mise.append(mise)

        fold_mises.append(fold_mise)

    fold_mises = np.array(fold_mises)
    mean_mise = fold_mises.mean(axis=0)
    std_mise = fold_mises.std(axis=0)

    return mean_mise, std_mise


def compute_mise_quantile(model, X_test, quantile_func, dim, tau, num_samples=1000, R=10, num_trees=None):
    B = num_trees if num_trees is not None else model.B
    ise_values = []

    if not X_test.any():

        for _ in range(R):
            # Generate quasi-random Halton sequence in [-1, 1]^dim
            sampler = qmc.Halton(d=dim, scramble=True)
            X_test = sampler.random(num_samples) * 2 - 1

            quantile_pred = model.predict(X_test, tau=tau, num_trees=B)

            quantile_true = np.array([
                quantile_func(x) 
                for x in X_test
            ])

            ise = np.mean((quantile_pred - quantile_true) ** 2)
            ise_values.append(ise)
        
    else:
        
        quantile_pred = model.predict(X_test, tau=tau, num_trees=B)
        quantile_true = np.array([
            quantile_func(x) 
            for x in X_test
        ])
        ise = np.mean((quantile_pred - quantile_true) ** 2)
        ise_values.append(ise)

    # mean over R replications
    mise = np.mean(ise_values)
    return mise