from sklearn.model_selection import GroupKFold, cross_val_predict, RandomizedSearchCV, GroupShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr 
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import time


def run_randomforest(X, y, record_time=True, random_state=2025, max_samples=None):
    """
    Run CV ablation study for RandomForestRegressor over grid of hyperparameters.
    Returns metrics on test set.
    """
    X, y = X.align(y, join="inner", axis=0)
    if max_samples is not None and X.shape[0] > max_samples:
        sampled_idx = X.sample(n=max_samples, random_state=random_state).index
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]

    groups = X.index
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]
    
    gkf = GroupKFold(n_splits=5)
    
    param_dist = {
    "n_estimators": [100, 200],           # keep small for speed
    "max_depth": [5, 10, None],      # explore depth (biggest effect)
    "min_samples_leaf": [1, 3, 5],   # stabilizes RF with small samples
    "max_features": ["sqrt", None]   # small search space
    }
    
    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    
    search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=5,                        
        scoring="r2",
        cv=gkf.split(X_train, y_train, groups=groups_train),
        random_state=random_state,
        n_jobs=-1
    )
    
    start = time.time()
    search.fit(X_train, y_train)
    fit_time = time.time() - start if record_time else None

    best_params = search.best_params_
    best_rf = search.best_estimator_

    print(f"Best params: {best_params}, CV R2: {search.best_score_:.3f}")
    
    y_pred = best_rf.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    
    print(f"Test R2: {r2:.3f}, RMSE: {rmse:.3f}, Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
    
    return {
    "model": best_rf,
    "X_test": X_test,
    "y_test": y_test,
    "y_pred": y_pred,
    "best_params": best_params,
    "r2": r2,
    "rmse": rmse,
    "pearson": pearson_corr,
    "spearman": spearman_corr,
    "fit_time": fit_time
    }