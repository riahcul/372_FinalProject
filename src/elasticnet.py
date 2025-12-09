from sklearn.model_selection import KFold, cross_val_predict, GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import time

def run_elasticnet(X, y, record_time=True, random_state=2025, max_samples=None):
    """
    Run CV ablation study for ElasticNet over grid of alphas and l1_ratios.
    Returns DataFrame with CV metrics for each hyperparameter combination.
    """
    X, y = X.align(y, join="inner", axis=0)
    if max_samples is not None and X.shape[0] > max_samples:
        sampled_idx = X.sample(n=max_samples, random_state=random_state).index
        X = X.loc[sampled_idx]
        y = y.loc[sampled_idx]
        
    zero_var = X.columns[X.std(axis=0, ddof=0) == 0]
    if len(zero_var) > 0:
        X = X.drop(columns=zero_var)
        
    groups = X.index
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]
    
    alphas = [0.01, 0.05, 0.1]
    l1_ratios = [0.25, 0.5, 0.75]
    
    best_r2 = -np.inf
    best_alpha, best_l1 = None, None
    
    gkf = GroupKFold(n_splits=5)
    
    # hyperparameter tuning (ablation study)
    for alpha in alphas:
        for l1 in l1_ratios:
            pipe = Pipeline([
            ("scale", StandardScaler()),
            ("enet", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=50000, random_state=2025))
        ])
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=gkf.split(X_train, y_train, groups=groups_train),
                                scoring='r2', n_jobs=-1).mean()
            
            if cv_r2 > best_r2:
                best_r2 = cv_r2
                best_alpha, best_l1 = alpha, l1
            
    print(f"Best alpha: {best_alpha}, Best l1_ratio: {best_l1}, CV R2: {best_r2:.3f}")
    
    """
    Selected best alpha and l1 ratio for ElasticNet, so now we're able to fit on the training and predict on the testing data
    """
    
    final_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("enet", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=50000, random_state=2025))])
    
    start = time.time()  
    final_pipe.fit(X_train, y_train)
    fit_time = time.time() - start if record_time else None
    
    y_pred = final_pipe.predict(X_test)
    
        
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    
    print(f"Test R2: {r2:.3f}, RMSE: {rmse:.3f}, Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")

    return {
        "pipeline": final_pipe,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "r2": r2,
        "rmse": rmse,
        "pearson": pearson_corr,
        "spearman": spearman_corr,
        "fit_time": fit_time,
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1
    } 