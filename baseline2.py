import time
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

def run_baseline_models_cv(df, drug, label_col, record_time=True, random_state=2025, n_splits=5):
    """
    Run baseline ML models for a single drug using cross-validation.
    Returns mean R2 and RMSE across folds.
    """
    # ---- subset for the selected drug ----
    drug_df = df[df["DRUG_NAME"] == drug].copy()
    if drug_df.empty:
        raise ValueError(f"No rows found for drug '{drug}'.")

    # ---- auto-detect gene columns by excluding non-gene metadata ----
    gene_cols = [
        col for col in df.columns
        if col not in [
            "DRUG_NAME", "DRUG_ID", "COSMIC_ID", "CELL_LINE_NAME", "LN_IC50",
            'DATASET', 'NLME_RESULT_ID', 'NLME_CURVE_ID', 'SANGER_MODEL_ID',
            'TCGA_DESC', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'COMPANY_ID',
            'WEBRELEASE', 'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE', 'Z_SCORE',
            'SAMPLE_NAME', 'WES', 'CNA', 'GENE_EXPRESSION', 'Methylation',
            'DRUG_RESPONSE', 'GDSC_TD1', 'GDSC_TD2', 'CANCER_TYPE', 'MSI',
            'SCREEN_MEDIUM', 'GROWTH_PROPERTIES'
        ]
    ]

    X = drug_df[gene_cols].astype(float)
    y = drug_df[label_col].astype(float)

    # ---- define baseline models ----
    models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state)
    }

    results = {}
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, model in models.items():
        start = time.time()

        # R2 across folds
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        # RMSE across folds
        rmse_scores = np.sqrt(-cross_val_score(
            model, X, y, cv=cv, scoring=make_scorer(mean_squared_error, greater_is_better=False)
        ))

        end = time.time()

        results[name] = {
            "R2_mean": np.mean(r2_scores),
            "R2_std": np.std(r2_scores),
            "RMSE_mean": np.mean(rmse_scores),
            "RMSE_std": np.std(rmse_scores)
        }

        if record_time:
            results[name]["fit_time_sec"] = end - start

    return results, drug_df, gene_cols
