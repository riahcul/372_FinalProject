import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

def run_baseline_models(df, drug, label_col, record_time=True, random_state=2025):
    """
    Run baseline ML models for a single drug.
    df: full GDSC data table
    drug: name of the drug ("Erlotinib", etc.)
    label_col: response variable ("AUC", "LN_IC50", etc.)
    gene_cols: list of gene expression columns. If None, auto-detect.
    record_time: measure compute cost for each model
    """

    # ---- subset for the selected drug ----
    drug_df = df[df["DRUG_NAME"] == drug].copy()
    if drug_df.empty:
        raise ValueError(f"No rows found for drug '{drug}'.")

    gene_cols = [
    col for col in df.columns
    if col not in ["DRUG_NAME", "DRUG_ID", "COSMIC_ID", "CELL_LINE_NAME", "LN_IC50",'DATASET', 'NLME_RESULT_ID',
                   'NLME_CURVE_ID', 'SANGER_MODEL_ID', 'TCGA_DESC', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'COMPANY_ID',
                   'WEBRELEASE', 'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE', 'Z_SCORE', 'SAMPLE_NAME', 'WES', 'CNA', 'GENE_EXPRESSION',
                   'Methylation', 'DRUG_RESPONSE', 'GDSC_TD1', 'GDSC_TD2', 'CANCER_TYPE', 'MSI', 'SCREEN_MEDIUM', 'GROWTH_PROPERTIES']]


    # ---- split into X and y ----
    X = drug_df[gene_cols].astype(float)
    y = drug_df[label_col].astype(float)

    # ---- train-test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=random_state),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=random_state)
    }

    results = {}

    # ---- train each baseline model ----
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        end = time.time()

        results[name] = {
            "R2": r2_score(y_test, preds),
            "RMSE": root_mean_squared_error(y_test, preds)
        }

        if record_time:
            results[name]["fit_time_sec"] = end - start

    return results, drug_df, gene_cols