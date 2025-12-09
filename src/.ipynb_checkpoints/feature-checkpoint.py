from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

def prepare_features(df, gene_cols=None, num_cols=None, cat_cols=None):
    """
    Prepares X matrix from a dataframe.
    Only includes gene_cols, num_cols, and one-hot encoded cat_cols.
    If already_onehot=True, categorical features are assumed one-hot encoded.
    """
    df_proc = df.copy()
    
    # Start with numeric features
    feature_frames = []

    if gene_cols is not None:
        df_proc[gene_cols] = df_proc[gene_cols].apply(pd.to_numeric, errors='coerce')
        df_proc[gene_cols] = df_proc[gene_cols].fillna(df_proc[gene_cols].median())
        feature_frames.append(df_proc[gene_cols])

    if num_cols is not None:
        df_proc[num_cols] = df_proc[num_cols].apply(pd.to_numeric, errors='coerce')
        df_proc[num_cols] = df_proc[num_cols].fillna(df_proc[num_cols].median())
        feature_frames.append(df_proc[num_cols])

    # One-hot encode categorical columns if needed
    if cat_cols is not None:
        df_proc[cat_cols] = df_proc[cat_cols].fillna("UNABLE TO CLASSIFY")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        one_hot_encoded = encoder.fit_transform(df_proc[cat_cols])
        one_hot_df = pd.DataFrame(
            one_hot_encoded, 
            columns=encoder.get_feature_names_out(cat_cols), 
            index=df_proc.index
        )
        feature_frames.append(one_hot_df)

    # Combine all features into a single DataFrame
    if feature_frames:
        X = pd.concat(feature_frames, axis=1)
    else:
        X = pd.DataFrame(index=df_proc.index)

    # Ensure everything is float
    X = X.astype(float)

    return X

def get_drug_data(df, mut_df, drug_name, label_col="LN_IC50"):
    """
    Prepare feature matrices for a given drug, including mutation data.
    
    Returns:
        X: features without mutation data
        X_mut: features with mutation data
        y: response vector
    """
    drug_df = df[df["DRUG_NAME"] == drug_name].copy()
    if drug_df.empty:
        raise ValueError(f"No rows found for drug '{drug_name}'.")
    
    drug_mut_df = mut_df[mut_df["DRUG_NAME"] == drug_name].copy()
    
    drug_df = drug_df.set_index("COSMIC_ID")
    drug_mut_df = drug_mut_df.set_index("COSMIC_ID")
    
    drug_df, drug_mut_df = drug_df.align(drug_mut_df, join="inner", axis=0)
    
    gene_cols = [col for col in drug_df.columns if col not in [
        "DRUG_NAME", "DRUG_ID", "COSMIC_ID", "CELL_LINE_NAME", label_col,
        'DATASET', 'NLME_RESULT_ID','NLME_CURVE_ID', 'SANGER_MODEL_ID', 
        'TCGA_DESC', 'PUTATIVE_TARGET', 'PATHWAY_NAME', 'COMPANY_ID',
        'WEBRELEASE', 'MIN_CONC', 'MAX_CONC', 'AUC', 'RMSE', 'Z_SCORE', 
        'SAMPLE_NAME', 'WES', 'CNA', 'GENE_EXPRESSION', 'Methylation', 
        'DRUG_RESPONSE', 'GDSC_TD1', 'GDSC_TD2', 'CANCER_TYPE', 'MSI', 
        'SCREEN_MEDIUM', 'GROWTH_PROPERTIES'
    ]]
    
    num_cols = ["WES", "CNA", "GENE_EXPRESSION", "Methylation"]
    cat_cols = ["TCGA_DESC", "CANCER_TYPE", "SCREEN_MEDIUM", "GROWTH_PROPERTIES", 
                "MSI", "PATHWAY_NAME", "PUTATIVE_TARGET", "GDSC_TD1", "GDSC_TD2"]
    

    gene_expr_cols = [c for c in drug_mut_df.columns if c.endswith("_expr")]
    gene_mut_cols = [c for c in drug_mut_df.columns if c.endswith("_mut")]
    
    num_mut_cols = num_cols + gene_mut_cols
    
    X = prepare_features(drug_df, gene_cols, num_cols, cat_cols)
    X_mut = prepare_features(drug_mut_df, gene_expr_cols, num_mut_cols, cat_cols)
    
    y = pd.to_numeric(drug_df[label_col], errors='coerce')
    
    X_mut = X_mut.fillna(0)  # indicate no mutation = 0
    
    X, y = X.align(y, join="inner", axis=0)
    X_mut, y = X_mut.align(y, join="inner", axis=0)
    
    return X, X_mut, y


