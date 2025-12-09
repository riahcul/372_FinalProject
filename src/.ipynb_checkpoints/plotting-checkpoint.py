import numpy as np
import pandas as pd
import seaborn as sns
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def plot_pred_vs_obs(y_true, y_pred, title, figsize=(5,5)): 
    """ Scatterplot of predicted vs observed values. """ 
    rho, pval = spearmanr(y_true, y_pred)
    
    plt.figure(figsize=figsize) 
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7) 
    #plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2) # identity line 
    plt.xlabel("Observed ln(IC50)") 
    plt.ylabel("Predicted ln(IC50)") 
    plt.title(f"{title}") 
    
    plt.text(
        0.95, 0.05, 
        f"ρ={rho:.3f}\np-val={pval:.2e}",
        transform=plt.gca().transAxes,
        fontsize=10,
        color='red',
        verticalalignment='bottom',
        horizontalalignment='right'
    )
    
    plt.tight_layout() 
    plt.show()


def analyze_drug_model_results(results_dict, top_n_drugs=3):
    """
    Analyze ElasticNet, NN, and Ensemble model results across drugs.
    
    Args:
        results_dict: dict with structure {drug: metrics_dict}
        top_n_drugs: number of drugs to plot in scatter/residual plots
        
    Returns:
        summary_df: DataFrame summarizing metrics per drug and model
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # ---- 1. Build summary DataFrame ----
    summary_rows = []
    for drug, metrics in results_dict.items():
        summary_rows.append({
            "drug": drug,
            "enet_r2": metrics["enet_r2"],
            "nn_r2": metrics["nn_r2"],
            "ensemble_r2": metrics["ensemble_r2"],
            "enet_rmse": metrics["enet_rmse"],
            "nn_rmse": metrics["nn_rmse"],
            "ensemble_rmse": metrics["ensemble_rmse"],
            "enet_pearson": metrics["enet_pearson"],
            "nn_pearson": metrics["nn_pearson"],
            "ensemble_pearson": metrics["ensemble_pearson"],
            "enet_spearman": metrics["enet_spearman"],
            "nn_spearman": metrics["nn_spearman"],
            "ensemble_spearman": metrics["ensemble_spearman"],
            "ensemble_improvement": metrics["ensemble_r2"] - max(metrics["enet_r2"], metrics["nn_r2"]),
            "n_samples": metrics["n_samples"]
        })
    
    summary_df = pd.DataFrame(summary_rows).sort_values("drug")
    
    # ---- 2. Bar plot: R² per drug ----
    r2_df = summary_df.melt(id_vars="drug", value_vars=["enet_r2","nn_r2","ensemble_r2"],
                            var_name="model", value_name="r2")
    plt.figure(figsize=(10,6))
    sns.barplot(data=r2_df, x="drug", y="r2", hue="model", palette={
        "enet_r2": "darkmagenta",
        "nn_r2": "navy",
        "ensemble_r2": "slateblue"
    })
    plt.title("R² per drug and model")
    plt.ylabel("R²")
    plt.xlabel("Drug")
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
    
    # ---- 3. Bar plot: Pearson per drug ----
    pearson_df = summary_df.melt(id_vars="drug", value_vars=["enet_pearson","nn_pearson","ensemble_pearson"],
                                 var_name="model", value_name="pearson")
    plt.figure(figsize=(10,6))
    sns.barplot(data=pearson_df, x="drug", y="pearson", hue="model", palette={
        "enet_pearson": "darkmagenta",
        "nn_pearson": "navy",
        "ensemble_pearson": "slateblue"})
    plt.title("Pearson correlation per drug and model")
    plt.ylabel("Pearson r")
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()
    
    # ---- 4. Ensemble improvement heatmap ----
    plt.figure(figsize=(8,5))
    ensemble_gain_df = summary_df.set_index('drug')[['ensemble_improvement']]
    sns.heatmap(ensemble_gain_df, annot=True, cmap='coolwarm', center=0)
    plt.title("Ensemble R² Improvement Over Best Single Model")
    plt.show()
    
    # ---- 5. Scatter plots and residual plots for top N drugs ----
    top_drugs = summary_df.sort_values("n_samples", ascending=False)["drug"].head(top_n_drugs)
    for drug in top_drugs:
        metrics = results_dict[drug]
        y_true = metrics["y_test"].values if hasattr(metrics["y_test"], "values") else metrics["y_test"]
        
        # --- Predicted vs true scatter ---
        plt.figure(figsize=(12,4))
        for i, (pred, model_name) in enumerate(zip(
            [metrics["y_pred_enet"], metrics["y_pred_nn"], metrics["y_pred_ensemble"]],
            ["ElasticNet","NN","Ensemble"]
        )):
            plt.subplot(1,3,i+1)
            plt.scatter(y_true, pred, alpha=0.6)
            plt.xlabel("Observed ln(IC50)")
            plt.ylabel("Predicted ln(IC50)")
            plt.title(f"{drug} - {model_name}")
            
            rho, pval = spearmanr(y_true, pred)
            plt.text(0.95, 0.05, f"ρ={rho:.3f}\np-val={pval:.2e}", transform=plt.gca().transAxes, fontsize=10, color='red',
                     verticalalignment='bottom', horizontalalignment='right')
        plt.tight_layout()
        plt.show()
        
        # --- Residual plots ---
        plt.figure(figsize=(12,4))
        for i, (pred, model_name) in enumerate(zip(
            [metrics["y_pred_enet"], metrics["y_pred_nn"], metrics["y_pred_ensemble"]],
            ["ElasticNet","NN","Ensemble"]
        )):
            plt.subplot(1,3,i+1)
            residuals = y_true - pred
            sns.histplot(residuals, bins=20, kde=True)
            plt.xlabel("Residual")
            plt.title(f"{drug} - {model_name} Residuals")
        plt.tight_layout()
        plt.show()

    
    # ---- 7. Top/Bottom ensemble improvement ----
    print("\nTop 5 drugs by ensemble improvement:")
    display(summary_df[['drug','ensemble_r2','ensemble_improvement']].sort_values('ensemble_improvement', ascending=False).head(5))
    
    print("\nBottom 5 drugs by ensemble improvement:")
    display(summary_df[['drug','ensemble_r2','ensemble_improvement']].sort_values('ensemble_improvement').head(5))
    
    return summary_df
