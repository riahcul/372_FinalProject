import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def select_genomic_features(
    expression_df, 
    mutation_df=None,  
    n_expr=1000,
    min_expr_mean=0.01,
    degree=2,
    min_mut_freq=0.05,
    plot=True
):
    """
    Feature selection for expression and mutation (optional) data:
    - Expression: Mean-variance trend HVG selection, similar to Seurat 
    - Mutation: Frequency-based selection
    
    Parameters
    ----------
    expression_df : DataFrame
        Expression data (rows=cell lines, cols=COSMIC_ID + genes)
    mutation_df : DataFrame or None
        Mutation data (rows=cell lines, cols=COSMIC_ID + genes)
        If None, only expression selection is performed
    n_expr : int
        Number of expression HVGs to select
    min_expr_mean : float
        Minimum mean expression threshold
    degree : int
        Polynomial degree for mean-variance trend
    min_mut_freq : float
        Minimum mutation frequency (0-1)
    plot : bool
        Whether to generate diagnostic plots
        
    Returns
    -------
    selected_expr_genes : list
        Selected expression gene names
    selected_mut_genes : list or None
        Selected mutation gene names (None if mutation_df not provided)
    expr_results : DataFrame
        Expression gene statistics
    mut_results : DataFrame or None
        Mutation gene statistics (None if mutation_df not provided)
    """
    print("="*60)
    print("GENOMIC FEATURE SELECTION")
    print("="*60)
    
    # ============================================
    # PART 1: EXPRESSION - Mean-Variance HVG
    # ============================================
    print("\n" + "-"*60)
    print("PART 1: EXPRESSION HVG SELECTION")
    print("-"*60)
    
    # Get gene columns only
    expr_genes = [col for col in expression_df.columns if col != 'COSMIC_ID']
    expr_matrix = expression_df[expr_genes]
    
    print(f"\nInput expression genes: {len(expr_genes)}")
    
    # 1. Compute mean and variance
    mu = expr_matrix.mean(axis=0)
    var = expr_matrix.var(axis=0)
    
    print(f"Mean range: [{mu.min():.3f}, {mu.max():.3f}]")
    print(f"Variance range: [{var.min():.3f}, {var.max():.3f}]")
    
    # 2. Filter lowly expressed genes
    keep = mu > min_expr_mean
    mu_filt = mu[keep]
    var_filt = var[keep]
    
    print(f"After filtering (mean > {min_expr_mean}): {len(mu_filt)} genes")
    
    # 3. Fit mean-variance trend
    X = mu_filt.values.reshape(-1, 1)
    y = var_filt.values.reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    r2 = model.score(X_poly, y)
    print(f"Mean-variance trend R²: {r2:.3f}")
    
    # 4. Compute residuals (excess variance)
    var_pred = model.predict(X_poly).flatten()
    residuals = var_filt.values - var_pred
    
    print(f"Residual range: [{residuals.min():.3f}, {residuals.max():.3f}]")
    
    # 5. Select top HVGs by residual
    ranked_idx = residuals.argsort()[::-1]
    selected_expr_genes = mu_filt.index[ranked_idx][:n_expr].tolist()
    
    print(f"\n✓ Selected {len(selected_expr_genes)} expression HVGs")
    
    # Create results dataframe
    expr_results = pd.DataFrame({
        "mean": mu_filt.values,
        "variance": var_filt.values,
        "expected_variance": var_pred,
        "residual": residuals,
        "selected": [g in selected_expr_genes for g in mu_filt.index]
    }, index=mu_filt.index).sort_values("residual", ascending=False)
    
    print("\nTop 10 expression HVGs:")
    print(expr_results.head(10)[['mean', 'variance', 'residual']])
    
    # ============================================
    # PART 2: MUTATIONS - Frequency Selection (OPTIONAL)
    # ============================================
    selected_mut_genes = None
    mut_results = None
    
    if mutation_df is not None:
        print("\n" + "-"*60)
        print("PART 2: MUTATION FREQUENCY SELECTION")
        print("-"*60)
        
        # Get mutation columns
        mut_genes = [col for col in mutation_df.columns if col != 'COSMIC_ID']
        mut_matrix = mutation_df[mut_genes]
        
        print(f"\nInput mutation genes: {len(mut_genes)}")
        
        # 1. Calculate mutation frequency (mean of binary = frequency)
        mut_freq = mut_matrix.mean(axis=0)
        
        print(f"Mutation frequency range: [{mut_freq.min():.2%}, {mut_freq.max():.2%}]")
        
        # 2. Select mutations above frequency threshold
        selected_mut_genes = mut_freq[mut_freq >= min_mut_freq].index.tolist()
        
        print(f"Mutation frequency threshold: ≥{min_mut_freq*100:.1f}%")
        print(f"\n✓ Selected {len(selected_mut_genes)} frequently mutated genes")
        
        # Create results dataframe
        mut_results = pd.DataFrame({
            "frequency": mut_freq.values,
            "n_mutated": (mut_matrix.sum(axis=0)).values,
            "n_samples": len(mut_matrix),
            "selected": [g in selected_mut_genes for g in mut_freq.index]
        }, index=mut_freq.index).sort_values("frequency", ascending=False)
        
        print("\nTop 10 most frequently mutated genes:")
        print(mut_results.head(10)[['frequency', 'n_mutated']])
    else:
        print("\n" + "-"*60)
        print("PART 2: MUTATION SELECTION SKIPPED (no mutation data provided)")
        print("-"*60)
    
    # ============================================
    # PART 3: SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("FEATURE SELECTION SUMMARY")
    print("="*60)
    print(f"Expression HVGs:      {len(selected_expr_genes)}")
    
    if selected_mut_genes is not None:
        print(f"Mutation genes:       {len(selected_mut_genes)}")
        print(f"Total features:       {len(selected_expr_genes) + len(selected_mut_genes)}")
        
        # Check overlap
        overlap = set(selected_expr_genes) & set(selected_mut_genes)
        if len(overlap) > 0:
            print(f"\nGenes in both lists:  {len(overlap)}")
            print(f"Examples: {list(overlap)[:5]}")
    else:
        print(f"Mutation genes:       N/A (not provided)")
        print(f"Total features:       {len(selected_expr_genes)}")
    
    # ============================================
    # PART 4: VISUALIZATION
    # ============================================
    if plot:
        print("\nGenerating diagnostic plots...")
        plot_feature_selection(expr_results, mut_results, 
                              selected_expr_genes, selected_mut_genes)
    
    return selected_expr_genes, selected_mut_genes, expr_results, mut_results


def plot_feature_selection(expr_results, mut_results, selected_expr, selected_mut):
    """
    Create diagnostic plots for feature selection.
    Adapts layout based on whether mutation data is available.
    """
    
    has_mut = mut_results is not None and not mut_results.empty

    if has_mut:
        fig = plt.figure(figsize=(16, 10))
        n_plots = 6
    else:
        fig = plt.figure(figsize=(16, 5))
        n_plots = 3
        
    # ============================================
    # Plot 1: Expression mean-variance
    # ============================================
    ax1 = plt.subplot(2 if has_mut else 1, 3, 1)
    ax1.scatter(expr_results['mean'], expr_results['variance'], 
                alpha=0.2, s=10, c='gray', label='Not selected')
    ax1.scatter(expr_results.loc[selected_expr, 'mean'], 
                expr_results.loc[selected_expr, 'variance'],
                alpha=0.7, s=20, c='red', label='Selected HVGs')
    
    # Plot fitted trend
    sorted_idx = expr_results['mean'].argsort()
    ax1.plot(expr_results['mean'].iloc[sorted_idx], 
             expr_results['expected_variance'].iloc[sorted_idx],
             'b--', linewidth=2, label='Fitted trend', alpha=0.8)
    
    ax1.set_xlabel('Mean Expression', fontsize=11)
    ax1.set_ylabel('Variance', fontsize=11)
    ax1.set_title('Expression: Mean-Variance Relationship', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # ============================================
    # Plot 2: Expression residuals
    # ============================================
    ax2 = plt.subplot(2 if has_mut else 1, 3, 2)
    ax2.scatter(expr_results['mean'], expr_results['residual'], 
                alpha=0.2, s=10, c='gray', label='Not selected')
    ax2.scatter(expr_results.loc[selected_expr, 'mean'], 
                expr_results.loc[selected_expr, 'residual'],
                alpha=0.7, s=20, c='red', label='Selected HVGs')
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Mean Expression', fontsize=11)
    ax2.set_ylabel('Variance Residual', fontsize=11)
    ax2.set_title('Expression: Residual (Excess Variability)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    # ============================================
    # Plot 3: Expression residual distribution
    # ============================================
    ax3 = plt.subplot(2 if has_mut else 1, 3, 3)
    ax3.hist(expr_results['residual'], bins=50, alpha=0.5, color='gray', label='All genes')
    ax3.hist(expr_results.loc[selected_expr, 'residual'], 
             bins=30, alpha=0.7, color='red', label='Selected HVGs')
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Variance Residual', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Expression: Residual Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ============================================
    # Mutation plots (only if data provided)
    # ============================================
    if mut_results is not None and selected_mut is not None:
        # Plot 4: Mutation frequency distribution
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(mut_results['frequency'], bins=50, alpha=0.5, color='gray', 
                 label='All mutation genes', edgecolor='black')
        ax4.hist(mut_results.loc[selected_mut, 'frequency'], 
                 bins=30, alpha=0.7, color='blue', label='Selected mutations',
                 edgecolor='black')
        ax4.set_xlabel('Mutation Frequency', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Mutation: Frequency Distribution', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Top mutated genes
        ax5 = plt.subplot(2, 3, 5)
        top_20 = mut_results.nlargest(20, 'frequency')
        colors = ['blue' if g in selected_mut else 'gray' for g in top_20.index]
        ax5.barh(range(20), top_20['frequency'], color=colors, alpha=0.7)
        ax5.set_yticks(range(20))
        ax5.set_yticklabels(top_20.index, fontsize=9)
        ax5.set_xlabel('Mutation Frequency', fontsize=11)
        ax5.set_title('Mutation: Top 20 Most Frequent', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        ax5.grid(True, alpha=0.3, axis='x')
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        summary_text = f"""
    FEATURE SELECTION SUMMARY
    
    Expression (HVG):
      • Total genes: {len(expr_results):,}
      • Selected: {len(selected_expr):,}
      • Mean residual: {expr_results.loc[selected_expr, 'residual'].mean():.3f}
      • Min residual: {expr_results.loc[selected_expr, 'residual'].min():.3f}
      • Max residual: {expr_results.loc[selected_expr, 'residual'].max():.3f}
    
    Mutations (Frequency):
      • Total genes: {len(mut_results):,}
      • Selected: {len(selected_mut):,}
      • Mean frequency: {mut_results.loc[selected_mut, 'frequency'].mean():.2%}
      • Min frequency: {mut_results.loc[selected_mut, 'frequency'].min():.2%}
      • Max frequency: {mut_results.loc[selected_mut, 'frequency'].max():.2%}
    
    Combined Features: {len(selected_expr) + len(selected_mut):,}
        """
        
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                 verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('genomic_feature_selection.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Plots saved to genomic_feature_selection.png")