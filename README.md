# FinalProject372

## Predicting Cancer Cell Line Drug Sensitivity with Ensemble Machine Learning Models
This project aims to accurately predict the response variable ln(IC50) based off of gene expression data, drug data, and mutation data. IC50 is the concentration that a drug needs to be at to reduce cancer cell growth by 50%. By combining datasets that paint a picture of the landscape of cell line-drug interactions, we are able to use ElasticNet, RandomForest, MLP, and ensemble models for this regression task. 

## What it Does 
This project takes 4 datasets in gene expression, mutation, cell lines, and pharmacology within cancer cell lines and first transforms them into one dataframe. Then, it performs dimensionality reduction on gene expression and mutation, reducing the number of expressed genes from 17k to 1k and the number of mutation genes from 300 to 14. Three models are then evaluated, including ElasticNet, RandomForest, and a simple MLP. Metrics like R^2, RMSE, and Spearman coefficients are tracked to compare accuracy. Finally, an ensemble model with half of the weights from ElasticNet and half of the weights from the MLP is used to predict drug sensitivity. Data visualizations are also provided to inspect runtime, predicted vs. observed ln(IC50), relative R^2, and more.

## Quick Start
This project was completed in Jupyter Notebooks.
1. Download data (provided) and ensure that it is in FinalProject/data
2. Run 1.exploratory_data_analysis.ipynb for data transformation under FinalProject/notebooks
3. Run 2.data_processing.ipynb for data transformation under FinalProject/notebooks
4. Run 3.dimensionality_reduction.ipynb for feature selection under FinalProject/notebooks
5. Run 4.baseline_elasticnet.ipynb for ElasticNet under FinalProject/models
6. Run 5.baseline_randomforest.ipynb for RandomForest under FinalProject/models
7. Run 6.nn_model_standard.ipynb for MLP and ensemble model with standard data under FinalProject/models
8. Run 7.nn_model_mut.ipynb for MLP and ensemble model with mutation data under FinalProject/models

## Video Links
Technical Walkthrough: https://drive.google.com/file/d/1Dx-o1W-YXRdQCCwrTunzYnj1XZ84xRBi/view?usp=sharing

## Evaluation
section that presents any quantitative results, accuracy metrics, or qualitative outcomes from testing,
