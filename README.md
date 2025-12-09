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
Demo: https://drive.google.com/file/d/1B0vjMSOMoWJXpCVSjLBasEWlKV60YQC0/view?usp=sharing

## Evaluation
### Highest-performing drug in terms of Spearman coefficient for ElasticNet model. Higher Spearman's coefficient than in Li et al. (2021) for the same drug, indicating that ElasticNet would have more success predicting drug sensitivity for Trametinib when compared to the paper's GA/KNN Algorithm. 

<img width="486" height="487" alt="Screenshot 2025-12-09 at 16 40 46" src="https://github.com/user-attachments/assets/288a6bcc-1c89-4689-b434-62011f6dda4e" />
<img width="486" height="487" alt="Screenshot 2025-12-09 at 16 41 12" src="https://github.com/user-attachments/assets/a4dfd0bf-d447-448e-bcc7-c3f46c7e004d" />

- Cisplatin:
  -- ElasticNet standard average fit time: 0.5725 s
  -- ElasticNet + mutation average fit time: 0.6203 s

- PLX-4720:
  -- ElasticNet standard average fit time: 1.3843 s
  -- ElasticNet + mutation average fit time: 0.9544 s

- Selumetinib:
  -- ElasticNet standard average fit time: 0.5012 s
  -- ElasticNet + mutation average fit time: 0.4615 s

- Trametinib:
  -- ElasticNet standard average fit time: 0.0699 s
  -- ElasticNet + mutation average fit time: 0.0693 s

- Nutlin-3a (-):
  -- ElasticNet standard average fit time: 0.1204 s
  -- ElasticNet + mutation average fit time: 0.1123 s

### Some drugs had a higher Spearman coefficient in RandomForest only when the mutation data was included.
<img width="486" height="487" alt="Screenshot 2025-12-09 at 16 46 00" src="https://github.com/user-attachments/assets/a48d7b68-2f14-4f68-9878-a96fcbf040c1" />
<img width="486" height="487" alt="Screenshot 2025-12-09 at 16 46 56" src="https://github.com/user-attachments/assets/d1a4ee75-c4c4-4dc1-be97-8ad371ba0027" />

- Cisplatin:
  -- ElasticNet standard average fit time: 1486.9712 s
  -- ElasticNet + mutation average fit time: 1487.1691 s

- PLX-4720:
  -- ElasticNet standard average fit time: 1575.1106 s
  -- ElasticNet + mutation average fit time: 1532.3851 s

- Selumetinib:
  -- ElasticNet standard average fit time: 1282.0442 s
  -- ElasticNet + mutation average fit time: 1272.8632 s

- Trametinib:
  -- ElasticNet standard average fit time: 475.4934 s
  -- ElasticNet + mutation average fit time: 492.4069 s

- Nutlin-3a (-):
  -- ElasticNet standard average fit time: 535.6381 s
  -- ElasticNet + mutation average fit time: 548.8906 s

### Standard Data 
<img width="983" height="583" alt="Screenshot 2025-12-09 at 16 52 32" src="https://github.com/user-attachments/assets/dae9bc1b-ba0c-4969-b7e5-9229444837a8" />
<img width="727" height="447" alt="Screenshot 2025-12-09 at 16 52 58" src="https://github.com/user-attachments/assets/0a231b48-2ac9-4f2e-8bb9-00f918710f2a" />
<img width="1188" height="783" alt="Screenshot 2025-12-09 at 16 54 05" src="https://github.com/user-attachments/assets/88db9aa4-7d97-4c01-b005-338f1c03eab7" />


### Mutation Data
<img width="983" height="583" alt="Screenshot 2025-12-09 at 16 54 27" src="https://github.com/user-attachments/assets/8786e1fa-0db0-4f53-bd95-94f934ef5047" />
<img width="739" height="449" alt="Screenshot 2025-12-09 at 16 54 40" src="https://github.com/user-attachments/assets/a4521f43-9e58-4f97-96cb-e9b7aaa4a7a0" />
<img width="1183" height="780" alt="Screenshot 2025-12-09 at 16 55 08" src="https://github.com/user-attachments/assets/369aa0bd-203c-4e31-a937-983f70964754" />

