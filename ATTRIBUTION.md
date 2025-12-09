# External Libraries:
- time
- pickle
- os
- sys
- numpy
- pandas 
- seaborn 
- torch
- matplotlib.pyplot
- sklearn.model_selection: train_test_split, cross_val_score, GroupKFold, cross_val_predict, RandomizedSearchCV, GroupShuffleSplit, cross_val_score
- sklearn.metrics: r2_score, mean_squared_error, root_mean_squared_error, r2_score
- sklearn.linear_model: Ridge, Lasso, ElasticNet, LinearRegression
- sklearn.ensemble: RandomForestRegressor
- sklearn.pipeline: Pipeline
- sklearn.neighbors: KNeighborsRegressor
- sklearn.neural_network: MLPRegressor
- sklearn.preprocessing: StandardScaler, PolynomialFeatures, OneHotEncoder
- scipy.stats: pearsonr, spearmanr
- torch.nn: nn
- torch.utils.data: TensorDataset, DataLoader, random_split
- torchvision.transforms: ToTensor
- torch.optim: optim

# Datasets:
## GDSC1 Dataset: 
https://www.cancerrxgene.org/downloads/bulk_download - GDSC1-Dataset
(https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC1_fitted_dose_response_27Oct23.xlsx)

## Cell Line Annotations Data:
https://www.cancerrxgene.org/downloads/bulk_download - Cell-line-annotation
(https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/Cell_Lines_Details.xlsx)

## Expression Data: 
https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
EXP	Preprocessed	Cell-lines	RMA normalised expression data for cell-lines	RMA normalised basal expression profiles for all the cell-lines

## Mutation Data:
https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources/Home.html
SEQ	Preprocessed	Cell-lines	Sequencing BEMs for cell-lines	Binary event matrices with status of CGs across cell-lines	CFEs in Cell Lines
BEMs

# AI-generated code:
Utilized AI to help me create the following files:
- src/hvg.py for dimensionality reduction, 
- src/plotting.py
- ensemble_all_drugs in models/nn_standard_model and models/nn_mut_model

Utilized AI to help me structure and format the following files:
- src/run_elasticnet.py
- src/run_randomforest.py
- src/mkp.py

And to assist with data transformation in models/nn_standard_model and models/nn_mut_model (first few code blocks)










