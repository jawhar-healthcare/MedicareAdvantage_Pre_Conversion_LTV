# MedicareAdvantage_Pre_Conversion_LTV

Pre-Conversion LTV for Medicare Advantage (MA) policies is used to prioritize more valuable customers and route them to the appropriate agents best suited for the task of closing the lead. <br>
The models used for predicting the LTVs can be used by ambassadors as well for routing potential customers to the carriers with the highest LTV dollar value. <br>

This repo contains python scripts and notebooks for querying ISC, Zipcode, Jornaya, and TransUnion Data, its preprocessing, and subsequent modelling of regression algorithms on this MA data for the purpose of calculating Pre Conversion LTV for Medicare Advantage phone, online lead forms and data lead pings.

## Brief descriptions on the repo's folders and their contents:

### config
#### config.ini
 - Contains important path and preprocessing variables that will be used by various scripts within this repo. 
 - User can modify these variables as per their needs without needing to edit them within individual scripts.

#### secret_names.json
 - AWS Secrets Manager secret names to fetch the user credentials.

### dockerfiles
#### Dockerfile
 - File to build the docker container in the local machine if needed.

### models_py
#### utils/load_config_file.py
- File with helper functions to load the config.ini file and its user-defined variables.

#### utils/utils.py
- File with utility functions that can be used and re-used in any part of the code to facilitate better code functionality.

#### requirements.txt
 - File containing python libraries that would be essential to run user's training scripts inside the docker container on MFLflow.

#### train_catboost.py , train_lgbm.py, train_xgboost.py
 - MLFlow training scripts

### utils
#### load_config_file.py
- File with helper functions to load the config.ini file and its user-defined variables.

#### ma_pre_conversion.py
- Script to run when collating pre-conversion data. Script calls various functions to query ISC, Zipcode, Jornaya, and TransUnion Data and merges them together using certain merging criteria.

#### ma_preprocessing_utils.py
 - Script containing functions to query data from various sources like AWS Redshift and Snowflake, and also do some preprocessing on the queried data.
 
#### post_processing_utils.py
- Script to run when some post-processing like dropping unwanted features, normalization/standardization and/or imputations are required to be applied on the dataset.

#### utils.py
- File with utility functions that can be used and re-used in any part of the code to facilitate better code functionality.

## Brief descriptions on the repo's Jupyter Notebooks

### All_models.ipynb
- Preliminary data exploration and CatBoost modelling on MA Data by @Jawhar.

### MACleaning.ipynb
- Preliminary data cleaning, preprocessing and XGBoost modelling on MA Data by @Jawhar.

### MA_Syn.ipynb
- Data exploration and analysis on MA Data by @Jawhar.

### model_dev.ipynb
- CatBoost model training on Local Machine with MA data only.
- Training, testing and evaluation of regression metrics and SHAP analysis by @Rutvik.

### ltv-models-training.ipynb
- CatBoost model training on MLFlow with MA data only.
- Training, testing and evaluation of regression metrics and SHAP analysis by @Rutvik.
- Single estimator fit and Hyperparameter Tuner fits included.

### cms_ma_model.ipynb
- CatBoost model training on MLFlow with MA data and Synthetic CMS MA data.
- Training, testing and evaluation of regression metrics and SHAP analysis by @Rutvik.
- Single estimator fit and Hyperparameter Tuner fits included.


### Pipfile 
- Pipfile facilitates the creation of a virtual environment or a pipenv (pip environment) in the MLflow container

### Pipfile.lock
- This file is created automatically when we build a pipenv
