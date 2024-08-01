#### Load libraries ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import copy
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from skopt import BayesSearchCV, space, plots

#### Read the data ####
""" 
Description:

You need to prepare two csv files: training set (or A hospital dataset) and test set (or B hospital dataset).
The training set was used for model training and internal validation, and the test set was used for external validation.

In this study, there are 3 scenarios:
  1. only clinical features
  2. only radiomics features
  3. all features (clinical features + radiomics features)

Since model learning and validation in all three scenarios follow the same structure, 
it is necessary to prepare 'train.csv' and 'test.csv' for each scenario when learning the model.
In all three scenarios, the target variable indicating GHD and ISS for each patient must be included.

Clinical features are follows:

  - Age: monthly age (integer)
  - Sex: 1 is male, otherwise 0 (binary)
  - Height SDS: standard deviation score for height (float)
  - Weight SDS: standard deviation score for weight (float)
  - BMI SDS: standard deviation score for body mass intex (float)
  - Growth velocity (float)
  - Puberty: 0 if prepuberty, 1 if puberty (binary)
  - MPH SDS: standard deviation score for mid-parental height (float)
  - MPH SDS - Height SDS: difference between MPH SDS and Height SDS (float)
  - IGF-1 SDS: standard deviation score for IGF-1 (float)
  - Bone age (float)
  - Chronological age - bone age: Difference between choronological age and bone age

Radiomics features were created using Python's 'pyradiomics' library. 
Since the number of radiomics features is very large, please refer to the example dataset we uploaded and the official pyradiomics documentation (https://pyradiomics.readthedocs.io/en/latest/) to see the meaning of each variable.

In our study, the target feature is named "GHD/ISS".

  - GHD/ISS: 1 if GHD, 0 if ISS (binary)

"""

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop(['GHD/ISS'], axis=1)
y_train = train['GHD/ISS']

X_test = test.drop(['GHD/ISS'], axis=1)
y_test = test['GHD/ISS']


#### Training the model and doing internal validation ####
"""
Description:

We did 5-fold cross validation for optimizing XGBoost hyperparameters.
For internal validation, given 'train' dataset is divided to training set and test set. 

"""

# Calculate weights for GHD and ISS
class_count = y_train.value_counts()
scale_pos_weight = class_count[0] / class_count[1]

# Initialize the XGBoost model
xgb_int = XGBClassifier(
    random_state=1234, 
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight
)

# Set the hyperparameter space for doing Bayesian optimization
search_spaces= {
    'max_depth': space.Integer(1, 10),
    'n_estimators': space.Integer(100, 1000),
    'learning_rate': space.Real(0.01, 1.0, 'log-uniform'),
    'gamma': space.Real(1e-9, 0.5, 'log-uniform'),
    'scale_pos_weight': space.Real(1e-6, 500, 'log-uniform'),
    'reg_lambda': space.Real(1e-3, 0.1, 'log-uniform'),
    'min_child_weight': space.Integer(1, 5)
}

# Set the 5-fold cross validation indices and parameters for Bayesian optimization
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
n_points = 4
n_iter = 60

# Do Bayesian optimization with 5-fold cross validation
bs_cv = BayesSearchCV(
    estimator=xgb_int,
    search_spaces=search_spaces,
    cv=cv,
    scoring='roc_auc',
    n_points=n_points,
    n_iter=n_iter,
    random_state=1234
)
bs_cv.fit(X_train, y_train)

# Get the optimized XGBoost model
best_xgb_int = bs_cv.best_estimator_

# Given 'train' dataaset is divided to training set and test set for internal validation
X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=1234)

# Refit the optimized XGBoost model
best_xgb_int.fit(X_train_int, y_train_int)

# Internal validation
y_test_pred_int = best_xgb_int.predict(X_test_int)
y_test_prob_int = best_xgb_int.predict_proba(X_test_int)[:, 1]
accuracy = metrics.accuracy_score(y_test_int, y_test_pred_int)
recall = metrics.recall_score(y_test_int, y_test_pred_int)
precision = metrics.precision_score(y_test_int, y_test_pred_int)
auroc = metrics.roc_auc_score(y_test_int, y_test_prob_int)

# Print the result of internal validation
print('<Confusion Matrix>')
print(metrics.confusion_matrix(y_test_int, y_test_pred_int))
print('=' * 60)
print('<Classification Report>')
print(metrics.classification_report(y_test_int, y_test_pred_int))
print('=' * 60)
print(f'Accuracy: {accuracy:.4f}')
print(f'Sensitivity: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'AUROC: {auroc:.4f}')


#### External validation ####
"""
Description:

Above, we completed model training and internal validation with a dataset from one hospital. 
Afterwards, when performing external validation on datasets from other hospitals, we first retrained the model on the 'train' dataset and then performed external validation.

"""

# Copy the best XGBoost for internal validation
best_xgb_ext = copy.deepcopy(best_xgb_int)

# Refit the best XGBoost to whole training set
best_xgb_ext.fit(X_train, y_train)

# External validation
y_test_pred = best_xgb_ext.predict(X_test)
y_test_prob = best_xgb_ext.predict_proba(X_test)[:, 1]
accuracy = metrics.accuracy_score(y_test, y_test_pred)
recall = metrics.recall_score(y_test, y_test_pred)
precision = metrics.precision_score(y_test, y_test_pred)
auroc = metrics.roc_auc_score(y_test, y_test_prob)

print('<Confusion Matrix>')
print(metrics.confusion_matrix(y_test, y_test_pred))
print('=' * 60)
print('<Classification Report>')
print(metrics.classification_report(y_test, y_test_pred))
print('=' * 60)
print(f'Accuracy: {accuracy:.4f}')
print(f'Sensitivity: {recall:.4f}')
print(f'Precision: {precision:.4f}')
print(f'AUROC: {auroc:.4f}')

#### Save the models ####
"""
Description:

Both the internal validation model and the external validation model were saved as objects using Python's pickle. 
This is for future model reuse and verification.
"""
with open(file='best_xgb_internal.pickle', mode='wb') as f:
    pickle.dump(best_xgb_int, f)

with open(file='best_xgb_external.pickle', mode='wb') as f:
    pickle.dump(best_xgb_ext, f)
