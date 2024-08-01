#### Load libraries ####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from xgboost import XGBClassifier
import shap

#### Read the data and trained XGBoost model ####
"""
Description:

This is a code that loads the XGBoost model used for external validation and the external validation dataset.
Since the same method is used for all three scenarios provided in pipeline.py, the analyst must pair the dataset and the trained model well according to the scenario set.
"""

# Loading XGBoost model and dataset
with open(file='best_xgb_external.pickle'), mode='rb') as f:
    best_xgb = pickle.load(f)

test = pd.read_csv('test.csv')
X = test.drop(['GHD/ISS'], axis=1)
y = test['GHD/ISS']

y_pred = best_xgb.predict(X)


#### Draw several SHAP plots ####
"""
Description:

'SHAP (SHapley Additive exPlanations)' is the method to describe machine learning models and the name of a Python library.
This code is the code to derive the SHAP results for external validation. 
If you want to derive the SHAP results for internal validation, it is strongly recommended to modify the XGBoost model and dataset loaded above appropriately.
"""

# Get 'Explainer' for SHAP and shap values for each record
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X)

# Get SHAP's feature importance plot
"""
Description:

This is a code for generating SHAP's feature importance plot.

It is necessary to adjust the matplotlib settings for the size, color, and font size of the figure.
Additionally, the number of variables displayed at once can be controlled ('max_display').
"""
plt.rc('font', size=18)
plt.figure(figsize=(6, 3))
f = plt.figure()
shap.plots.bar(shap_values, max_display=X.shape[1])
plt.tight_layout()
f.savefig('feature_importance.tif', dpi=300, bbox_inches='tight')

# Get SHAP's summary plot
"""
Description:

This is a code for generating SHAP's summary plot.
It displays feature importance and impact across all records in a dataset, showing both the magnitude and direction of each feature's effect on model output.

This code generates a plot in grayscale, and is designed to draw plots for up to 10 features.
It is possible to generate plots in other forms by adjusting the options.

"""
colors = ["lightgray", "darkgray", "black"]
plt.rc('font', size=18)
plt.figure(figsize=(8, 6))
f = plt.figure()
shap.summary_plot(shap_values, X, max_display=10, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", colors), alpha=0.7)
f.savefig('dot_summary.tif', dpi=300, bbox_inches='tight')

# Get SHAP's waterfall plot
"""
Description:

This is a code for generating SHAP's summary plot.
It visualizes how each feature contributes to pushing the model output from the base value to the final prediction for a single instance.
It is possible to create several plots by changing the index values ​​within the dataset.
In this study, one index was extracted for each of true positive, true negative, false positive, and false negative to derive an interpretable plot.

When using each data, a process of finding an index is required by comparing the class predicted by XGBoost(`y_pred`) with the actual class(`y`).

"""

idx = 2 ## Change the index.

print('True label:', y[idx])
print('Predicted label:', y_pred[idx])
shap.plots.waterfall(shap_values[idx])
f.savefig('waterfall_plot.tif', dpi=300, bbox_inches='tight')

