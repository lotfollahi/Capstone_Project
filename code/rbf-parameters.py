print(__doc__)

import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import classify
import re
import os
import os.path as path
import warnings
import envi

# #############################################################################
# Load and prepare data set
#
# dataset for grid search

# load train data
data_path = '/brazos/mayerich/berisha/data/hd/pca16/'
masks_path = '/brazos/mayerich/berisha/data/hd/masks/com-left/'

classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
C = classify.filenames2class(classimages)  # generate the class images for training

# open the ENVI file for reading, use the validation image for batch access

# Load the envi file names.
data_file_names = []
compiled = re.compile("\.")
for f in os.listdir(data_path):
    if not f.startswith('.'):
        abs_path = path.join(data_path, f)
        file_name = re.split(compiled, abs_path)[0]
        if path.isfile(abs_path) and not (file_name in data_file_names):
            data_file_names.append(file_name)

if len(data_file_names) < 1:
    raise ValueError('Missing data files in folder %s!' % data_path)
elif len(data_file_names) > 1:
    warnings.warn("More than one ENVI file in data folder")

Etrain = envi.envi(data_file_names[0])
x_train, y_train = Etrain.loadtrain_balance(C, num_samples=10000)
Etrain.close()


# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.

C_range = np.logspace(-3, 3, 7)
gamma_range = np.logspace(-3, 3, 7)
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv, n_jobs=28)
grid.fit(x_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))
