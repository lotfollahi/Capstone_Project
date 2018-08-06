import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import glob
import classify
import envi
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

"""
=====================
Optimal params comparison
=====================

A comparison of several classifiers using ('optimal' params) from scikit-learn on FTIR data.

Code source: Sebastian Berisha
"""


names = ["Random Forest n_estimators=20, max_features=2, min_samples_leaf=9, min_samples_split=6",
         "Random Forest n_estimators=20, max_features=3, min_samples_leaf=10, min_samples_split=3"]

classifiers = [
    RandomForestClassifier(n_estimators=20, max_features=2, min_samples_leaf=9, min_samples_split=6),
    RandomForestClassifier(n_estimators=20, max_features=3, min_samples_leaf=10, min_samples_split=3)
    ]

'''
# RF params
n_estimators = 10,
criterion = "gini",
max_depth = None,
min_samples_split = 2,
min_samples_leaf = 1,
min_weight_fraction_leaf = 0.,
max_features = "auto",
max_leaf_nodes = None,
min_impurity_split = 1e-7,
bootstrap = True,
oob_score = False,
n_jobs = 1,
random_state = None,
verbose = 0,
warm_start = False,
class_weight = None
'''

# load train data
data_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/no-mnf/new-cnn/'
masks_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/masks/no-mnf-bcemn/'

classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
C = classify.filenames2class(classimages)  # generate the class images for training

# open the ENVI file for reading, use the validation image for batch access
Etrain = envi.envi(data_path + 'br1003-br2085b-bas-nor-fin-bip-pca16')
x_train, y_train = Etrain.loadtrain(C)
#x_train, y_train = Etrain.loadtrain_balance(C, num_samples=60000)
Etrain.close()

# load test data
data_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/no-mnf/brc961-proj/brc961-proj/new-cnn/'
masks_path = '/media/stim-processed/berisha/breast-processing/lm/brc961/masks/no-mnf-bcemn/'

classimages = sorted(glob.glob(masks_path + '*.png'))   # load the class file names
C = classify.filenames2class(classimages)   # generate the class images for testing
C = C.astype(np.uint32)

bool_mask = np.sum(C.astype(np.uint32), 0)

# get number of classes
num_classes = C.shape[0]

for i in range(1, num_classes):
    C[i, :, :] *= i+1

total_mask = np.sum(C.astype(np.uint32), 0)  # validation mask

test_set = envi.envi(data_path + 'brc961-br1001-bas-nor-fin-bip-proj-br1003-pca16', mask=total_mask)

N = np.count_nonzero(total_mask)  # set the batch size
Tv = []  # initialize the target array to empty
x_test = test_set.loadbatch(N)
y_test = total_mask.flat[np.flatnonzero(total_mask)]  # get the indices of valid pixels

# iterate over classifiers
for name, clf in zip(names, classifiers):
    clf.fit(x_train, y_train)
    print('\n=========================================')
    print(name, ' train accuracy: ', accuracy_score(y_train, clf.predict(x_train)))
    test_predictions = clf.predict(x_test.transpose())
    print(name, ' test accuracy: ', accuracy_score(y_test, test_predictions))
    print(name, ' confusion matrix \n', confusion_matrix(y_test, test_predictions))
