from __future__ import print_function
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import glob
import classify
import envi
import numpy as np

print(__doc__)

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

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score, n_jobs=8)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(x_test)
    print(classification_report(y_true, y_pred))
    print()

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.