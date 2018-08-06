import time
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import classify
import glob
import re
import os
import os.path as path
import warnings
import envi
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_train(data_file_names):
    # load train data
    masks_path = '/media/stim-scratch/berisha/00-annotations/hd/left/com-left/'

    classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
    C = classify.filenames2class(classimages)  # generate the class images for training

    # open the ENVI file for reading, use the validation image for batch access
    Etrain = envi.envi(data_file_names[0])
    # x_train, y_train = Etrain.loadtrain(C)
    x_train, y_train = Etrain.loadtrain_balance(C, num_samples=100000)
    Etrain.close()

    return x_train, y_train

def load_test(data_file_names):
    # load test data
    masks_path = '/media/stim-scratch/berisha/00-annotations/hd/right/com-right/'

    classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
    C = classify.filenames2class(classimages)  # generate the class images for testing
    C = C.astype(np.uint32)

    bool_mask = np.sum(C.astype(np.uint32), 0)

    # get number of classes
    num_classes = C.shape[0]

    for i in range(1, num_classes):
        C[i, :, :] *= i + 1

    total_mask = np.sum(C.astype(np.uint32), 0)  # validation mask

    Etest = envi.envi(data_file_names[0], mask=total_mask)

    N = np.count_nonzero(total_mask)  # set the batch size
    Tv = []  # initialize the target array to empty
    x_test = Etest.loadbatch(N)
    y_test = total_mask.flat[np.flatnonzero(total_mask)]  # get the indices of valid pixels

    Etest.close()

    return x_test, y_test



data_path = '/media/stim-processed/berisha/breast-processing/hd/brc961/pca/pca16/'

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

n_estimators = 10
start = time.time()
clf = SVC(gamma=0.1, C=1)
clf = OneVsRestClassifier(
    BaggingClassifier(SVC(gamma=0.1, C=1), max_samples=1.0 / n_estimators,
                      n_estimators=n_estimators))

x_train, y_train = load_train(data_file_names)
clf.fit(x_train, y_train)
end = time.time()
print
"Bagging SVC", end - start

print('\n rbf svm')
print('Linear SVM', ' train accuracy: ', accuracy_score(y_train, clf.predict(x_train)))

x_test, y_test = load_train(data_file_names)

print('\n================predicting=========================')
start = time.time()
test_predictions = clf.predict(x_test.transpose())
end = time.time()

print('Linear SVM', ' test accuracy: ', accuracy_score(y_test, test_predictions))
print('Linear SVM', ' confusion matrix \n', confusion_matrix(y_test, test_predictions))

print
"test time", end - start







