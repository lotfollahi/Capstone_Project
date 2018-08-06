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
from scipy import io

def load_train(data_file_names):
    # load train data
    masks_path = '/brazos/mayerich/berisha/data/hd/masks/com-left/'

    classimages = sorted(glob.glob(masks_path + '*.png'))  # load the class file names
    C = classify.filenames2class(classimages)  # generate the class images for training

    # open the ENVI file for reading, use the validation image for batch access
    Etrain = envi.envi(data_file_names[0])
    # x_train, y_train = Etrain.loadtrain(C)
    x_train, y_train = Etrain.loadtrain_balance(C, num_samples=10000)
    Etrain.close()

    return x_train, y_train

def load_test(data_file_names):
    # load test data
    masks_path = '/brazos/mayerich/berisha/data/hd/masks/new-com-right/'

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


if __name__ == '__main__':

    data_path = '/brazos/mayerich/berisha/data/hd/left-pca16/'

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

    n_estimators = 1 
    start = time.time()

    #clf = OneVsRestClassifier(
    #    BaggingClassifier(SVC(), max_samples=1.0 / n_estimators,
    #                      n_estimators=n_estimators), n_jobs=-1)

    clf = OneVsRestClassifier(SVC(), n_jobs=-1)

    x_train, y_train = load_train(data_file_names)
    

    data_path = '/brazos/mayerich/berisha/data/hd/right-pca16/'

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



    x_test, y_test = load_test(data_file_names)

    print('\n rbf svm -- training')
    clf.fit(x_train, y_train)
    end = time.time()
    print("Bagging SVC", end - start)

    print('\n================predicting=========================')
    start = time.time()
    test_predictions = clf.predict(x_test.transpose())
    
    #predict_proba = clf.predict_proba(x_test.transpose())
    end = time.time()
    #io.savemat('pred_prob_svm_hd_rbf_10k.mat', mdict={'pred_prob_svm_rbf_10k': predict_proba})
    #io.savemat('true_labels_svm_hd_rbf_10k.mat', mdict={'true_labels_svm_hd_rbf_10k': y_test})

    print('SVM', ' test accuracy: ', accuracy_score(y_test, test_predictions))
    print('SVM', ' confusion matrix \n', confusion_matrix(y_test, test_predictions))

    print("test time", end - start)







