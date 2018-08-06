import numpy as np
from sklearn.svm import SVC
import glob
import classify
import envi
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import io
from sklearn.metrics import auc
import matplotlib
#matplotlib.use('Agg')  # use backend that doesn't display to the user
import matplotlib.pyplot as plt
import re
import os.path as path
import warnings
import os

"""
=========================
SVM for HD data
=========================

Train and validate an SVM classifier on HD data.

Code source: Sebastian Berisha
"""



# load train data
data_path = '/media/stim-processed/berisha/breast-processing/hd/brc961/pca/pca16/'
masks_path = '/media/stim-scratch/berisha/00-annotations/hd/left/com-left/'

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
#x_train, y_train = Etrain.loadtrain(C)
x_train, y_train = Etrain.loadtrain_balance(C, num_samples=100000)
Etrain.close()

# load test data
data_path = '/media/stim-processed/berisha/breast-processing/hd/brc961/pca/pca16/'
masks_path = '/media/stim-scratch/berisha/00-annotations/hd/right/com-right/'

classimages = sorted(glob.glob(masks_path + '*.png'))   # load the class file names
C = classify.filenames2class(classimages)   # generate the class images for testing
C = C.astype(np.uint32)

bool_mask = np.sum(C.astype(np.uint32), 0)

# get number of classes
num_classes = C.shape[0]

for i in range(1, num_classes):
    C[i, :, :] *= i+1

total_mask = np.sum(C.astype(np.uint32), 0)  # validation mask

Etest = envi.envi(data_file_names[0], mask=total_mask)

N = np.count_nonzero(total_mask)  # set the batch size
Tv = []  # initialize the target array to empty
x_test = Etest.loadbatch(N)
y_test = total_mask.flat[np.flatnonzero(total_mask)]  # get the indices of valid pixels

Etest.close()

#clf = SVC(kernel="linear", C=0.025, probability=True)
#clf = SVC(kernel="linear", C=0.025)
clf = SVC(gamma=0.1, C=1)

print('\n================training=========================')
clf.fit(x_train, y_train)
print('\n rbf svm')
print('Linear SVM', ' train accuracy: ', accuracy_score(y_train, clf.predict(x_train)))
print('\n================predicting=========================')
test_predictions = clf.predict(x_test.transpose())
print('Linear SVM', ' test accuracy: ', accuracy_score(y_test, test_predictions))
print('Linear SVM', ' confusion matrix \n', confusion_matrix(y_test, test_predictions))

'''
predict_proba = clf.predict_proba(x_test.transpose())  #predicted probabilities
io.savemat('pred_prob_svm_sd_balanced.mat', mdict={'pred_prob_svm_sd_balanced': predict_proba})


fpr = []
tpr = []
thr = []
auc_scores = []

for i in range(0, num_classes):
    class_mask = np.zeros(total_mask.shape)
    class_mask[total_mask == i+1] = 1
    f, t, th = classify.prob2roc(predict_proba[:, i], class_mask.flat[np.flatnonzero(total_mask)])
    fpr.append(f)
    tpr.append(t)
    thr.append(th)
    auc_scores.append(auc(f, t))

# save a plot of all ROC curves
lw = 2
colors = ['red', 'blue', 'green', 'yellow', 'magenta']
class_names = ['blood', 'collagen', 'epithelium', 'myo', 'necrosis']
for i, name, color in zip(range(num_classes), class_names, colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (auc = {1:0.2f})'.format(name, auc_scores[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    #plt.show()

plt.savefig('roc_auc_svm_sd_balanced.png')

fpr = np.array(fpr)
tpr = np.array(tpr)
thr = np.array(thr)
auc_ = np.array(auc_scores)

io.savemat('fpr_svm_sd_balanced.mat', mdict={'fpr_svm_sd_balanced': fpr})
io.savemat('tpr_svm_sd_balanced.mat', mdict={'tpr_svm_sd_balanced': tpr})
io.savemat('auc_svm_sd_balanced.mat', mdict={'auc_svm_sd_balanced': auc_})
'''

