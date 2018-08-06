import sys
sys.path.append('/home/sberisha/source/stimlib/python')
import envi
import numpy
import sklearn.neural_network
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import classify


data_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/no-mnf/new-cnn/'
masks_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/masks/no-mnf-bcemn/'

classimages = sorted(glob.glob(masks_path + '*.png'))                   #load the class file names
C = classify.filenames2class(classimages)                               #generate the class images for training


Etrain = envi.envi(data_path + 'br1003-br2085b-bas-nor-fin-bip-pca16')  #open the ENVI file for reading, use the validation image for batch access
Ftrain, Ttrain = Etrain.loadtrain(C)
#Ftrain, Ttrain = Etrain.load_train_balance(C, 60000)  #load the training set

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10)
rf_model.fit(Ftrain, Ttrain)

#CLASS1 = sklearn.naive_bayes.GaussianNB()
#train a Bayesian classifier
#CLASS1.fit(Ft1, Tt1)
Etrain.close()

###########################################testing###############################################
data_path = '/media/stim-processed/berisha/breast-processing/lm/br1003/no-mnf/brc961-proj/brc961-proj/new-cnn/'
masks_path = '/media/stim-processed/berisha/breast-processing/lm/brc961/masks/no-mnf-bcemn/'

classimages = sorted(glob.glob(masks_path + '*.png'))   #load the class file names
C = classify.filenames2class(classimages)   #generate the class images for testing
C = C.astype(numpy.uint32)

bool_mask = numpy.sum(C.astype(numpy.uint32), 0)

# get number of classes
num_classes = C.shape[0]

for i in range(1, num_classes):
    C[i,:,:] *= i+1

total_mask = numpy.sum(C.astype(numpy.uint32), 0) # validation mask

test_set = envi.envi(data_path + 'brc961-br1001-bas-nor-fin-bip-proj-br1003-pca16', mask=total_mask)

#plt.ion()
N = numpy.count_nonzero(total_mask)                    #set the batch size
Tv = []                                                             #initialize the target array to empty
Fv = test_set.loadbatch(N)
Tv = total_mask.flat[numpy.flatnonzero(total_mask)]                                     #get the indices of valid pixels

predictions = rf_model.predict(Fv.transpose())
predict_proba = rf_model.predict_proba(Fv.transpose())

print("Train Accuracy :: ", accuracy_score(Ttrain, rf_model.predict(Ftrain)))
print("Test Accuracy  :: ", accuracy_score(Tv, predictions))
print(" Confusion matrix ", confusion_matrix(Tv, predictions))

fpr = []
tpr = []
thr = []
auc = []

for i in range(0, num_classes):
    class_mask = numpy.zeros(total_mask.shape)
    class_mask[total_mask == i+1] = 1;
    f, t, th = classify.prob2roc(predict_proba[:,i], class_mask.flat[numpy.flatnonzero(total_mask)])
    fpr.append(f)
    tpr.append(t)
    thr.append(th)
    auc.append(sklearn.metrics.auc(f, t))

fpr = numpy.array(fpr)
tpr = numpy.array(tpr)
thr = numpy.array(thr)
#auc = numpy.array(auc)
print('\n\n AUC: ', auc)

'''
k = 0#load the first batch
while not Fv == []:                                                             #loop until an empty batch is returned
    Tv = numpy.append(Tv, CLASS1.predict(Fv.transpose()))                        #append the predicted labels from this batch to those of previous batches
    Lv = hyperspectral.unsift2(Tv, test_set.batchmask())                                    #convert the matrix of class labels to a 2D array
    Cv = classify.label2class(Lv[0, :, :], background=0)                        #convert the labels to a binary class image
    RGB1 = classify.class2color(Cv)                                              #convert the binary class image to a color image
    plt.imshow(RGB1)                                                             #display it
    plt.pause(0.05)
    Fv = test_set.loadbatch(N)
    k += 1#load the next batch
    print('\n\t k: ', k)
test_set.close()
plt.imsave(data_path + 'hd-bayesian.png', RGB1)
'''
