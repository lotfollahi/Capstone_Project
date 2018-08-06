import numpy as np
import os
from tflearn.data_utils import to_categorical
from hsi_cnn_reader import hsi_cnn_reader
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
#import classify
#import matplotlib.pyplot as plt


def envi_dims(data_path, masks_path):
    # create the reader object
    reader = hsi_cnn_reader(data_path,
                            masks_path,
                            )

    return reader.data_dims()


def cnn_metrics(data_path, masks_path, crop_size, num_classes, model):

    # create the reader object
    reader = hsi_cnn_reader(data_path,
                            masks_path,
                            crop_size=(crop_size, crop_size),
                            )

    num_samples, num_bands, rows, cols = reader.data_dims()

    # compute and the confusion matrix and response array file
    envi_probs = np.zeros((rows, cols, num_classes), dtype=np.float32)
    y_true = []
    y_pred = []
    i = 0
    pbar = ProgressBar(maxval=num_samples).start()
    k = 0
    print('\n Total number of pixels to classify: ', num_samples, '\n')

    for (input_, labels, idx) in reader:
        if len(input_) > 0:
            i = 0
            y_true.append(labels)
            for j in idx:
                temp = np.expand_dims(input_[i, :, :, :], 0)

                # Run the model on one example
                prediction = model.predict(temp)
                envi_probs[j[0], j[1], :] = np.asarray(prediction)
                y_pred.append(np.argmax(prediction))
                i += 1
                k += 1
                pbar.update(k)

    y_true_flat = [item for sublist in y_true for item in sublist]

    conf_mat = confusion_matrix(y_true_flat, y_pred)

    return envi_probs, conf_mat


def cnn_classify_batch(data_path, masks_path, crop_size, num_classes, model, npixels):

    # create the reader object
    reader = hsi_cnn_reader(data_path,
                            masks_path,
                            crop_size=(crop_size, crop_size),
                            npixels=npixels
                            )

    num_samples, num_bands, rows, cols = reader.data_dims()

    # compute and the confusion matrix and response array file
    envi_probs = np.zeros((rows, cols, num_classes), dtype=np.float32)
    pbar = ProgressBar(maxval=num_samples).start()
    k = 0
    print('\n Total number of pixels to classify: ', num_samples, '\n')
    # plt.ion()
    y_true = []
    y_pred = []

    for (input_, labels, idx) in reader:
        if len(input_) > 0:
            y_true.append(labels)
            # Run the model
            prediction = model.predict(input_)
            envi_probs[idx[:, 0], idx[:, 1], :] = np.asarray(prediction)
            #class_image = classify.prob2class(np.rollaxis(envi_probs, 2, 0))
            #rgb = classify.class2color(class_image)
            #plt.imshow(rgb)
            #plt.pause(0.05)
            #print('\n\t k: ', k)
            y_pred.append(np.argmax(prediction, 1))
            k += len(input_)
            pbar.update(k)
    
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    conf_mat = confusion_matrix(y_true_flat, y_pred_flat)

    return envi_probs, conf_mat


def chp_folder(folder_path):
    """
    Create checkpoint folder if it doesn't exist. Otherwise delete folder contents.

    @param folder_path: Path to directory - including directory name.
    @return: True if successful.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    # create checkpoint dir if it doesn't exist
    else:
        # delete contents of folder
        for the_file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    # elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    return True


def load_data(data_path, masks_path, crop_size, num_classes, samples=None, balance=False):
    # Load CNN data in a format readable by tflearn.
    print('\n\n balance: ', balance)
    
    # create the reader object
    reader = hsi_cnn_reader(data_path,
                            masks_path,
                            num_samples=samples,
                            balance=balance,
                            crop_size=(crop_size, crop_size),
                            )

    num_samples, num_bands, _, _ = reader.data_dims()
    print('\n number of samples: ', num_samples, '\n')

    X = np.zeros((num_samples, crop_size, crop_size, num_bands), dtype=np.float32)

    Y = []
    i = 0
    sidx = 0
    total_samples = 0
    samples_per_class = np.zeros((num_classes, 1))

    # load  data
    for (input_, labels, _) in reader:
        if input_ is not None:
            num_samples = input_.shape[0]
            X[sidx:sidx + num_samples, :, :, :] = input_
            sidx = sidx + num_samples
            # X.append(input_)
            Y.append(labels)
            samples_per_class[i] = num_samples
            i += 1
            total_samples += num_samples
            print('loading class: ', i, ' - samples: ', num_samples)

    X = X[0:total_samples, :, :, :]

    Y = np.concatenate([np.asarray(i) for i in Y])
    Y = to_categorical(Y, num_classes).astype(np.int32)

    return X, Y, num_bands


def is_empty(o):
    if isinstance(o, (tuple, list)):
        if len(o) < 1:
            return True
        else:
            return False
    elif isinstance(o, np.ndarray):
        if o.size < 1:
            return True
        else:
            return False
    elif isinstance(o, object):
        if o is None:
            return True
        else:
            return False
    elif not o:
        return True
    return False
