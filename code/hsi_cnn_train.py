import numpy as np
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import argparse
from tflearn.layers.core import input_data
import hsi_cnn_model
import utils
from tflearn.data_utils import to_categorical
#import convnet_cifar10
import vgg_network
import alexnet
import network_in_network
import vgg19
import googlenet
import vgg_network

'''
    # Train CNN model on HSI data.
    # Example usage: 
        python hsi_cnn_train.py --data /media/buffer/berisha/cnn-hsi/lm/br1003/no-mnf/new-cnn/ 
                                --masks /media/stim-processed/berisha/breast-processing/lm/br1003/masks/no-mnf-bcemn/ 
                                --checkpoint /media/buffer/berisha/cnn-hsi/chp/tmp/ 
                                --crop 17 
                                --classes 5 
                                --epochs 8 
                                --batch 128 
                                --balance 
                                --samples 60000 
                                --validate 
                                --valdata /media/buffer/berisha/cnn-hsi/lm/br1003/no-mnf/brc961-proj/new-cnn/ 
                                --valmasks /media/stim-processed/berisha/breast-processing/lm/brc961/masks/all-corrected-masks/ 
                                --valsamples 20000
'''

################ read command line arguments####################################################
parser = argparse.ArgumentParser(description="Train a CNN model on FTIR HSI data -- \
                                             the input samples are cropped patches around each pixel")

#required args
required = parser.add_argument_group('required named arguments')
required.add_argument("--data", help="Path to train data folder.", type=str)
required.add_argument("--masks", help="Path to train masks folder.", type=str)
required.add_argument("--checkpoint", help="Path to checkpoint directory.", type=str)
required.add_argument("--crop", help="Crop size", type=int)
required.add_argument("--classes", help="Num of classes.", type=int)
required.add_argument("--epochs", help="Num of epochs to use for training.", type=int)
required.add_argument("--batch", help="Batch size to use for training.", type=int)

# optional arguments
parser.add_argument("--balance", help="Balance number of training samples for each class", action="store_true")
parser.add_argument("--samples", help="Num of train samples per class.", type=int)
parser.add_argument("--validate", help="validate on new set", action="store_true")
parser.add_argument("--valdata", help="Path to validation data folder.", type=str)
parser.add_argument("--valmasks", help="Path to validation masks folder.", type=str)
parser.add_argument("--valbalance", help="Balance number of validation samples for each class.", action="store_true")
parser.add_argument("--valsamples", help="Num of validation samples per class.", type=int)

args = parser.parse_args()

utils.chp_folder(args.checkpoint)   # delete contents of checkpoint folder if it exists

print('\n.............loading training data...........\n')
X,Y, num_bands = utils.load_data(args.data, args.masks, args.crop, args.classes, samples=args.samples, balance=args.balance)
print('\n..............done loading training data.........\n')
###############################################################################
# load new validation set
###############################################################################

if args.validate:
    # load validation data from different envi file
    print('\n...........loading validation data............\n')
    X_val,Y_val, num_bands = utils.load_data(args.valdata, args.valmasks, args.crop, args.classes, samples=args.valsamples, balance=args.valbalance)
    print('\n...........done loading valdiation data......\n')
else:
    # validate on a subset of training data
    X_val = None
    Y_val = None


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()

# set up network

network = input_data(shape=[None, args.crop, args.crop, num_bands],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug,
                     name='input')
'''
 # necessary for 1-d cnn
X = np.rollaxis(X, 3, 2)
X_val = np.rollaxis(X_val, 3, 2)

network = input_data(shape=[None, 1, num_bands, 1],
                         data_preprocessing=img_prep,
                         name='input')
'''

print('============ training with hsi_cnn_model==========\n')

model = hsi_cnn_model.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)


'''
model = alexnet.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)


model = network_in_network.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)

model = vgg19.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)

model = googlenet.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)


model = vgg_network.build_net(network,
                                X,
                                Y,
                                args.classes,
                                args.epochs,
                                args.checkpoint,
                                args.batch,
                                Xval=X_val,
                                Yval=Y_val)
'''
model.save(args.checkpoint)

print('\n\t Training done.')
