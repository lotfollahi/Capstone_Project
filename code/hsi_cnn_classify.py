import numpy as np
from tflearn.layers.core import input_data
from tflearn.data_preprocessing import ImagePreprocessing
import argparse
import hsi_cnn_model
from scipy import io
import matplotlib.pyplot as plt
import spectral.io.envi as envi
import time
import utils
import scipy.misc
import classify
import alexnet
import network_in_network
import googlenet
import vgg19
import vgg_network

'''
    # Script to classify FTIR HSI data using CNN.
    # Example usage:
        -for overall accuracy:
            python hsi_cnn_classify.py --data /media/buffer/berisha/cnn-hsi/lm/br1003/no-mnf/brc961-proj/new-cnn/ 
                                   --masks /media/stim-processed/berisha/breast-processing/lm/brc961/masks/no-mnf-bcemn/ 
                                   --checkpoint /media/buffer/berisha/cnn-hsi/chp/tmp/ 
                                   --crop 17 
                                   --classes 5 
                                   --batch 128 
        -for metrics:
            python hsi_cnn_classify.py --data /media/buffer/berisha/cnn-hsi/lm/br1003/no-mnf/brc961-proj/new-cnn/ 
                                   --masks /media/stim-processed/berisha/breast-processing/lm/brc961/masks/no-mnf-bcemn/ 
                                   --checkpoint /media/buffer/berisha/cnn-hsi/chp/tmp/ 
                                   --crop 17 
                                   --classes 5 
                                   --batch 128 
                                   --metrics cnn
                                   --npixels 10000 -> this is an optional param for loading data in batches of pixels
'''

################ read command line arguments####################################################
parser = argparse.ArgumentParser(description="Validate a CNN model on FTIR HSI data -- \
                                             the input samples are cropped patches around each pixel")

#required args
required = parser.add_argument_group('required named arguments')
required.add_argument("--data", help="Path to train data folder.", type=str)
required.add_argument("--masks", help="Path to train masks folder.", type=str)
required.add_argument("--checkpoint", help="Path to checkpoint directory.", type=str)
required.add_argument("--crop", help="Crop size", type=int)
required.add_argument("--classes", help="Num of classes.", type=int)
required.add_argument("--batch", help="Batch size to use for training.", type=int)

# optional arguments
parser.add_argument("--metrics",
                    help="compute and save metrics, given a prefix name after --metrics to specify \
                    the prefix of saved files", default=False, nargs='?', type=str)
parser.add_argument("--npixels",
                    help="batch size for classifying data batch-wise, after --bach give number of pixels per batch  \
                    (e.g. 1000)", default=0, nargs='?', type=int)

args = parser.parse_args()

############################# Real-time data preprocessing ##################
img_prep = ImagePreprocessing()

img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

num_samples, num_bands, _, _ = utils.envi_dims(args.data, args.masks)

####### set up network #######

network = input_data(shape=[None, args.crop, args.crop, num_bands],
                     data_preprocessing=img_prep)

'''
 # needed for 1d cnn
network = input_data(shape=[None, 1, num_bands, 1],
                         data_preprocessing=img_prep,
                         name='input')
'''
print('\n ======= classifying using hsi_cnn_model =========\n')

model = hsi_cnn_model.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )
'''
model = alexnet.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )

model = network_in_network.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )

model = googlenet.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )

model = vgg19.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )

model = vgg_network.build_net(network,
                                None,
                                None,
                                args.classes,
                                None,
                                args.checkpoint,
                                args.batch,
                                train=False,
                                )
'''
model.load(args.checkpoint)

total_samples = 0
if args.npixels > 0 and args.metrics:
    print('\n\n ... classifying data using a load batch size of: ', args.npixels, ' pixels')
    probs, conf_mat = utils.cnn_classify_batch(args.data, args.masks, args.crop, args.classes, model, args.npixels)
    # save response array as envi file
    envi.save_image(args.metrics + '-response.hdr', probs, dtype=np.float32, interleave='bsq', force=True)

    print('\n Confusion matrix \n', conf_mat)
    np.savetxt(args.metrics + '-conf-mat', conf_mat, delimiter=",", fmt="%1f")
    oa = np.trace(conf_mat)/np.sum(conf_mat)
    print('\n\t==================>OA: %0.2f%%' % (oa * 100))

    # save a classified image
    class_image = classify.prob2class(np.rollaxis(probs, 2, 0))
    rgb = classify.class2color(class_image)
    scipy.misc.imsave(args.metrics + '-classified-image.png', rgb)

elif args.metrics:
    print('\n\n ... computing cnn metrics \n\n')
    probs, conf_mat = utils.cnn_metrics(args.data, args.masks, args.crop, args.classes, model)

    # save response array as envi file
    envi.save_image(args.metrics + '-response.hdr', probs, dtype=np.float32, interleave='bsq', force=True)
    print('\n Confusion matrix \n', conf_mat)
    np.savetxt(args.metrics + '-conf-mat', conf_mat, delimiter=",", fmt="%1f")
else:
    print('\n\n ... computing cnn overall accuracy \n\n')
    # get overall accuracy
    X, Y, num_bands = utils.load_data(args.data, args.masks, args.crop, args.classes)
    
    #X = np.rollaxis(X, 3, 2) # needed for 1d cnn

    # Evaluate model
    start_time = time.time()
    score = model.evaluate(X, Y)
    print("\n\n-------------test time: %s seconds\n\n" % (time.time() - start_time))

    print('\n\t==================>Test accuracy: %0.2f%%' % (score[0] * 100))
