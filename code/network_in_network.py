'''
    # CNN model for FTIR data.
'''
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization, batch_normalization
import time

def build_net(network, X, Y, num_classes, num_epochs, checkpoint_path, size_batch, Xval=None, Yval=None, dec_step=100,
              train=True):
    tn = tflearn.initializations.truncated_normal(seed=100)
    xav = tflearn.initializations.xavier(seed=100)
    nor = tflearn.initializations.normal(seed=100)

    network = conv_2d(network, 192, 5, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, 160, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, 96, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    network = max_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)

    network = conv_2d(network, 192, 5, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, 192, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, 192, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    network = avg_pool_2d(network, 3, strides=2)
    network = dropout(network, 0.5)

    network = conv_2d(network, 192, 3, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, 192, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)
    
    network = conv_2d(network, num_classes, 1, weights_init=nor, regularizer='L2')
    network = batch_normalization(network)
    network = tflearn.activations.softplus(network)

    network = avg_pool_2d(network, 9)
    network = flatten(network)
    
    adadelta = tflearn.optimizers.AdaDelta(learning_rate=0.01, rho=0.95, epsilon=1e-08)

    network = regression(network, optimizer=adadelta,
                         loss='softmax_categorical_crossentropy'
                         )

    # Train
    model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path=checkpoint_path)
    if train:
        start_time = time.time()
        if Xval is None or Yval is None:
            model.fit(X, Y, n_epoch=num_epochs,
                      validation_set=0.0,
                      show_metric=True, run_id='hsi_cnn_model', shuffle=True, batch_size=size_batch)
        else:
            model.fit(X, Y, n_epoch=num_epochs,
                      validation_set=(Xval, Yval),
                      show_metric=True, run_id='hsi_cnn_model', shuffle=True, batch_size=size_batch)

        print("\n\n-------------train time: %s seconds\n\n" % (time.time() - start_time))

    return model

