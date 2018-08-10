# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 16:04:33 2017

@author: david
"""

import numpy
import colorsys
import sklearn
import sklearn.metrics
import scipy
import scipy.misc
import envi
import hyperspectral
import random
import progressbar
import matplotlib.pyplot as plt

#generate N qualitative colors and return the value for color c
def qualcolor(c, N):
    dN = numpy.ceil(numpy.sqrt(N)).astype(numpy.int32)
    h = c/N
    
    sp = c/N * 2 * numpy.pi * dN + numpy.pi/2
    s = numpy.sin(sp) * 0.25 + 0.75
    
    vp = c/N * 2 * numpy.pi * dN
    v = numpy.sin(vp) * 0.25 + 0.75
    
    rgb = numpy.array(colorsys.hsv_to_rgb(h, s, v))
    return rgb * 255

#generate a 2D color class map using a stack of binary class images
#input: C is a C x Y x X binary image
#output: an RGB color image with a unique color for each class
def class2color(C):
    
    #determine the number of classes
    nc = C.shape[0]
    
    s = C.shape[1:]
    s = numpy.append(s, 3)

    #generate an RGB image
    RGB = numpy.zeros(s, dtype=numpy.ubyte)
    
    #for each class
    for c in range(0, nc):
        color = qualcolor(c, nc)
        RGB[C[c, ...], :] = color
    
    return RGB

#create a function that loads a set of class images as a stack of binary masks
#input: list of class image names
#output: C x Y x X binary image specifying class/pixel membership
#example: image2class(("class_coll.bmp", "class_epith.bmp"))
def filenames2class(masks):
    #get num of mask file names
    num_masks = len(masks)

    if num_masks == 0:
        print("ERROR: mask filenames not provided")
        print("Usage example: image2class(('class_coll.bmp', 'class_epith.bmp'))")
        return

    classimages = []
    bar = progressbar.ProgressBar(max_value=num_masks)
    for m in range(0, num_masks):
        img = scipy.misc.imread(masks[m], flatten=True).astype(numpy.bool)
        classimages.append(img)
        bar.update(m+1)

    result = numpy.stack(classimages)
    sum_images = numpy.sum(result.astype(numpy.uint32), 0)

    #identify and remove redundant pixels
    bad_idx = sum_images > 1
    result[:, bad_idx] = 0

    return result


#create a class mask stack from an C x Y x X probability image
#input: C x Y x X image giving the probability P(c |x,y)
#output: C x Y x X binary class image
def prob2class(prob_image):
    class_image = numpy.zeros(prob_image.shape, dtype=numpy.bool)
    #get nonzero indices
    nnz_idx = numpy.transpose(numpy.nonzero(numpy.sum(prob_image, axis=0)))
    
    #set pixel corresponding to max probability to 1
    for idx in nnz_idx:
        idx_max_prob = numpy.argmax(prob_image[:, idx[0], idx[1]])
        class_image[idx_max_prob, idx[0], idx[1]] = 1

    return class_image

#calculate an ROC curve given a probability image and mask of "True" values
#input:
#       P is a Y x X probability image specifying P(c | x,y)
#       t_vals is a Y x X binary image specifying points where x,y = c
#       mask is a mask specifying all pixels to be considered (positives and negatives)
#           use this mask to limit analysis to regions of the image that have been classified
#output: fpr, tpr, thresholds
#       fpr is the false-positive rate (x-axis of an ROC curve)
#       tpr is the true-positive rate (y-axis of an ROC curve)
#       thresholds stores the threshold associated with each point on the ROC curve
#
#note: the AUC can be calculated as auc = sklearn.metrics.auc(fpr, tpr)
def prob2roc(P, t_vals, mask=[]):
    
    if not P.shape == t_vals.shape:
        print("ERROR: the probability and mask images must be the same shape")
        return
    
    #if a mask image isn't provided, create one for the entire image
    if mask == []:
        mask = numpy.ones(t_vals.shape, dtype=numpy.bool)
    
    #create masks for the positive and negative probability scores
    mask_p = t_vals
    mask_n = mask - mask * t_vals
    
    #calculate the indices for the positive and negative scores
    idx_p = numpy.nonzero(mask_p)
    idx_n = numpy.nonzero(mask_n)
    
    Pp = P[idx_p]
    Pn = P[idx_n]

    Lp = numpy.ones((Pp.shape), dtype=numpy.bool)
    Ln = numpy.zeros((Pn.shape), dtype=numpy.bool)
    
    scores = numpy.concatenate((Pp, Pn))
    labels = numpy.concatenate((Lp, Ln))
    
    return sklearn.metrics.roc_curve(labels, scores)

#convert a label image to a C x Y x X class image
def label2class(L, background=[]):
    unique = numpy.unique(L)

    if not background == []:                                                #if a background value is specified
        unique = numpy.delete(unique, numpy.nonzero(unique == background))  #remove it from the label array
    s = L.shape
    s = numpy.append(numpy.array((len(unique))), s)
    C = numpy.zeros(s, dtype=numpy.bool)
    for i in range(0, len(unique)):
        C[i, :, :] = L == unique[i]
    return C

#randomizes a given mask to include a subset of n pixels in the original
def random_mask(M, n):
    idx = numpy.flatnonzero(M)
    new_idx = numpy.random.permutation(idx)
    new_mask = numpy.zeros(M.shape, dtype=numpy.bool)
    new_mask[numpy.unravel_index(new_idx[0:n], new_mask.shape)] = True
    return new_mask

#perform classification of an ENVI image using batch processing
# input:    E is the ENVI object (file is assumed to be loaded)
#           C is a classifier - anything in sklearn should work
#           batch is the batch size
def envi_batch_predict(E, C, batch=10000):

    Fv = E.loadbatch(batch)
    i = 0
    Tv = []
    plt.ion()
    bar = progressbar.ProgressBar(max_value=numpy.count_nonzero(E.mask))
    while not Fv == []:
        Fv = numpy.nan_to_num(Fv)                                                     #remove infinite values        
        if i == 0:
            Tv = C.predict(Fv.transpose())
        else:
            Tv = numpy.concatenate((Tv, C.predict(Fv.transpose()).transpose()), 0)
        tempmask = E.batchmask()
        Lv = hyperspectral.unsift2(Tv, tempmask)
        Cv = label2class(Lv.squeeze(), background=0)
        RGB = class2color(Cv)
        plt.imshow(RGB)
        plt.pause(0.05)
        Fv = E.loadbatch(batch)   
        i = i + 1
        bar.update(len(Tv))
