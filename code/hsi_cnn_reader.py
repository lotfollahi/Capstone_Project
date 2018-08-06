import spectral.io.envi as envi
from scipy.misc import imread
import os
import os.path as path
import numpy as np
import re
from math import floor
import utils


HDR_EXT = '.hdr'


class hsi_cnn_reader(object):
    '''
    Reads an envi file and returns a stack of cropped areas around each pixel given by a mask.
    '''

    # Parameters
    __crop_size = None
    __data_type = None
    __data_folder_name = None
    __masks_folder_name = None
    __num_samples = 0
    __balance = False
    __batch = False
    __npixels = None

    # Class variables
    __data_file_names = None
    __masks_file_names = None
    __cur_file_idx = -1
    __cur_data = None
    __masks = []
    __cur_shape = None
    __cur_item_idx = -1
    __cur_r = None
    __cur_c = None
    __num_masks = None
    __mask_idx = 0
    __samples_per_class = None
    __data_idx = 0

    def __init__(self,
                 data_folder_name,
                 masks_folder_name,
                 num_samples=None,
                 balance=False,
                 crop_size=(17, 17),  # (Height, Width)
                 data_type=np.float32,
                 npixels=None
                 ):

        self.__crop_size = crop_size
        self.__data_type = data_type
        self.__data_folder_name = data_folder_name
        self.__masks_folder_name = masks_folder_name
        self.__data_file_names = []
        self.__masks_file_names = []
        self.__num_samples = num_samples
        self.__balance = balance
        self.__masks = []
        self.__npixels = npixels

        # load file names
        self._load_file_names()

        # get
        # load masks
        self._load_mask_files()

    def __iter__(self):
        return self

    def __next__(self):

        record = self._next_record()
        if record:
            return record
        else:
            raise StopIteration()

    def _next_file(self):

        # read the next envi file in the folder
        self.__cur_file_idx += 1

        if self.__cur_file_idx == len(self.__data_file_names):
            return False

        file_name = self.__data_file_names[self.__cur_file_idx]
        file_name = path.join(self.__data_folder_name, file_name)

        # only load metadata
        self.__cur_data = envi.open(file_name + HDR_EXT, file_name)
        self.__cur_shape = self.__cur_data.shape

        return True

    def _load_file_names(self):
        '''
            # Loads the envi and mask file names.
        '''
        compiled = re.compile("\.")
        for f in os.listdir(self.__data_folder_name):
            if not f.startswith('.'):
                abs_path = path.join(self.__data_folder_name, f)
                file_name = re.split(compiled, abs_path)[0]
                if path.isfile(abs_path) and not (file_name in self.__data_file_names):
                    self.__data_file_names.append(file_name)

        if len(self.__data_file_names) < 1:
            raise ValueError('Missing data files in folder %s!' % self.__data_folder_name)

        file_list = []

        # get file names in masks directory sorted in alphabetical order
        for f in os.listdir(self.__masks_folder_name):
            file_list.append(f)

        file_list.sort()

        for f in file_list:
            if not f.startswith('.'):
                file_name = path.join(self.__masks_folder_name, f)
                if path.isfile(file_name) and not (file_name in self.__masks_file_names):
                    self.__masks_file_names.append(file_name)

        if len(self.__masks_file_names) < 1:
            raise ValueError('Missing masks files in folder %s!' % self.__data_folder_name)

        return True

    def _load_mask_files(self):
        '''
            # Load all mask files from the mask directory. 
        '''
        self.__num_masks = len(self.__masks_file_names)
        self.__samples_per_class = np.zeros(self.__num_masks, dtype=np.int32)
        for i in range(self.__num_masks):
            file_name = self.__masks_file_names[i]
            temp = imread(file_name, flatten=True)
            temp = np.divide(temp, np.amax(temp))
            # temp = np.multiply(temp, i+1)
            self.__masks.append(temp)  # array consisting of all masks, size rows x cols x # of masks
            if self.__num_samples is None:
                self.__samples_per_class[i] = np.count_nonzero(temp)
            else:
                # if user has specified a max number of samples per class
                if self.__num_samples > np.count_nonzero(temp):
                    self.__samples_per_class[i] = np.count_nonzero(temp)
                else:
                    self.__samples_per_class[i] = self.__num_samples

        self.__masks = np.asarray(self.__masks)
        self.__masks = np.transpose(self.__masks, (1, 2, 0))
        return True

    def _next_record(self):
        '''
            # Read and return the cropped samples corresponding to pixels for each mask.
            # Input: envoked in a loop on an hsi_reader object.
            # Output: -array of samples of size: number of masked pixels x crop_size x crop_size x bands
            #         -list of labels of size: 1 x pixels
            #         -number of read samples
            #         -index of read pixels
        '''
        if utils.is_empty(self.__cur_data) or \
                (self.__mask_idx == self.__num_masks):

            if not self._next_file():
                return None

            self.__mask_idx = 0
            self.__data_idx == 0

        if self.__npixels:
            return self._loadbatch()
        else:
            if self.__balance:
                return self._load_balanced_data()
            else:
                return self._load_data()

    def _load_data(self):
        '''
        Load data by cropping around each pixel for each mask.

        @return:
            # input_ - array of size num_samples x crop_size x crop_size x num_bands
            # labels - vector of labels for the loaded samples
        '''

        # Crop area around a pixel corresponding to a particular mask
        idx = np.transpose(np.nonzero(self.__masks[:, :, self.__mask_idx]))
        input_ = []
        labels = []
        total_idx = []  # all the loaded indices

        if self.__num_samples is not None:
            # use specific number of samples for training
            np.random.shuffle(idx)
            idx = idx[0:self.__samples_per_class[self.__mask_idx], :]

        k = 0

        l_idx = []  # indices of the loaded pixels

        # read all masked pixels
        for (r, c) in idx:
            r_begin = r - floor(self.__crop_size[0] / 2.0)
            c_begin = c - floor(self.__crop_size[0] / 2.0)
            r_end = r_begin + self.__crop_size[0]
            c_end = c_begin + self.__crop_size[1]
            # thresh=0.9
            # img_size = int(self.__crop_size[0] * self.__crop_size[1]*thresh)
            # print('\n\t img_size: ', img_size)

            if r_begin >= 0 and c_begin >= 0:
                if r_end <= self.__cur_data.shape[0] and c_end <= self.__cur_data.shape[1]:
                    # temp = self.__cur_data[r_begin:r_end,
                    #             c_begin:c_end,
                    #              ...]

                    # if np.count_nonzero(np.sum(temp, axis=2)) > img_size:
                    input_.append(self.__cur_data[r_begin:r_end,
                                  c_begin:c_end, :])

                    labels.append(self.__mask_idx)
                    l_idx.append(k)
            k += 1
            # else:
            # print('\n\t too much background\n')
            # print('from next record: ', len(labels))
        # keep only the indices of pixels which where loaded
        idx = idx[l_idx, :]
        self.__mask_idx += 1

        return np.asarray(input_), labels, idx

    def _load_balanced_data(self):
        '''
        Load a balanced data set for a particular class.

        @return:
            # input_ - array of size num_samples x crop_size x crop_size x num_bands
            # labels - vector of labels for the loaded samples
        '''

        # Crop area around a pixel corresponding to a particular mask
        idx = np.transpose(np.nonzero(self.__masks[:, :, self.__mask_idx]))
        input_ = []
        labels = []
        total_idx = []  # all the loaded indices

        if self.__num_samples is not None:
            # use specific number of samples for training
            np.random.shuffle(idx)
            idx = idx[0:self.__samples_per_class[self.__mask_idx], :]

        # increase number of samples by copying them over multiple times
        max_samples = np.amax(self.__samples_per_class)
        copy_times = int(floor(
            max_samples / self.__samples_per_class[self.__mask_idx]))  # num of times to copy for even division
        rem = max_samples % self.__samples_per_class[self.__mask_idx]  # remaining samples

        for i in range(0, copy_times):
            l_idx = []  # indices of the loaded pixels
            np.random.shuffle(idx)
            k = 0
            for (r, c) in idx:
                r_begin = r - floor(self.__crop_size[0] / 2.0)
                c_begin = c - floor(self.__crop_size[0] / 2.0)
                r_end = r_begin + self.__crop_size[0]
                c_end = c_begin + self.__crop_size[1]

                if r_begin >= 0 and c_begin >= 0:
                    if r_end <= self.__cur_data.shape[0] and c_end <= self.__cur_data.shape[1]:
                        input_.append(self.__cur_data[r_begin:r_end,
                                      c_begin:c_end,
                                      :]
                                      )
                        labels.append(self.__mask_idx)
                        l_idx.append(k)  # keep track of the loaded indices
                k += 1
            total_idx.append(idx[l_idx, :])  # save loaded indices

        k = 0

        # copy the remaning samples so the total matches the max number
        # of samples chosen by user
        if rem > 0:
            l_idx = []  # indices of the loaded pixels
            np.random.shuffle(idx)
            idx = idx[0:rem, :]

            for (r, c) in idx:
                r_begin = r - floor(self.__crop_size[0] / 2.0)
                c_begin = c - floor(self.__crop_size[0] / 2.0)
                r_end = r_begin + self.__crop_size[0]
                c_end = c_begin + self.__crop_size[1]

                if r_begin >= 0 and c_begin >= 0:
                    if r_end <= self.__cur_data.shape[0] and c_end <= self.__cur_data.shape[1]:
                        input_.append(self.__cur_data[r_begin:r_end,
                                      c_begin:c_end,
                                      :]
                                      )
                        labels.append(self.__mask_idx)
                        l_idx.append(k)  # keep track of the loaded indices
                k += 1
            total_idx.append(idx[l_idx, :])  # save loaded indices

        self.__mask_idx += 1

        return np.asarray(input_, dtype=np.float32), labels, np.asarray(total_idx)

    def _loadbatch(self):
        '''
            # Load batches of data -- can be used for classifying large size images that don't fit on memory
            # Input: 
            #       - envoked from an object of type hsi_cnn_reader 
            #       - it reads data on an iterator loop as an hsi_reader object.
            #       - it iterates over the mask and over batches consisting of number of pixels specified in the
            #           --npixels command line parameter
            # Output: 
            #       -array of samples of size: number pixels (per batch) x crop_size x crop_size x bands
            #       -list of labels of size: 1 x npixels
            #       -number of read samples
            #       -index of read pixels
        
        :return: 
        '''
        idx = np.transpose(np.nonzero(self.__masks[:, :, self.__mask_idx]))  # get the row, col indices of valid pixels

        l_idx = []  # indices of the loaded pixels

        k = 0
        input_ = []
        labels = []

        npixels = min(self.__npixels,
                      len(idx) - self.__data_idx)  # if there aren't enough pixels, change the batch size
        idx_chunk = idx[self.__data_idx:self.__data_idx + npixels, :]

        # load cropped regions around each pixel
        for (r, c) in idx_chunk:
            r_begin = r - floor(self.__crop_size[0] / 2.0)
            c_begin = c - floor(self.__crop_size[0] / 2.0)
            r_end = r_begin + self.__crop_size[0]
            c_end = c_begin + self.__crop_size[1]

            if r_begin >= 0 and c_begin >= 0 and r_end <= self.__cur_data.shape[0] and c_end <= self.__cur_data.shape[
                1]:
                input_.append(self.__cur_data[r_begin:r_end,
                              c_begin:c_end, :])
                labels.append(self.__mask_idx)
                l_idx.append(k)
            k += 1

        # keep only the indices of pixels which where loaded
        idx_chunk = idx_chunk[l_idx, :]
        self.__data_idx += npixels

        # keep only the indices of pixels which where loaded
        # idx = idx[l_idx, :]

        if len(idx) == self.__data_idx:  # if all of the pixels have been read in this mask, continue with the next mask
            self.__mask_idx += 1
            self.__data_idx = 0

        return np.asarray(input_, dtype=np.float32), labels, idx_chunk

    def data_dims(self):
        '''
            # Returns the predicted number of samples and the number of bands.
        '''

        # these would be used to take care of boundary conditions
        h_rep = 0
        w_rep = 0

        rows = self.__masks.shape[0] + h_rep
        cols = self.__masks.shape[1] + w_rep

        annotated_pixels = np.zeros((self.__masks.shape[0], self.__masks.shape[1]))
        for i in range(self.__num_masks):
            annotated_pixels += self.__masks[:, :, i]

        annotated_pixels[annotated_pixels > 1] = 0
        num_samples = 0

        if self.__balance:
            if self.__num_samples is None:

                max_samples = np.amax(self.__samples_per_class)

                for i in range(0, self.__num_masks):
                    num_samples += self.__samples_per_class[i] * int(
                        floor(max_samples / self.__samples_per_class[i]))
                    num_samples += max_samples % self.__samples_per_class[i]  # add remaining samples
            else:
                num_samples = 0

                max_samples = np.amax(self.__samples_per_class)

                for i in range(0, self.__num_masks):
                    num_samples += self.__samples_per_class[i] * int(
                        floor(max_samples / self.__samples_per_class[i]))
                    num_samples += max_samples % self.__samples_per_class[i]  # add remaining samples

        elif self.__num_samples is None:

            num_samples = np.count_nonzero(annotated_pixels)
            # print('\n========>num_samples: ', num_samples)
        else:
            for i in range(0, self.__num_masks):
                if self.__num_samples > np.count_nonzero(self.__masks[:, :, i]):
                    num_samples += np.count_nonzero(self.__masks[:, :, i])
                else:
                    num_samples += self.__num_samples
                    # print('balancing -- num-samples: ', num_samples)

        # assuming a reader object has been initialized and filenames has been loaded
        file_name = self.__data_file_names[0]
        file_name = path.join(self.__data_folder_name, file_name)

        # only load metadata
        cur_data = envi.open(file_name + HDR_EXT, file_name)

        return num_samples, cur_data.nbands, rows, cols
