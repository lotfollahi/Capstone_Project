import spectral.io.envi as envi
from scipy.misc import imread
import os
import os.path as path
import numpy as np
import re
import utils
from math import floor

HDR_EXT = '.bin.hdr'
FILE_EXT = '.bin'
MASK_EXT = '.png'


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

    def __init__(self,
                 data_folder_name,
                 masks_folder_name,
                 num_samples=0,
                 balance=False,
                 crop_size=(32, 32),  # (Height, Width)
                 data_type=np.float32):

        self.__crop_size = crop_size
        self.__data_type = data_type
        self.__data_folder_name = data_folder_name
        self.__masks_folder_name = masks_folder_name
        self.__data_file_names = []
        self.__masks_file_names = []
        self.__num_samples = num_samples
        self.__balance = balance
        self.__masks = []

        # load file names
        self._load_file_names()

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
        self.__cur_data = envi.open(file_name + HDR_EXT, file_name + FILE_EXT)

        self.__cur_shape = self.__cur_data.shape

        return True

    def _load_file_names(self):
        '''
            # Loads the envi and mask file names from the data folder.
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
        self.__samples_per_class = np.zeros((self.__num_masks, 1), dtype=np.int32)
        for i in range(self.__num_masks):
            file_name = self.__masks_file_names[i]
            temp = imread(file_name, flatten=True)
            temp = np.divide(temp, np.amax(temp))
            # temp = np.multiply(temp, i+1)
            self.__masks.append(temp)
            if self.__num_samples == 0:
                self.__samples_per_class[i] = np.count_nonzero(temp)

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

        # Crop area around a pixel corresponding to a particular mask
        idx = np.transpose(np.nonzero(self.__masks[:, :, self.__mask_idx]))
        input_ = []
        labels = []

        if self.__num_samples > 0:
            # use specific number of samples for training
            ns = self.__num_samples
            if self.__num_samples > np.count_nonzero(self.__masks[:, :, self.__mask_idx]):
                ns = np.count_nonzero(self.__masks[:, :, self.__mask_idx])
            self.__samples_per_class[self.__mask_idx] = ns;
            np.random.shuffle(idx)
            idx = idx[0:ns, :]

        k = 0

        if self.__balance:
            # increase number of samples by copying them over multiple times
            max_samples = np.amax(self.__samples_per_class)
            copy_times = int(floor(
                max_samples / self.__samples_per_class[self.__mask_idx, 0]))  # num of times to copy for even division
            rem = max_samples % self.__samples_per_class[self.__mask_idx, 0]  # remaining samples

            for i in range(0, copy_times):
                np.random.shuffle(idx)
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
                            k += 1

            # copy the remaning samples so the total matches the max number
            # of samples chosen by user
            if rem > 0:
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
                            k += 1

        else:
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

        return np.asarray(input_), labels, k, idx

    def data_dims(self):
        '''
            # return data dimensions
        '''

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
            if self.__num_samples == 0:

                max_samples = np.amax(self.__samples_per_class)

                for i in range(0, self.__num_masks):
                    num_samples += self.__samples_per_class[i, 0] * int(
                        floor(max_samples / self.__samples_per_class[i, 0]))
                    num_samples += max_samples % self.__samples_per_class[i, 0]  # add remaining samples
            else:
                # if user has specified a max number of samples per class
                ns = self.__num_samples
                for i in range(0, self.__num_masks):
                    if self.__num_samples > np.count_nonzero(self.__masks[:, :, i]):
                        ns = np.count_nonzero(self.__masks[:, :, i])
                        self.__samples_per_class[i] = ns;
                    else:
                        self.__samples_per_class[i] = self.__num_samples

                num_samples = 0

                max_samples = np.amax(self.__samples_per_class)

                for i in range(0, self.__num_masks):
                    num_samples += self.__samples_per_class[i, 0] * int(
                        floor(max_samples / self.__samples_per_class[i, 0]))
                    num_samples += max_samples % self.__samples_per_class[i, 0]  # add remaining samples

        elif self.__num_samples == 0:

            num_samples = np.count_nonzero(annotated_pixels)
            #print('\n========>num_samples: ', num_samples)
        else:
            for i in range(0, self.__num_masks):
                if self.__num_samples > np.count_nonzero(self.__masks[:, :, i]):
                    num_samples += np.count_nonzero(self.__masks[:, :, i])
                else:
                    num_samples += self.__num_samples
                    # print('balancing -- num-samples: ', num_samples)

        return num_samples, rows, cols
