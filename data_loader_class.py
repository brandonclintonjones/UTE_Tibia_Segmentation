###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       January 11, 2021
#########
#########
#########
###############################
###############################
###############################
###############################
###############################

import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy.io import loadmat
import numpy.fft as fft
import numpy.matlib
import os
import tensorflow as tf
import nibabel as nib
import scipy
import cv2
import skimage
import skimage.transform
from helpers import *
import copy

#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############


from show_3d_images import show_3d_images

# from create_kspace_mask import gen_pdf, gen_sampling_mask, view_mask_and_pdfs


class data_generator:
    def __init__(
        self,
        create_new_dataset=False,
        valid_split=0.2,
        channel_str=["Echo2"],
        seed=1,
        noise=0,
        zoomRange=(1, 1),
        rotationRange=0,
        shearRange=0,
        widthShiftRange=0,
        heightShiftRange=0,
        flipLR=False,
        flipUD=False,
    ):

        super(data_generator, self).__init__()

        project_folder = os.getcwd()
        study_dir = os.path.join(os.path.dirname(project_folder), "DATA")

        self.project_folder = project_folder
        self.study_dir = study_dir
        echo1_name = os.path.join(self.project_folder, "Echo1_train.npy")

        # Variable declarations for image generation and augmentation
        self.valid_split = valid_split
        self.noise = noise
        self.zoomRange = zoomRange
        self.rotationRange = rotationRange
        self.widthShiftRange = widthShiftRange
        self.heightShiftRange = heightShiftRange
        self.flipLR = flipLR
        self.flipUD = flipUD

        if not os.path.exists(echo1_name) or create_new_dataset:
            print("\n\n CREATING NEW DATASET !!!! \n\n\n")
            self.load_data_from_nii()
        else:
            print("\n\n DATASET EXISTS. LOADING !!!! \n\n\n")
            self.load_dataset()

        self.n_channels = len(channel_str)
        self.channel_str = channel_str

        self.augmentor = Data_Augmentation(
            seed=seed,
            img_size=(self.n_rows, self.n_cols),
            nChannels=self.n_channels,
            noise=noise,
            zoomRange=zoomRange,
            rotationRange=rotationRange,
            shearRange=shearRange,
            widthShiftRange=widthShiftRange,
            heightShiftRange=heightShiftRange,
            flipLR=flipLR,
            flipUD=flipUD,
        )

    def load_dataset(self):
        # print('LOADING DATA FROM DISK!!!! \n\n')

        # THIS ASSUMES THAT THE DATA IS ALREADY SAVED IN CONVENIENT .NPY FILES.
        # IT LOADS THE TRAIN AND VALIDATION IMAGES
        # NOTE THAT THE IMAGES ARE LOADED INTO MEMEORY AND SAVED AS CLASS OBJECTS
        #           IF MEMORY CONSTRAINTS BECOME AN ISSUE, CAN SIMPLY LOAD SUBSET ON
        #           EACH CALL OF .GENERATOR( )

        self.echo1_train = np.load(os.path.join(self.project_folder, "Echo1_train.npy"))
        self.echo2_train = np.load(os.path.join(self.project_folder, "Echo2_train.npy"))
        self.irrute_train = np.load(
            os.path.join(self.project_folder, "Irrute_train.npy")
        )
        self.mask_train = np.load(os.path.join(self.project_folder, "Mask_train.npy"))
        self.train_scan_folders = np.load(
            os.path.join(self.project_folder, "Train_names.npy")
        )

        self.echo1_valid = np.load(os.path.join(self.project_folder, "Echo1_valid.npy"))
        self.echo2_valid = np.load(os.path.join(self.project_folder, "Echo2_valid.npy"))
        self.irrute_valid = np.load(
            os.path.join(self.project_folder, "Irrute_valid.npy")
        )
        self.mask_valid = np.load(os.path.join(self.project_folder, "Mask_valid.npy"))
        self.valid_scan_folders = np.load(
            os.path.join(self.project_folder, "Valid_names.npy")
        )

        # SAVE CLASS OBJECTS WITH IMPORTANT DATASET INFO

        self.total_num_scans = len(self.train_scan_folders) + len(
            self.valid_scan_folders
        )
        self.num_train_scans = len(self.train_scan_folders)
        self.num_valid_scans = len(self.valid_scan_folders)

        echo1_shape = self.echo1_train.shape
        self.n_slices = echo1_shape[3]
        self.n_rows = echo1_shape[1]
        self.n_cols = echo1_shape[2]

    def get_info(self, return_names=False):

        # RETURN IMPORTANT DATASET INFO, LIKE THE TOTAL NUMBER OF VALID/TRAIN SCANS,
        # TENSOR DIMENSIONS, NAMES OF IMAGES, ETC

        echo1_shape = self.echo1_train.shape
        total_num_scans = echo1_shape[0]
        n_rows = echo1_shape[1]
        n_cols = echo1_shape[2]
        n_slices = echo1_shape[3]

        total_train_slices = n_slices * self.num_train_scans
        total_valid_slices = n_slices * self.num_valid_scans

        if return_names:
            return (
                total_num_scans,
                self.n_channels,
                n_rows,
                n_cols,
                n_slices,
                total_train_slices,
                total_valid_slices,
                self.train_scan_folders,
                self.valid_scan_folders,
            )

        else:

            return (
                total_num_scans,
                self.n_channels,
                n_rows,
                n_cols,
                n_slices,
                total_train_slices,
                total_valid_slices,
            )

    def load_data_from_nii(self):
        # print("\nLoading images and saving to npy files . . . \n")

        # THIS FUNCTION LOADS DATA FROM EACH NIFTI FILE
        #       RANDOMLY SPLITS THE DATASET INTO TRAIN AND VALIDATION SETS
        #           AND SAVES CORRECT TENSORS TO CONVENEIENT .NPY FILES

        study_dir = self.study_dir
        project_folder = self.project_folder

        mask_names = os.path.join(study_dir, "SCAN_*/DL_mask.nii")

        print(mask_names)

        # use regular expressions to find correct Deep learrning mask data
        mask_files = glob.glob(mask_names)

        print(mask_files)

        total_num_scans = len(mask_files)
        self.total_num_scans = total_num_scans

        valid_split = self.valid_split
        # determine number of train and validation scans
        num_valid_scans = int(np.ceil(total_num_scans * valid_split))
        num_train_scans = int(total_num_scans - num_valid_scans)

        # NOW RANDOMLY SHUFFLE THE FILE ORDER TO HAVE BALANCED SETS
        index_arange = np.arange(total_num_scans)
        np.random.shuffle(index_arange)
        mask_files_randomized = copy.deepcopy(mask_files)

        for iter_scan in range(total_num_scans):
            current_index = index_arange[iter_scan]
            mask_files[current_index] = mask_files_randomized[iter_scan]

        self.num_train_scans = num_train_scans
        self.num_valid_scans = num_valid_scans

        train_scan_folders = []
        valid_scan_folders = []
        print("FOUND " + str(total_num_scans) + " TOTAL SCANS")

        echo1_name = "DL_echo1.nii"
        echo2_name = "DL_echo2.nii"
        irrute_name = "DL_ir_rute.nii"
        mask_name = "DL_mask.nii"

        ############
        ############
        ############   CREATE TRAINING DATASET
        ############
        ############

        global_index = 0

        for iter_scan in range(num_train_scans):

            current_scan_folder = os.path.dirname(mask_files[global_index])
            global_index += 1

            train_scan_folders.append(current_scan_folder)

            mask_file_nii = nib.load(os.path.join(current_scan_folder, mask_name))
            echo1_file_nii = nib.load(os.path.join(current_scan_folder, echo1_name))
            echo2_file_nii = nib.load(os.path.join(current_scan_folder, echo2_name))
            irrute_file_nii = nib.load(os.path.join(current_scan_folder, irrute_name))

            mask_data = mask_file_nii.get_fdata()
            echo1_data = echo1_file_nii.get_fdata()
            echo2_data = echo2_file_nii.get_fdata()
            irrute_data = irrute_file_nii.get_fdata()

            # CROP IMAGES TO HALF SIZE CENETERED AT FOV
            start_ind = 256 / 4
            end_ind = 3 * 256 / 4

            mask_data = crop_img(mask_data, start_ind, end_ind)
            echo1_data = crop_img(echo1_data, start_ind, end_ind)
            echo2_data = crop_img(echo2_data, start_ind, end_ind)
            irrute_data = crop_img(irrute_data, start_ind, end_ind)

            # NOW TAKE LOADED IMAGES AND SCALE BETWEEN 0 AND 1
            echo1_data = scale_to_0_1(echo1_data)
            echo2_data = scale_to_0_1(echo2_data)
            irrute_data = scale_to_0_1(irrute_data)
            mask_data = scale_to_0_1(mask_data)

            # IF ON THE FIRST ITERATION, PRE-ALLOCATE THE FINAL NUMPY TENSORS WHICH WE'LL
            # SAVE EACH PATIENT DATA INTO
            if iter_scan == 0:

                (n_rows, n_cols, n_slices) = mask_data.shape

                n_slices = 46
                self.n_slices = 46

                echo1_train = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                echo2_train = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                irrute_train = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                mask_train = np.zeros((num_train_scans, n_rows, n_cols, n_slices))

            # ITERATE THROUGH DATASET AND SAVE
            echo1_train[iter_scan, :, :, :] = echo1_data[:, :, 0:n_slices]
            echo2_train[iter_scan, :, :, :] = echo2_data[:, :, 0:n_slices]
            irrute_train[iter_scan, :, :, :] = irrute_data[:, :, 0:n_slices]
            mask_train[iter_scan, :, :, :] = mask_data[:, :, 0:n_slices]

        np.save(os.path.join(project_folder, "Echo1_train.npy"), echo1_train)
        np.save(os.path.join(project_folder, "Echo2_train.npy"), echo2_train)
        np.save(os.path.join(project_folder, "Irrute_train.npy"), irrute_train)
        np.save(os.path.join(project_folder, "Mask_train.npy"), mask_train)
        np.save(os.path.join(project_folder, "Train_names.npy"), train_scan_folders)

        self.echo1_train = echo1_train
        del echo1_train
        self.echo2_train = echo2_train
        del echo2_train
        self.irrute_train = irrute_train
        del irrute_train
        self.mask_train = mask_train
        del mask_train
        self.train_scan_folders = train_scan_folders
        del train_scan_folders

        ############
        ############
        ############   CREATE VALIDATION DATASET
        ############
        ############

        for iter_scan in range(num_valid_scans):

            current_scan_folder = os.path.dirname(mask_files[global_index])
            global_index += 1
            # print(global_index)

            valid_scan_folders.append(current_scan_folder)

            mask_file_nii = nib.load(os.path.join(current_scan_folder, mask_name))
            echo1_file_nii = nib.load(os.path.join(current_scan_folder, echo1_name))
            echo2_file_nii = nib.load(os.path.join(current_scan_folder, echo2_name))
            irrute_file_nii = nib.load(os.path.join(current_scan_folder, irrute_name))

            mask_data = mask_file_nii.get_fdata()
            echo1_data = echo1_file_nii.get_fdata()
            echo2_data = echo2_file_nii.get_fdata()
            irrute_data = irrute_file_nii.get_fdata()

            # CROP IMAGES TO HALF SIZE CENETERED AT FOV
            start_ind = 256 / 4
            end_ind = 3 * 256 / 4

            mask_data = crop_img(mask_data, start_ind, end_ind)
            echo1_data = crop_img(echo1_data, start_ind, end_ind)
            echo2_data = crop_img(echo2_data, start_ind, end_ind)
            irrute_data = crop_img(irrute_data, start_ind, end_ind)

            echo1_data = scale_to_0_1(echo1_data)
            echo2_data = scale_to_0_1(echo2_data)
            irrute_data = scale_to_0_1(irrute_data)
            mask_data = scale_to_0_1(mask_data)

            # print('LABEL')
            # print_img_characteristics(mask_data)

            if iter_scan == 0:

                (n_rows, n_cols, n_slices) = mask_data.shape

                n_slices = 46
                self.n_slices = 46

                echo1_valid = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                echo2_valid = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                irrute_valid = np.zeros((num_train_scans, n_rows, n_cols, n_slices))
                mask_valid = np.zeros((num_train_scans, n_rows, n_cols, n_slices))

            echo1_valid[iter_scan, :, :, :] = echo1_data[:, :, 0:n_slices]
            echo2_valid[iter_scan, :, :, :] = echo2_data[:, :, 0:n_slices]
            irrute_valid[iter_scan, :, :, :] = irrute_data[:, :, 0:n_slices]
            mask_valid[iter_scan, :, :, :] = mask_data[:, :, 0:n_slices]

        np.save(os.path.join(project_folder, "Echo1_valid.npy"), echo1_valid)
        np.save(os.path.join(project_folder, "Echo2_valid.npy"), echo2_valid)
        np.save(os.path.join(project_folder, "Irrute_valid.npy"), irrute_valid)
        np.save(os.path.join(project_folder, "Mask_valid.npy"), mask_valid)
        np.save(os.path.join(project_folder, "Valid_names.npy"), valid_scan_folders)

        self.echo1_valid = echo1_valid
        del echo1_valid
        self.echo2_valid = echo2_valid
        del echo2_valid
        self.irrute_valid = irrute_valid
        del irrute_valid
        self.mask_valid = mask_valid
        del mask_valid
        self.valid_scan_folders = valid_scan_folders
        del valid_scan_folders

        echo1_shape = self.echo1_valid.shape
        self.n_slices = echo1_shape[3]
        self.n_rows = echo1_shape[1]
        self.n_cols = echo1_shape[2]

        print("DONE saving data \n\n\n")

    def display_images(self, index=-1, is_train=True):

        # print(index)
        # HELPFUL FUNCTION FOR TROUBLESHOOTING DATA LOADING
        # CREATES  FIGURE WINDOW WHICH YOU CAN SCROLL THROUGH WITH
        # YOUR MOUSE

        if index < 0:
            index = np.random.randint(low=0, high=self.total_num_scans - 1)

        if is_train:

            echo1 = self.echo1_train[index, :, :, :]
            echo2 = self.echo2_train[index, :, :, :]
            irrute = self.irrute_train[index, :, :, :]
            mask = self.mask_train[index, :, :, :]
        else:
            echo1 = self.echo1_valid[index, :, :, :]
            echo2 = self.echo2_valid[index, :, :, :]
            irrute = self.irrute_valid[index, :, :, :]
            mask = self.mask_valid[index, :, :, :]

        echo1 = scale_to_0_1(echo1)
        echo2 = scale_to_0_1(echo2)
        irrute = scale_to_0_1(irrute)
        print("SHOWING CASE " + str(index) + ", " + self.scan_folders[index])

        show_3d_images(np.hstack((mask, echo1, echo2, irrute)))

    def generator(self, batch_ind, batch_size, is_train=True):

        ############
        ############
        ############   THIS IS THE REAL MEAT OF THE GENERATOR CODE
        ############
        ############
        ############
        ############        GENERATOR IS CALLED ON EACH STEP-PER-EPOCH
        ############           AND YIELDS THE PAIRED IMAGE AND MASK LABEL
        ############            DATA FOR A PARTICULAR BATCH
        (batch_dim, n_rows, n_cols, n_slices) = self.mask_valid.shape

        # IF A NEW EPOCH HAS STARTED, SHUFFLE THE DATA ORDER
        #### SHUFFLE DATA #####

        if batch_ind == 0:
            self.shuffle_data()

        #### DONE SHUFFLIN DATA #####

        # determine which images to load
        first_ind = batch_ind * batch_size
        last_ind = (batch_ind + 1) * batch_size - 1

        n_channels = len(self.channel_str)
        mask_output = np.zeros((batch_size, 1, n_rows, n_cols))
        echo1_output = np.zeros((batch_size, 1, n_rows, n_cols))
        echo2_output = np.zeros((batch_size, 1, n_rows, n_cols))
        irrute_output = np.zeros((batch_size, 1, n_rows, n_cols))
        img_output = np.zeros((batch_size, n_channels, n_rows, n_cols))

        channel_counter = 0
        # WANT TO STACK THEM IN ORDER ECHO2, ECHO1, IR-RUTE
        if "Echo2" in self.channel_str:
            # print('FOUND ECHO 2')
            if is_train:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.echo2_batch_train[:, :, iter_slice + first_ind]
            else:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.echo2_batch_valid[:, :, iter_slice + first_ind]
            channel_counter += 1
        if "Echo1" in self.channel_str:
            # print('FOUND ECHO 1')
            if is_train:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.echo1_batch_train[:, :, iter_slice + first_ind]
            else:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.echo1_batch_valid[:, :, iter_slice + first_ind]
            channel_counter += 1

        if "IR-rUTE" in self.channel_str:
            # print('FOUND IR')
            if is_train:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.irrute_batch_train[:, :, iter_slice + first_ind]
            else:
                for iter_slice in range(batch_size):
                    img_output[
                        iter_slice, channel_counter, :, :
                    ] = self.irrute_batch_valid[:, :, iter_slice + first_ind]
            channel_counter += 1

        if n_channels != channel_counter:
            raise ValueError(
                "n channels is not equal to counter. Check channel string and which ehcoes are added"
            )

        if is_train:
            for iter_slice in range(batch_size):
                mask_output[iter_slice, 0, :, :] = self.mask_batch_train[
                    :, :, iter_slice + first_ind
                ]

        else:
            for iter_slice in range(batch_size):
                mask_output[iter_slice, 0, :, :] = self.mask_batch_valid[
                    :, :, iter_slice + first_ind
                ]

        if is_train:
            (img_output, mask_output) = self.augmentor.perturb_data(
                img_output, mask_output
            )

        for iter_batch in range(batch_size):
            for iter_channel in range(n_channels):
                img_output[iter_batch, iter_channel, :, :] = scale_to_z_curve(
                    img_output[iter_batch, iter_channel, :, :].squeeze()
                )

        return (img_output.astype(np.float32), mask_output.astype(np.float32))

    def gen_one_image(self, batch_ind, batch_size):
        # USEFUL SCRIPT WHEN YOU ARE TRYING TO TROUBLESHOOT GRADIENT ISSUES AND WANT TO OVERFIT

        (batch_dim, n_rows, n_cols, n_slices) = self.mask_matrix.shape

        mask_output = np.zeros((1, 1, n_rows, n_cols))
        echo2_output = np.zeros((1, 1, n_rows, n_cols))

        mask_output[0, :, :, :] = self.mask_matrix[0, :, :, 0]
        echo2_output[0, :, :, :] = scale_to_z_curve(self.echo2_matrix[0, :, :, 0])
        img_output = echo2_output

        # img_output = img_output.reshape((1,n_rows,n_cols,1))
        # mask_output = mask_output.reshape((1,n_rows,n_cols,1))

        # tmp=np.hstack((echo2_output.squeeze().reshape(n_rows,n_cols,1),
        #     scale_to_0_1(mask_output.squeeze()).reshape(n_rows,n_cols,1)))
        # show_3d_images(tmp)

        return (img_output, mask_output)
        # return (img_output.astype(np.float32),mask_output.astype(np.float32))

    def shuffle_data(self):
        # print("\n Shuffling data . . . \n")

        # THIS SHUFFLES THE DATA ORDER ON EACH EPOCH START
        # FOR CONVENIENCE, DATA IS LOADED INTO MEMORY IN TWO COPIES
        # IF MEMORY CONSTRAINTS BECOME ISSUE, CAN LOAD DIRECTLY
        # FROM FILES
        total_train_slices = self.n_slices * self.num_train_scans
        total_valid_slices = self.n_slices * self.num_valid_scans

        (batch_dim, n_rows, n_cols, n_slices) = self.mask_train.shape

        self.mask_batch_train = np.zeros((n_rows, n_cols, total_train_slices))
        self.irrute_batch_train = np.zeros((n_rows, n_cols, total_train_slices))
        self.echo1_batch_train = np.zeros((n_rows, n_cols, total_train_slices))
        self.echo2_batch_train = np.zeros((n_rows, n_cols, total_train_slices))

        index_arange_train = np.arange(total_train_slices)
        np.random.shuffle(index_arange_train)

        for iter_slice in range(total_train_slices):
            overall_index = index_arange_train[iter_slice]
            scan_index = np.floor_divide(overall_index, 46)
            slice_index = np.mod(overall_index, 46)

            self.mask_batch_train[:, :, iter_slice] = self.mask_train[
                scan_index, :, :, slice_index
            ]
            self.echo1_batch_train[:, :, iter_slice] = self.echo1_train[
                scan_index, :, :, slice_index
            ]
            self.echo2_batch_train[:, :, iter_slice] = self.echo2_train[
                scan_index, :, :, slice_index
            ]
            self.irrute_batch_train[:, :, iter_slice] = self.irrute_train[
                scan_index, :, :, slice_index
            ]

        self.mask_batch_valid = np.zeros((n_rows, n_cols, total_valid_slices))
        self.irrute_batch_valid = np.zeros((n_rows, n_cols, total_valid_slices))
        self.echo1_batch_valid = np.zeros((n_rows, n_cols, total_valid_slices))
        self.echo2_batch_valid = np.zeros((n_rows, n_cols, total_valid_slices))

        index_arange_valid = np.arange(total_valid_slices)
        np.random.shuffle(index_arange_valid)

        for iter_slice in range(total_valid_slices):
            overall_index = index_arange_valid[iter_slice]
            scan_index = np.floor_divide(overall_index, 46)
            slice_index = np.mod(overall_index, 46)

            self.mask_batch_valid[:, :, iter_slice] = self.mask_valid[
                scan_index, :, :, slice_index
            ]
            self.echo1_batch_valid[:, :, iter_slice] = self.echo1_valid[
                scan_index, :, :, slice_index
            ]
            self.echo2_batch_valid[:, :, iter_slice] = self.echo2_valid[
                scan_index, :, :, slice_index
            ]
            self.irrute_batch_valid[:, :, iter_slice] = self.irrute_valid[
                scan_index, :, :, slice_index
            ]


class Data_Augmentation:
    def __init__(
        self,
        seed=1,
        img_size=(200, 200),
        nChannels=1,
        noise=0.05,
        zoomRange=(1, 1),
        rotationRange=0,
        shearRange=0,
        widthShiftRange=0,
        heightShiftRange=0,
        flipLR=False,
        flipUD=False,
    ):

        super(Data_Augmentation, self).__init__()

        "Initialization"

        self.img_size = img_size
        self.noise = noise

        self.random = np.random.RandomState(seed=seed)

        self.zoomRange = zoomRange
        self.shearRange = shearRange
        self.rotationRange = rotationRange
        self.widthShiftRange = widthShiftRange
        self.heightShiftRange = heightShiftRange

        self.flipLR = flipLR
        self.flipUD = flipUD

        if self.shearRange == 0:
            self.shear = False
        else:
            self.shear = True

    def perturb_data(self, input_data, input_label):

        img_shape = input_data.shape
        batch_dim = img_shape[0]
        n_channels = img_shape[1]
        n_rows = img_shape[2]
        n_cols = img_shape[3]

        if n_channels > 1:
            is_multi_channel = True
        else:
            is_multi_channel = False

        # THIS ITERATES THROUGH EACH IMAGE-MASK PAIR AND APPLIES THE SAME AUGMENTATION TO EACH PAIR

        for iter_img in range(batch_dim):

            (
                input_data[iter_img, :, :, :],
                input_label[iter_img, :, :, :],
            ) = self.perturb_image(
                input_data[iter_img, :, :, :],
                input_label[iter_img, :, :, :],
                is_multi_channel=is_multi_channel,
            )

        return input_data, input_label

    def perturb_image(self, img, mask, is_multi_channel=True):
        fill_mode = "constant"

        if not is_multi_channel:

            img = img.squeeze()
            mask = mask.squeeze()

            # batch_dim, h, w = img.shape
            h, w = img.shape

            zf = self.getRandomZoomConfig(self.zoomRange)
            theta = self.getRandomRotation(self.rotationRange)
            tx, ty = self.getRandomShift(
                *self.img_size, self.widthShiftRange, self.heightShiftRange
            )
            sr = self.getRandomShearVal(self.shearRange)

            if self.noise != 0:
                img = self.apply_gaussian_noise(img, h, w)

            if self.shear != 0:
                img = self.apply_shear(img.squeeze(), sr=self.shear)
                mask = self.apply_shear(mask.squeeze(), sr=self.shear)

            if zf != 0:
                img = self.applyZoom(img, zf, fill_mode=fill_mode)
                mask = self.applyZoom(mask, zf, fill_mode=fill_mode)
                mask = mask > 0.5

            if theta != 0:
                img = self.applyRotation(img, theta, fill_mode=fill_mode)
                mask = self.applyRotation(mask, theta, fill_mode=fill_mode)
                mask = mask > 0.5

            if (tx != 0) or (ty != 0):
                img = self.applyShift(img, tx, ty, fill_mode=fill_mode)
                mask = self.applyShift(mask, tx, ty, fill_mode=fill_mode)

            if self.flipLR and self.getRandomFlipFlag():
                img = np.fliplr(img)
                mask = np.fliplr(mask)
            if self.flipUD and self.getRandomFlipFlag():
                img = np.flipud(img)
                mask = np.flipud(mask)

            img = img.reshape(1, h, w)
            mask = mask.reshape(1, h, w)

        else:

            img_shape = img.shape
            channel_dim = img_shape[0]
            h = img_shape[1]
            w = img_shape[2]

            zf = self.getRandomZoomConfig(self.zoomRange)
            theta = self.getRandomRotation(self.rotationRange)
            tx, ty = self.getRandomShift(
                *self.img_size, self.widthShiftRange, self.heightShiftRange
            )
            sr = self.getRandomShearVal(self.shearRange)

            if self.noise != 0:
                for iter_img in range(channel_dim):
                    # print('ADDING NOISE \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = self.apply_gaussian_noise(
                        img[iter_img, :, :], h, w
                    )
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

            if self.shear != 0:
                for iter_img in range(channel_dim):

                    img[iter_img, :, :] = self.apply_shear(
                        img[iter_img, :, :].squeeze(), sr=sr
                    )

                mask = self.apply_shear(mask.squeeze(), sr=sr)
                mask = mask > 0.5

            if zf != 0:
                for iter_img in range(channel_dim):
                    # print('APPLYING ZOOM \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = self.applyZoom(
                        img[iter_img, :, :].squeeze(), zf, fill_mode=fill_mode
                    )
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

                mask = self.applyZoom(mask.squeeze(), zf, fill_mode=fill_mode)
                mask = mask > 0.5
                # show_3d_images(mask.reshape((h,w,1)))

            if theta != 0:
                for iter_img in range(channel_dim):
                    # print('ROTATING IMAGE \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = self.applyRotation(
                        img[iter_img, :, :].squeeze(), theta, fill_mode=fill_mode
                    )
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

                mask = self.applyRotation(mask.squeeze(), theta)
                mask = mask > 0.5
                # show_3d_images(mask.reshape((h,w,1)))

            if (tx != 0) or (ty != 0):
                for iter_img in range(channel_dim):
                    # print('SHIFTING IMAGE \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = self.applyShift(
                        img[iter_img, :, :].squeeze(), tx, ty, fill_mode=fill_mode
                    )
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

                mask = self.applyShift(mask.squeeze(), tx, ty, fill_mode=fill_mode)
                # show_3d_images(mask.reshape((h,w,1)))

            if self.flipLR and self.getRandomFlipFlag():
                for iter_img in range(channel_dim):
                    # print('FLIP LR \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = np.fliplr(img[iter_img, :, :])
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

                mask = np.fliplr(mask)

            if self.flipUD and self.getRandomFlipFlag():
                for iter_img in range(channel_dim):
                    # print('FLIP UD \n')
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))
                    img[iter_img, :, :] = np.flipud(img[iter_img, :, :].squeeze())
                    # show_3d_images(img[iter_img,:,:].reshape(h,w,1))

                mask = np.flipud(mask)

        return img, mask

    def apply_gaussian_noise(self, img, h, w):
        noise = np.random.normal(loc=0.0, scale=self.noise, size=h * w).reshape(h, w)
        img = noise + img

        return img.clip(0, 1)

    def getRandomFlipFlag(self):
        return self.random.choice([True, False])

    def getRandomZoomConfig(self, zoomRange):
        if zoomRange[0] == 1 and zoomRange[1] == 1:
            zf = 1
        else:
            zf = self.random.uniform(zoomRange[0], zoomRange[1], 1)[0]
        return zf

    def getRandomShearVal(self, shearRange):
        if shearRange != 0:

            shear_val = self.random.uniform(low=-shearRange, high=shearRange, size=1)[0]

        else:
            shear_val = 0

        return shear_val

    def apply_shear(self, img, sr):
        tform = skimage.transform.AffineTransform(shear=sr)

        img = skimage.transform.warp(image=img, inverse_map=tform, preserve_range=True)
        return img.clip(0, 1)

    def applyZoom(self, img, zf, fill_mode="nearest", cval=0.0, interpolation_order=0):
        img = img.squeeze()
        origShape = img.shape
        h, w = origShape

        img = scipy.ndimage.zoom(
            img, zf, mode=fill_mode, cval=cval, order=interpolation_order
        )

        if zf < 1:
            canvas = np.zeros(origShape, dtype=img.dtype)

            rowOffset = int(np.floor((origShape[0] - img.shape[0]) / 2))
            colOffset = int(np.floor((origShape[1] - img.shape[1]) / 2))

            canvas[
                rowOffset : (rowOffset + img.shape[0]),
                colOffset : (colOffset + img.shape[1]),
            ] = img
            img = canvas
        elif zf > 1:
            rowOffset = int(np.floor((img.shape[0] - origShape[0]) / 2))
            colOffset = int(np.floor((img.shape[1] - origShape[1]) / 2))
            img = img[
                rowOffset : (rowOffset + origShape[0]),
                colOffset : (colOffset + origShape[1]),
            ]

        return img

    def getRandomRotation(self, rotationRange):
        theta = self.random.uniform(-rotationRange, rotationRange)
        return theta

    def applyRotation(
        self,
        img,
        theta,
        fill_mode="nearest",
        cval=0.0,
        interpolation_order=3,
        prefilter=False,
    ):
        # def applyRotation(self, img, theta, fill_mode='wrap', cval=0., interpolation_order=3):
        # print(theta)
        img = img.astype(np.float64)
        img_shape = img.shape
        img_center = (int(img_shape[0] / 2), int(img_shape[1] / 2))
        rot_mat = cv2.getRotationMatrix2D(img_center, theta, 1.0)

        img = cv2.warpAffine(img, rot_mat, img_shape[1::-1], flags=cv2.INTER_LINEAR)

        return img

    def getRandomShift(self, h, w, widthShiftRange, heightShiftRange):
        tx = self.random.uniform(-heightShiftRange, heightShiftRange) * h
        ty = self.random.uniform(-widthShiftRange, widthShiftRange) * w
        return (tx, ty)

    def applyShift(
        self, img, tx, ty, fill_mode="constant", cval=0.0, interpolation_order=0
    ):
        img = scipy.ndimage.shift(
            img, [tx, ty], mode=fill_mode, cval=cval, order=interpolation_order
        )

        return img


if __name__ == "__main__":

    # print('\n\n\n')

    project_folder = os.getcwd()
    study_dir = os.path.join(os.path.dirname(project_folder), "DATA")

    #######
    #######
    #######     NOTE: CHANNEL_STR IS THE INPUT IMAGE CHANNELS
    #######         FOR THIS PARTICULAR SEGMENTATION PROBLEM, WE HAVE 3 CO-LOCALIZED MR IMAGE CONTRAST
    #######             TE_1 - 50 US, TE_2 - 4600 US, AND AN IR SEQUENCE WHICH SUPPRESSES LONG T2 SPECIES
    #######
    #######             SO CALLING CHANNEL_STR = [ 'IR-rUTE','Echo2','Echo1' ]
    #######                 THE GENERATOR WILL LOAD ALL 3. TO LOAD ONE JUST CALL CHANNEL_STR = [ 'IR-rUTE] FOR EXAMPLE

    # channel_str = [ 'Echo1' ]
    # channel_str = [ 'IR-rUTE','Echo2' ]

    channel_str = ["IR-rUTE", "Echo2", "Echo1"]

    # Instantiate the generator class. This class will call the data augmentation class at the time of image generation when needed
    # NOTE: THIS DATASET IS SET UP TO RETURN IN A CHANNELS FIRST FORMAT
    # SO THE TENSOR DIMENSIONS ARE ( BATCH , CHANNELS , N_ROWS  , N_COLS )
    my_gen = data_generator(
        create_new_dataset=True,
        valid_split=0.2,
        channel_str=channel_str,
        seed=1,
        noise=0.04,
        zoomRange=(0.9, 1.1),
        shearRange=0.2,
        rotationRange=0,
        widthShiftRange=0.1,
        heightShiftRange=0.1,
        flipLR=True,
        flipUD=True,
    )
    # Create_new_dataset -
    #           If true, will load all data from .nii files and will save new .npy files
    #                   Keep false most of the time when training/optimizing

    #  Valid_split -        0.2 implies 20% of data set to validation, 0.8 to train
    #
    #   channel_str - See comments above
    #       seed = 1            Leave this if you don't know what this does
    #       noise = 0.4       ==        This is the level of Gaussian noise to add to the images.
    #   zoomRange = ( 0.8 , 1.2 )    ==             This implies +/- 20% zoom
    #    shearRange = (0.2 )        ==          This means a shear parameter of 0.2. Its a Scipy parameter but I think it means 20% of FOV?
    #       rotationRange ==            Angle in degrees. If set to 45 and FlipUD and flipLR are on you get every orientation
    #       shiftRange ==           These are the FOV ratios it can be shift. 0.1 means 10% of FOV
    #       flipLR and flipUD ==        Do you want the image to be flipped across X and Y dimensions for augmentation?

    # Simply return important data set information.
    (
        total_num_scans,
        n_channels,
        n_rows,
        n_cols,
        n_slices,
        num_train,
        num_valid,
        train_names,
        valid_names,
    ) = my_gen.get_info(return_names=True)

    # FIND TOTAL NUMBER OF TRAIN AND VALIDATION IMAGES
    num_train_scans = len(train_names)
    num_valid_scans = len(valid_names)

    print(num_train)
    print(num_valid)
    print(train_names)
    print(valid_names)

    # print('TRAIN NAMES ')
    # for iter_scan in range(num_train_scans):
    #     print(train_names[iter_scan])

    # print('\n\n VALID NAMES ')
    # for iter_scan in range(num_valid_scans):
    #     print(valid_names[iter_scan])

    batch_size = 30

    num_train_steps = int(np.floor(num_train / batch_size))
    num_valid_steps = int(np.floor(num_valid / batch_size))

    # THIS SHOWS AN EXAMPLE OF HOW THE DATA LOADER IS ACTUALLY CALLED WITHIN EACH STEP-PER-EPOCH
    # IF YOU RUN INTO MEMEORY CONSTRAINTS, SIMPLY SET THE BATCH SIZE TO SMALLER AND INCREASE num_train_Steps

    for iter_scan in range(num_train_steps):
        print("TRAIN SET, ITER SCAN " + str(iter_scan))
        (echo_image, mask) = my_gen.generator(
            batch_ind=iter_scan, batch_size=batch_size - 1, is_train=True
        )

        img_masked = np.multiply(
            echo_image[:, 0, :, :].squeeze(), mask.squeeze()
        ).reshape(mask.shape)

        plot1 = prep_images_for_plot(echo_image, mask, img_masked)

    for iter_scan in range(num_valid_steps):
        print("VALID SET, ITER SCAN " + str(iter_scan))
        (echo_image, mask) = my_gen.generator(
            batch_ind=iter_scan, batch_size=batch_size - 1, is_train=False
        )

        img_masked = np.multiply(
            echo_image[:, 0, :, :].squeeze(), mask.squeeze()
        ).reshape(mask.shape)

        plot2 = prep_images_for_plot(echo_image, mask, img_masked)
