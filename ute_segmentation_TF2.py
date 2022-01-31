###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########       January 11, 2022
#########
#########
#########
###############################
###############################
###############################
###############################


# TF WRITE TRAINING LOOP FROM SCRATCH
# https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

import tensorflow as tf
import numpy as np
import os
import glob
from show_3d_images import show_3d_images
from helpers import *
import timeit
import tqdm
import os
import glob
from data_loader_class import (
    data_generator,
    Data_Augmentation,
    print_img_characteristics,
)
import model_architectures_TF2
import skimage
import skimage.measure
import copy
import scipy


#############
#############
#############       CUSTOM DEPENDENCIES!!!!!
#############
#############

# from data_loader_class import data_generator


class CNN:
    # class CNN(tf.Module):
    def __init__(
        self,
        project_folder,
        study_dir,
        channel_str,
        model_name,
        batch_size=10,
        max_epoch=100,
        learn_rate=1e-3,
    ):
        """
            Defines the CNN structure

            Parameters:
           


            """

        ######
        ######      DEFINE TENSORFLOW CONFIGURATIONS. NOT IMPORTANT TO UNDERSTAND
        ######
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        ######
        ######      SET RANDOM SEED FOR VARIABLE WEIGHTS TO HELP WITH REPRODUCIBILITY
        ######
        tf.random.set_seed(seed=1)

        # The model weight is the lamba scale factor I use for the
        # Weighted cross-entropy loss. I find it's better for initial training
        # to put large value (~200) to prioritize detecting the bone boundary
        #
        #       and then I kick it down to 1 when its already doing ok in segmentation
        # self.loss_weight = 200
        self.loss_weight = 1

        # This is just defining where the data is and where I want the model to be saved
        self.project_folder = project_folder
        self.study_dir = study_dir

        # Define default datatype. Pretty much always Float32. Float64 is just overkill
        self.dtype = tf.float32

        # this is the learning rate. I typically start with 1e-3 and quickly kick it down
        # to 1e-5 after its initial few epochs
        self.learn_rate = learn_rate

        # Maximum epochs (or max iterations. An epoch is an iteration i nwhich we cycle through
        # our entire training set and update gradient weights)
        self.max_epoch = int(max_epoch)

        # Batch size is how much of the data we load at one time. If we're using 10k images
        # we couldn't possibly load everything into memory. We load N each batch and update
        # gradients wiht respect to only those N
        self.batch_size = batch_size

        # Model name and directories for saving purposes
        self.model_name = model_name
        self.save_dir = os.path.join(self.project_folder, self.model_name) + "/"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.save_str = self.save_dir + self.model_name

        self.save_dir = os.path.join(self.project_folder, self.model_name + "/")

        self.model_save_name = os.path.join(self.save_dir, "model.h5")

        self.tensorboard_dir = os.path.join(self.project_folder, "TENSORBOARD/")

        # Channel string is how I tell the model which images to load.
        # We have 3 contrasts- Echo 1 (ultrashort, TE = 50 us)
        # Echo 2, TE = 4.6 ms
        # And an adiabatic IR-rUTE which suppresses long T2 species
        # Feed in like ['Echo1'] or ['Echo1','Echo2','IR-rUTE']
        self.channel_str = channel_str
        self.learn_rate = learn_rate

        # self.my_data_generator = data_generator(
        #     create_new_dataset=True,
        #     valid_split=0.2,
        #     channel_str=channel_str,
        #     seed=1,
        #     noise=0.04,
        #     zoomRange=(0.9,1.1),
        #     shearRange=0.2,
        #     rotationRange=0,
        #     widthShiftRange=0.1,
        #     heightShiftRange=0.1,
        #     flipLR=True,
        #     flipUD=True,
        # )

        ######
        ######      HERE I INSTANTIATE MY DATA AUGMENTATION AND LOADER CLASS
        ######              AND THEN I PASS IT TO THE MAIN CNN CLASS SO ITS EASY
        ######                      TO CALL WITHIN EACH CNN CLASS FUNCTION WHENEVER I NEED IT
        ######
        self.my_data_generator = data_generator(
            create_new_dataset=False,
            channel_str=self.channel_str,
            valid_split=0.2,
            seed=1,
            noise=0,
            zoomRange=(1, 1),
            rotationRange=0,
            shearRange=0,
            widthShiftRange=0,
            heightShiftRange=0,
            flipLR=True,
            flipUD=True,
        )
        # The inputs are just the augmentation features.
        # Valid_split is typically 0.2, meaning 20%
        # Typically leave create new dataset as false once its made

        # Get dataset info-number of scans, image sizes, etc.
        (
            self.total_num_scans,
            self.n_channels,
            self.n_rows,
            self.n_cols,
            self.n_slices,
            self.num_train,
            self.num_valid,
        ) = self.my_data_generator.get_info()

        # NOTE: REMEMBER IN TENSORFLOW 2 YOU DON'T FEED NONE FOR BATCH SHAPE
        # Define tuple of desired input. This changes based on number of channels
        # 1 2 or 3 depending on which images I'm feeding in - TE1, TE2, IR sequnce
        self.input_image_shape = (self.n_channels, self.n_rows, self.n_cols)

        # The network architecture is defined in the model_architectures_TF2 file. This calls it
        (
            self.cnn_input_placeholder,
            self.cnn_output_placeholder,
        ) = model_architectures_TF2.UNet_FINAL(input_tensor=self.input_image_shape)

        # And here we finally create the CNN =D
        # NOTE THAT IT IS SAVED AS A CLASS OBJECT SO IT IS EASILY ACCESSIBLE FROM
        #  THE FORWARD PASS, TRAIN, AND OTHER CLASS FUNCTIONS
        #
        self.my_cnn = tf.keras.Model(
            inputs=self.cnn_input_placeholder, outputs=self.cnn_output_placeholder
        )

    def forward_pass(self, tensor_input):
        """
                
                This is literally just a forward pass.
                It calls the forward output of the model.
                So if it is a U-Net, it takes in the image tensor
                And runs the serious convolutions, Activations, etc
                To produce the predicted mask

            """
        tensor_output = self.my_cnn(tensor_input)

        return tensor_output

    def train(self):
        """

            """

        # This creates a file writer which I can use to write
        # constants to Tensorboard file. Useful for parameter tuning
        train_writer = tf.summary.create_file_writer(
            os.path.join(self.tensorboard_dir, self.model_name)
        )

        # Ok here we have Ntrain and Nvalidation images and
        # Based on GPU constrains and batch size, we want to figure out
        # how many nested iterations it will take us to complete on epoch

        self.train_steps = self.num_train // self.batch_size
        self.valid_steps = self.num_valid // self.batch_size

        # ADAM IS BAE <33333
        my_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

        # These are counters. Used for writing model updates and
        # Schedulin learning rates
        train_global_counter = 0
        valid_global_counter = 0
        global_epoch_counter = 0

        # Define empty lists from which I will append the loss and image
        # metrics and save them in. I ended up not doing anythin with them
        # since Tensorboard is so pretty and useful, but its super easier to then
        # just take the values and write them to excel or txt file for viewing later on
        train_epoch_loss = []
        train_epoch_dsc = []
        train_epoch_iou = []

        valid_epoch_loss = []
        valid_epoch_dsc = []
        valid_epoch_iou = []

        # Start counting time for training to print it out to line
        start_time = timeit.default_timer()

        for epoch_num in range(self.max_epoch):
            print("\n\n TRAIN EPOCH NUMBER " + str(epoch_num + 1))
            global_epoch_counter += 1

            train_batch_loss = []
            train_batch_dsc = []
            train_batch_iou = []

            elapsed = timeit.default_timer()

            # NOTE: THIS IS A SIMPLE HACKY WAY TO DEFINE LEARNING RATE SCHEDULING
            # YOU CAN LOAD AND TRAIN A MODEL WITH GIVEN PARAMETERS AND JUST UPDATE
            # PARAMS FOR LOWER LEARNING RATES, OR YOU CAN JUST CODE THIS IN
            if epoch_num == 3:
                self.learn_rate = 1e-4
                my_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

            # if epoch_num == 30:
            #     self.learn_rate = 1e-5
            #     my_optimizer = tf.keras.optimizers.Adam(
            #         learning_rate=self.learn_rate
            #         )

            if epoch_num == 0:
                self.learn_rate = 1e-5 / 2
                my_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

            # if epoch_num == 30:
            #     self.learn_rate = 1e-5
            #     my_optimizer = tf.keras.optimizers.Adam(
            #         learning_rate=self.learn_rate
            #         )
            # Iterate through max number of epochs.
            # TQDM just prints a pretty loading bar to command line. I like it
            for counter in tqdm.tqdm(range(self.train_steps)):

                # Epoch count increases
                train_global_counter += 1

                # Load on batch. I have the generator set to
                # shuffle the data every time the batch_index is back to 0
                (
                    batch_input_image,
                    batch_label_mask,
                ) = self.my_data_generator.generator(
                    batch_ind=counter, batch_size=self.batch_size, is_train=True,
                )

                # I DONT REALLY KNOW HOW TO EXPLAIN THIS PART EXCEPT TO SAY
                # THIS IS TENSORFLOW 2'S SYNTAX FOR LOW-LEVEL CUSTOM MODEL WEIGHT TRAINING
                # INSTEAD OF RUNNING LIKE KERAS MODEL.COMPILE() YOU CALL
                # TENSORFLOW "GRADIENT TAPE" (WAHTEVER THAT IS)
                # AND YOU RUN THE FORWARD PASS WITHIN THE NAME SPACE
                with tf.GradientTape() as tape:

                    # THIS IS THE FORWARD PASS OF THE MODEL.
                    # GIVEN THE CURRENT STATE OF THE CNN, CREATE THE PREDICTION
                    batch_predicted_mask = self.forward_pass(batch_input_image)

                    # THEN YOU COMPUTE THE LOSS IN THE NAMESPACE
                    loss_value = self.get_loss(
                        mask_gt=batch_label_mask,
                        mask_pred=batch_predicted_mask,
                        epoch_num=epoch_num,
                    )

                    # THEN YOU COMPUTE THE GRADIENT OF THE LOSS WITH RESPECT TO THE MODEL WEIGHTS
                    grads = tape.gradient(loss_value, self.my_cnn.trainable_weights)

                    # AND THEN FINALLY YOU UPDATE THE WEIGHTS
                    my_optimizer.apply_gradients(
                        zip(grads, self.my_cnn.trainable_weights)
                    )

                ######
                ######      COMPUTE AND STORE LOSS AND IMAGE METRICS for a batch. CAN ADD WHATEVER
                ######              METRICS YOU WANT HERE SINCE EVERYTHING IS A NUMPY ARRAY.
                ######                  SUPER CONVENIENT. NO NEED TO CAST
                ######
                training_loss_value = loss_value.numpy()
                tmp_iou, tmp_dsc = get_dsc_iou(
                    mask_gt=batch_label_mask, mask_pred=batch_predicted_mask.numpy()
                )

                train_batch_dsc.append(tmp_dsc)
                train_batch_iou.append(tmp_iou)
                train_batch_loss.append(training_loss_value)

                ######
                ######      OK THIS PART USES THE FILEWRITER TO WRITE THE IMAGE METRICS TO TENSORBOARD
                ######
                with train_writer.as_default():
                    with tf.name_scope("TRAIN"):
                        tf.summary.scalar("DSC", tmp_dsc, step=valid_global_counter)
                        tf.summary.scalar("IOU", tmp_iou, step=valid_global_counter)
                        tf.summary.scalar(
                            "LOSS", training_loss_value, step=valid_global_counter
                        )

            ######
            ######      COMPUTE AND STORE LOSS AND IMAGE METRICS for the entire epoch. CAN ADD WHATEVER
            ######              METRICS YOU WANT HERE SINCE EVERYTHING IS A NUMPY ARRAY.
            ######                  SUPER CONVENIENT. NO NEED TO CAST
            ######

            train_batch_loss = np.asarray(train_batch_loss).mean()
            train_epoch_loss.append(train_batch_loss)

            train_batch_dsc = np.asarray(train_batch_dsc).mean()
            train_epoch_dsc.append(train_batch_dsc)
            train_batch_iou = np.asarray(train_batch_iou).mean()
            train_epoch_iou.append(train_batch_iou)

            total_elapsed = timeit.default_timer() - start_time
            elapsed = timeit.default_timer() - elapsed

            print(
                "TRAIN ==> Epoch [%d/%d], Loss: %.12f \n DSC: %.6f, IOU: %.4f, Time: %.2f s, Total: %.2f min"
                % (
                    epoch_num + 1,
                    self.max_epoch,
                    train_batch_loss,
                    train_batch_dsc,
                    train_batch_iou,
                    elapsed,
                    total_elapsed / 60,
                )
            )

            valid_batch_loss = []
            valid_batch_dsc = []
            valid_batch_iou = []

            elapsed = timeit.default_timer()

            for counter in tqdm.tqdm(range(self.valid_steps)):

                (
                    batch_input_image_valid,
                    batch_label_mask_valid,
                ) = self.my_data_generator.generator(
                    batch_ind=counter, batch_size=self.batch_size, is_train=False,
                )

                ######
                ######      THE ONLY REAL DIFFERENCE WITH VALIDATION IS WE DON'T
                ######          CALL GRADIENT UPDATES OR APPLY AUGMENTAITON TO DATA. OTHERWISE ITS SAME
                ######
                batch_predicted_mask_valid = self.my_cnn(
                    batch_input_image_valid, training=False
                )

                loss_value_valid = self.get_loss(
                    mask_gt=batch_label_mask,
                    mask_pred=batch_predicted_mask,
                    epoch_num=epoch_num,
                )

                valid_loss_value = loss_value_valid.numpy()

                tmp_iou_valid, tmp_dsc_valid = get_dsc_iou(
                    mask_gt=batch_label_mask_valid,
                    mask_pred=batch_predicted_mask_valid.numpy(),
                )

                valid_batch_dsc.append(tmp_dsc_valid)
                valid_batch_iou.append(tmp_iou_valid)
                valid_batch_loss.append(valid_loss_value)

                with train_writer.as_default():
                    with tf.name_scope("VALID"):
                        tf.summary.scalar(
                            "DSC", tmp_dsc_valid, step=train_global_counter
                        )
                        tf.summary.scalar(
                            "IOU", tmp_iou_valid, step=train_global_counter
                        )
                        tf.summary.scalar(
                            "LOSS", valid_loss_value, step=train_global_counter
                        )

            valid_batch_loss = np.asarray(valid_batch_loss).mean()
            valid_epoch_loss.append(valid_batch_loss)

            valid_batch_dsc = np.asarray(valid_batch_dsc).mean()
            valid_epoch_dsc.append(valid_batch_dsc)
            valid_batch_iou = np.asarray(valid_batch_iou).mean()
            valid_epoch_iou.append(valid_batch_iou)

            total_elapsed = timeit.default_timer() - start_time
            elapsed = timeit.default_timer() - elapsed

            print(
                "VALID ==> Epoch [%d/%d], Loss: %.12f \n DSC: %.6f, IOU: %.4f, Time: %.2f s, Total: %.2f min"
                % (
                    epoch_num + 1,
                    self.max_epoch,
                    valid_batch_loss,
                    valid_batch_dsc,
                    valid_batch_iou,
                    elapsed,
                    total_elapsed / 60,
                )
            )

            # TELL THE MODEL TO SAVE ITSELF EVERY NTH ITERATION
            if (epoch_num + 1) % 10 == 0:
                print("SAVING MODEL . . . ")
                self.my_cnn.save(
                    os.path.join(self.model_save_name, "e_" + str(global_epoch_counter))
                )

            ######
            ######      AT THE END OF EVERY EPOCH, WRITE THE TENSORBOARD DATA TO FILE
            ######              BY DEFAULT, THE TF WRITER STORES EVERYTHING IN A CACHE
            ######                  WHICH ISN'T SAVED UNTIL YOU EXPLICITLY CALL FLUSH
            ######
            with train_writer.as_default():
                tf.summary.flush()

        print("SAVING MODEL . . . ")
        self.my_cnn.save(
            os.path.join(self.model_save_name, "e_" + str(global_epoch_counter))
        )

    def load_final(self):
        print("LOADING MODEL!!!!! \n\n\n")

        self.save_dir = os.path.join(self.project_folder, self.model_name)

        last_checkpoint = os.path.join(self.save_dir, "model.h5/FINAL")

        # print('FINAL EXISTS')
        # print(os.path.isdir(last_checkpoint))

        self.my_cnn = tf.keras.models.load_model(last_checkpoint)

        print("DONE LOADING MODEL!!!!! \n\n\n")

    def load(self):
        print("LOADING MODEL!!!!! \n\n\n")

        model_dir = os.path.join(self.model_save_name, "e_*")

        models_in_dir = glob.glob(model_dir)

        num_checkpoints = len(models_in_dir)
        model_checkpoints = np.zeros(num_checkpoints)
        for iter_checkpoint in range(num_checkpoints):
            model_str = models_in_dir[iter_checkpoint].split("/")[-1]
            model_epoch_num = int(model_str.split("_")[-1])
            model_checkpoints[iter_checkpoint] = model_epoch_num

        last_checkpoint = models_in_dir[np.argmax(model_checkpoints)]
        self.my_cnn = tf.keras.models.load_model(last_checkpoint)

        print("DONE LOADING MODEL!!!!! \n\n\n")

    def get_loss(self, mask_gt, mask_pred, epoch_num):

        mask_gt = tf.convert_to_tensor(mask_gt)
        mask_pred = tf.cast(mask_pred, mask_gt.dtype)

        # loss = self.custom_loss_2(
        #     mask_gt = mask_gt,
        #     mask_pred = mask_pred)

        if epoch_num < 10:
            loss = self.custom_loss_1(mask_gt=mask_gt, mask_pred=mask_pred)
        else:
            loss = self.custom_loss_2(mask_gt=mask_gt, mask_pred=mask_pred)

        return loss

    def custom_loss_1(self, mask_gt, mask_pred):

        alpha = 1
        beta = 1

        # NOTE:
        ######
        ######      I HAVE FOUND EMPIRICALLY THAT USING LARGER LEARNING RATE
        ######              AND SETTING A LARGE WEIGHTED CROSS-ENTROPY (~200) LOSS INITIALLY HELPS TO TRAIN THE MODEL WEIGHTS
        ######              TO ~0.8 DSC QUICKLY. IT ALSO HELPS TO ADD IN MEAN ABSOLUTE ERROR AND MEAN SQUARE ERROR
        ######              REGULARIZERS. ONCE IT FLATLINES THEN I DROP THE LEARNING RATE AND GO BACK TO NORMAL BINARY CROSS ENTROPY
        ######

        self.loss_weight = 200

        mask_gt = tf.convert_to_tensor(mask_gt)
        mask_pred = tf.cast(mask_pred, mask_gt.dtype)

        mse = self.mse_loss(mask_gt, mask_pred) * alpha
        mae = self.mae_loss(mask_gt, mask_pred) * beta
        wce = self.wce_loss(mask_gt, mask_pred)

        loss = wce + mse + mae

        return loss

    def custom_loss_2(self, mask_gt, mask_pred):

        self.loss_weight = 1

        mask_gt = tf.convert_to_tensor(mask_gt)
        mask_pred = tf.cast(mask_pred, mask_gt.dtype)
        wce = self.wce_loss(mask_gt, mask_pred)

        return wce

    def mse_loss(self, mask_gt, mask_pred):
        return tf.reduce_mean(tf.math.square(mask_gt - mask_pred))

    def wce_loss(self, mask_gt, mask_pred):
        return tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=mask_gt, logits=mask_pred, pos_weight=self.loss_weight
            )
        )

    def mae_loss(self, mask_gt, mask_pred):
        return tf.reduce_mean(tf.math.abs(mask_gt - mask_pred))


def print_image_metrics(iou_list, dsc_list):

    iou_list = np.asarray(iou_list).flatten()
    dsc_list = np.asarray(dsc_list).flatten()

    dsc_mean = np.mean(dsc_list)
    dsc_std = np.std(dsc_list)
    dsc_max = np.amax(dsc_list)
    dsc_min = np.amin(dsc_list)

    iou_mean = np.mean(iou_list)
    iou_std = np.std(iou_list)
    iou_max = np.amax(iou_list)
    iou_min = np.amin(iou_list)

    print("\n\n IOU")
    print("MEAN:   " + str(iou_mean))
    print("STD:   " + str(iou_std))
    print("MAX:   " + str(iou_max))
    print("MIN:   " + str(iou_min))

    print("\n\n DSC")
    print("MEAN:   " + str(dsc_mean))
    print("STD:   " + str(dsc_std))
    print("MAX:   " + str(dsc_max))
    print("MIN:   " + str(dsc_min))


def main():
    """
        Tests the CNN.

    """

    project_folder = os.getcwd()
    study_dir = os.path.join(os.path.dirname(project_folder), "DATA")

    # batch_size = 18 # 18 worked well for echo1
    batch_size = 18
    max_epoch = 300
    lr = 1e-3

    # image_channels = [ 'Echo1' ]
    image_channels = ["IR-rUTE", "Echo2", "Echo1"]
    # image_channels = [ 'Echo2' ]
    # image_channels = [ 'Echo1' ]
    # image_channels = [ 'IR-rUTE' ]
    name = ""

    if "Echo2" in image_channels:
        name = name + "Echo2_"
    if "Echo1" in image_channels:
        name = name + "Echo1_"
    if "IR-rUTE" in image_channels:
        name = name + "IR-rUTE_"

    name = name + "b_{}_e_{}_lr_{}".format(str(batch_size), str(max_epoch), str(lr))

    name = "BEST_Echo1_Echo2_IR-rUTE"

    conv_net = CNN(
        project_folder=project_folder,
        study_dir=study_dir,
        channel_str=image_channels,
        batch_size=batch_size,
        max_epoch=max_epoch,
        model_name=name,
        learn_rate=lr,
    )

    #
    #
    #
    #
    #
    # conv_net.train()
    # conv_net.load()
    # conv_net.train()
    #
    #
    #
    #
    #
    conv_net.load_final()

    the_data_generator = data_generator(
        create_new_dataset=False,
        valid_split=0.2,
        channel_str=image_channels,
        seed=1,
        noise=0,
        zoomRange=(1, 1),
        rotationRange=0,
        shearRange=0,
        widthShiftRange=0,
        heightShiftRange=0,
        flipLR=False,
        flipUD=False,
    )
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
    ) = the_data_generator.get_info(return_names=True)

    batch_iou = []
    batch_dsc = []

    for iter_scan in range(int(num_train // batch_size)):

        # print('TRAIN SET, ITER SCAN '+str(iter_scan))

        (echo_image, mask) = the_data_generator.generator(
            batch_ind=iter_scan,
            # batch_size = num_valid-1,
            batch_size=batch_size,
            is_train=True,
        )

        batch_predicted_mask = conv_net.forward_pass(echo_image)

        batch_predicted_mask = batch_predicted_mask.numpy() > 0.5
        (iou_batch_mean, dsc_batch_mean, iou_vals, dsc_vals) = get_dsc_iou(
            mask_gt=mask,
            mask_pred=batch_predicted_mask,
            return_all_vals=True,
            only_largest_structure=True,
        )

        # print('VALIDATION IOU IS '+str(iou_batch_mean))
        # print('VALIDATION DSC IS '+str(dsc_batch_mean)+'\n\n\n\n')

        batch_dsc.append(dsc_vals)
        batch_iou.append(iou_vals)

        prep_images_for_plot(
            input_image=echo_image,
            predicted_mask=batch_predicted_mask,
            ground_truth=mask,
        )

    print("\n\nTRAIN!!!! \n\n")
    print_image_metrics(iou_list=iou_vals, dsc_list=dsc_vals)

    batch_iou = []
    batch_dsc = []

    for iter_scan in range(int(num_valid // batch_size)):

        # print('VALID SET, ITER SCAN '+str(iter_scan))

        (echo_image, mask) = the_data_generator.generator(
            batch_ind=iter_scan,
            # batch_size = num_valid-1,
            batch_size=batch_size,
            is_train=False,
        )

        batch_predicted_mask = conv_net.forward_pass(echo_image)

        batch_predicted_mask = batch_predicted_mask.numpy() > 0.5
        (iou_batch_mean, dsc_batch_mean, iou_vals, dsc_vals) = get_dsc_iou(
            mask_gt=mask,
            mask_pred=batch_predicted_mask,
            return_all_vals=True,
            only_largest_structure=True,
        )

        # print('VALIDATION IOU IS '+str(iou_batch_mean))
        # print('VALIDATION DSC IS '+str(dsc_batch_mean)+'\n\n\n\n')

        batch_dsc.append(dsc_vals)
        batch_iou.append(iou_vals)

        prep_images_for_plot(
            input_image=echo_image,
            predicted_mask=batch_predicted_mask,
            ground_truth=mask,
        )

    print("\n\n VALIDATION!!!! \n\n")
    print_image_metrics(iou_list=iou_vals, dsc_list=dsc_vals)


if __name__ == "__main__":

    main()
