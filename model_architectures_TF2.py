###############################
###############################
###############################
###############################
#########
#########
#########   CREATED BY: BRANDON CLINTON JONES
#########		AND CARLOS ADOLFO OSUNA
#########       APRIL 23, 2020
#########
#########
#########
###############################
###############################
###############################
###############################

import tensorflow as tf
from tensorflow.keras.layers import *
import os
import numpy as np
import tensorflow as tf


def conv_block_simple_2d(
    prevlayer,
    num_filters,
    prefix,
    kernel_size=(3, 3),
    initializer="he_normal",
    strides=(1, 1),
):

    conv = Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer=initializer,
        strides=strides,
        name=prefix + "_conv",
        data_format="channels_first",
    )(prevlayer)

    conv = BatchNormalization(name=prefix + "_bn", axis=1)(conv)

    conv = tf.nn.relu(conv, name=prefix + "_activation")

    return conv


def UNet_FINAL(input_tensor, output_tensor_channels=1):
    # print('\n\n\n')
    # print(input_tensor)
    inputs = Input(input_tensor)

    # print('\n\n\n')
    # print(inputs)
    # print('\n\n\n')

    mp_param = (2, 2)
    stride_param = (2, 2)
    d_format = "channels_first"
    pad = "same"
    kern = (3, 3)

    droput_rate = 0.2

    filt = (64, 128, 256, 512, 1024)

    conv1 = conv_block_simple_2d(prevlayer=inputs, num_filters=filt[0], prefix="conv1")
    # conv1 = Dropout(rate=droput_rate)(conv1)
    conv1 = conv_block_simple_2d(prevlayer=conv1, num_filters=filt[0], prefix="conv1_1")
    pool1 = MaxPooling2D(
        pool_size=mp_param,
        strides=stride_param,
        padding="same",
        data_format="channels_first",
        name="pool1",
    )(conv1)

    conv2 = conv_block_simple_2d(prevlayer=pool1, num_filters=filt[1], prefix="conv2")
    # conv2 = Dropout(rate=droput_rate)(conv2)
    conv2 = conv_block_simple_2d(prevlayer=conv2, num_filters=filt[1], prefix="conv2_1")
    pool2 = MaxPooling2D(
        pool_size=mp_param,
        strides=stride_param,
        padding="same",
        data_format="channels_first",
        name="pool2",
    )(conv2)

    conv3 = conv_block_simple_2d(prevlayer=pool2, num_filters=filt[2], prefix="conv3")
    # conv3 = Dropout(rate=droput_rate)(conv3)
    conv3 = conv_block_simple_2d(prevlayer=conv3, num_filters=filt[2], prefix="conv3_1")
    pool3 = MaxPooling2D(
        pool_size=mp_param,
        strides=stride_param,
        padding="same",
        data_format="channels_first",
        name="pool3",
    )(conv3)

    conv4 = conv_block_simple_2d(prevlayer=pool3, num_filters=filt[3], prefix="conv4")
    # conv4 = Dropout(rate=droput_rate)(conv4)
    conv4 = conv_block_simple_2d(prevlayer=conv4, num_filters=filt[3], prefix="conv4_1")
    pool4 = MaxPooling2D(
        pool_size=mp_param,
        strides=stride_param,
        padding="same",
        data_format="channels_first",
        name="pool4",
    )(conv4)

    conv5 = conv_block_simple_2d(prevlayer=pool4, num_filters=filt[4], prefix="conv_5")
    # conv5 = Dropout(rate=droput_rate)(conv5)
    conv5 = conv_block_simple_2d(
        prevlayer=conv5, num_filters=filt[4], prefix="conv_5_1"
    )

    up6 = Conv2DTranspose(
        filters=filt[3],
        kernel_size=kern,
        strides=(2, 2),
        padding="same",
        data_format="channels_first",
    )(conv5)
    up6 = concatenate([up6, conv4], axis=1)
    conv6 = conv_block_simple_2d(prevlayer=up6, num_filters=filt[3], prefix="conv6_1")
    # conv6 = Dropout(rate=droput_rate)(conv6)
    conv6 = conv_block_simple_2d(prevlayer=conv6, num_filters=filt[2], prefix="conv6_2")

    up7 = Conv2DTranspose(
        filters=filt[2],
        kernel_size=kern,
        strides=(2, 2),
        padding="same",
        data_format="channels_first",
    )(conv6)
    up7 = concatenate([up7, conv3], axis=1)
    conv7 = conv_block_simple_2d(prevlayer=up7, num_filters=filt[2], prefix="conv7_1")
    # conv7 = Dropout(rate=droput_rate)(conv7)
    conv7 = conv_block_simple_2d(prevlayer=conv7, num_filters=filt[1], prefix="conv7_2")

    up8 = Conv2DTranspose(
        filters=filt[1],
        kernel_size=kern,
        strides=(2, 2),
        padding="same",
        data_format="channels_first",
    )(conv7)
    up8 = concatenate([up8, conv2], axis=1)
    conv8 = conv_block_simple_2d(prevlayer=up8, num_filters=filt[1], prefix="conv8_1")
    # conv8 = Dropout(rate=droput_rate)(conv8)
    conv8 = conv_block_simple_2d(prevlayer=conv8, num_filters=filt[0], prefix="conv8_2")

    up9 = Conv2DTranspose(
        filters=filt[0],
        kernel_size=kern,
        strides=(2, 2),
        padding="same",
        data_format="channels_first",
    )(conv8)
    up9 = concatenate([up9, conv1], axis=1)
    conv9 = conv_block_simple_2d(prevlayer=up9, num_filters=filt[0], prefix="conv9_1")
    # conv9 = Dropout(rate=droput_rate)(conv9)
    conv9 = conv_block_simple_2d(prevlayer=conv9, num_filters=filt[0], prefix="conv9_2")

    prediction = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        activation="sigmoid",
        name="prediction",
        data_format=d_format,
    )(conv9)

    return inputs, prediction
