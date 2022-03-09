from __future__ import absolute_import
import math
import tensorflow as tf

from tensorflow.keras.regularizers import l2

from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker


def classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]

    pool = tf.keras.layers.GlobalAveragePooling2D()(
        base_model) if pooling == 'avg' else tf.keras.layers.GlobalMaxPool2D()(base_model)

    pre_classifier = tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )(
        pool
    )  # was 128

    drop_out_class = tf.keras.layers.Dropout(dropout)(pre_classifier)
    classifier = tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)
    res.preprocessing = backbone_.preprocessing
    return res


def wise_srnet_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pool_size=3, size=512, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]

    avg = tf.keras.layers.AveragePooling2D(
        pool_size, padding='valid')(base_model) if pooling == 'avg' else tf.keras.layers.MaxPool2D(
        pool_size, padding='valid')(base_model)

    out_size = input_tensor.shape[1]/(math.pow(2, deep_layer))
    pool_out_size = math.floor((out_size - pool_size)/pool_size + 1)

    depthw = tf.keras.layers.DepthwiseConv2D(pool_out_size,
                                             activation=pool_activation,
                                             depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                 mean=0.0, stddev=0.01),
                                             bias_initializer=tf.keras.initializers.Zeros(), depthwise_constraint=tf.keras.constraints.NonNeg())(avg)
    flat = tf.keras.layers.Flatten()(depthw)

    pre_classifier = tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )(
        flat
    )  #

    drop_out_class = tf.keras.layers.Dropout(dropout)(pre_classifier)
    classifier = tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)
    res.preprocessing = backbone_.preprocessing
    return res


def distiller_classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pool_size=3, macrofeatures_number=8, size=64, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]

    avg = tf.keras.layers.AveragePooling2D(
        pool_size, padding='valid')(base_model) if pooling == 'avg' else tf.keras.layers.MaxPool2D(
        pool_size, padding='valid')(base_model) 

    out_size = input_tensor.shape[1]/(math.pow(2, deep_layer))
    pool_out_size = math.floor((out_size - pool_size)/pool_size + 1)

    depthw = []

    for i in range(macrofeatures_number):
        d = tf.keras.layers.DepthwiseConv2D(pool_out_size,
                                            activation=pool_activation,
                                            depthwise_initializer=tf.keras.initializers.RandomNormal(
                                                mean=0.0, stddev=0.01),
                                            bias_initializer=tf.keras.initializers.Zeros(), depthwise_constraint=tf.keras.constraints.NonNeg())(avg)
        flat = tf.keras.layers.Flatten()(d)
        drop_out = tf.keras.layers.Dropout(dropout)(flat)
        depthw.append(drop_out)

    concatenate = tf.keras.layers.Concatenate()(depthw)

    pre_classifier = tf.keras.layers.Dense(
        size,
        activation=activation,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
    )(
        concatenate
    )  #

    drop_out_class = tf.keras.layers.Dropout(dropout)(pre_classifier)
    classifier = tf.keras.layers.Dense(
        n_classes, activation='softmax', kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)(drop_out_class)

    res = tf.keras.models.Model(inputs=input_tensor, outputs=classifier)
    res.preprocessing = backbone_.preprocessing
    return res
