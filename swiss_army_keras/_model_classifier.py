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


def conv_block(x, filters, kernel_size, strides, activation=tf.keras.layers.LeakyReLU(0.2)):
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides,
                                        padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    return x


def res_block(x, resizer_filters=16):
    inputs = x
    x = conv_block(x, resizer_filters, 3, 1)
    x = conv_block(x, resizer_filters, 3, 1, activation=None)
    return tf.keras.layers.Add()([inputs, x])


def learnable_resizer_classifier(input_tensor, n_classes, target_size=(480, 480), resizer_filters=16, resizer_kernel_size=5, num_res_blocks=1,
                                 interpolation='bilinear', backbone='EfficientNetLiteB0', weights='imagenet', freeze_backbone=True,
                                 freeze_batch_norm=True, name='classifier', deep_layer=5,
                                 pooling='avg', size=1024, activation="swish", kernel_regularizer=l2(0.001),
                                 bias_regularizer=l2(0.001), dropout=0.3):

    inputs = tf.keras.layers.Input(shape=[target_size[0], target_size[1], 3])

    # First, perform naive resizing.
    naive_resize = tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    )(input_tensor)

    # First convolution block without batch normalization.
    x = tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=resizer_kernel_size,
                               strides=1, padding="same")(input_tensor)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Second convolution block with batch normalization.
    x = tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=1,
                               strides=1, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    )(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck, resizer_filters)

    # Projection.
    x = tf.keras.layers.SeparableConv2D(
        filters=resizer_filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip connection.
    x = tf.keras.layers.Add()([bottleneck, x])

    # Final resized image.
    x = tf.keras.layers.Conv2D(
        filters=3, kernel_size=resizer_kernel_size, strides=1, padding="same")(x)
    final_resize = tf.keras.layers.Add()([naive_resize, x])

    backbone_ = backbone_zoo(
        backbone, weights, inputs, deep_layer, freeze_backbone, freeze_batch_norm)  # , return_outputs=True)

    base_model = backbone_([final_resize, ])[deep_layer-1]

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


def learnable_resizer_distiller_classifier(input_tensor, n_classes, target_size=(480, 480), resizer_filters=16, resizer_kernel_size=5, num_res_blocks=1,
                                           interpolation='bilinear',  backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', pool_size=3, macrofeatures_number=8, size=64, activation="swish", pool_activation=None, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    inputs = tf.keras.layers.Input(shape=[target_size[0], target_size[1], 3])

    # First, perform naive resizing.
    naive_resize = tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    )(input_tensor)

    # First convolution block without batch normalization.
    x = tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=resizer_kernel_size,
                               strides=1, padding="same")(input_tensor)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    # Second convolution block with batch normalization.
    x = tf.keras.layers.Conv2D(filters=resizer_filters, kernel_size=1,
                               strides=1, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = tf.keras.layers.Resizing(
        *target_size, interpolation=interpolation
    )(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck, resizer_filters)

    # Projection.
    x = tf.keras.layers.SeparableConv2D(
        filters=resizer_filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip connection.
    x = tf.keras.layers.Add()([bottleneck, x])

    # Final resized image.
    x = tf.keras.layers.Conv2D(
        filters=3, kernel_size=resizer_kernel_size, strides=1, padding="same")(x)
    final_resize = tf.keras.layers.Add()([naive_resize, x])

    backbone_ = backbone_zoo(
        backbone, weights, inputs, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([final_resize, ])[deep_layer-1]

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
