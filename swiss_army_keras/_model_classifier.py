import tensorflow as tf

from tensorflow.keras.regularizers import l2

from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker


def classifier(input_tensor, n_classes, backbone='MobileNetV3Large', weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='classifier', deep_layer=5, pooling='avg', size=1024, activation="swish", kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), dropout=0.3):

    backbone_ = backbone_zoo(
        backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)

    base_model = backbone_([input_tensor, ])[deep_layer-1]

    pool = tf.keras.layers.GlobalAveragePooling2D()(
        base_model) if pooling == 'avg' else tf.keras.layers.GlobalAveragePooling2D()(base_model)

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
