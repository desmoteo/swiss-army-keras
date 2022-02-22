
from __future__ import absolute_import
from multiprocessing.spawn import prepare

from sklearn import preprocessing

from swiss_army_keras.layer_utils import *
from swiss_army_keras.activations import GELU, Snake
from swiss_army_keras._backbone_zoo import backbone_zoo, bach_norm_checker
from swiss_army_keras._model_unet_2d import UNET_left, UNET_right

from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeNormal

from tensorflow.nn import relu


def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return relu(x)


def depth_convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
    stride=1,
    depth_padding='same',
    epsilon=1e-5
):

    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(dilation_rate, dilation_rate),
                        padding=depth_padding, use_bias=False, name=f'depthwise_{dilation_rate}')(block_input)
    x = BatchNormalization(
        name=f'depthwise_BN__{dilation_rate}', epsilon=epsilon)(x)
    x = Activation(relu)(x)
    x = Conv2D(num_filters, (1, 1), padding='same',
               use_bias=False, name=f'pointwise_{dilation_rate}')(x)
    x = BatchNormalization(
        name=f'pointwise_BN_{dilation_rate}', epsilon=epsilon)(x)
    x = Activation(relu)(x)
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return relu(x)


def DilatedSpatialPyramidPooling(dspp_input, atrous_rates):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)

    outputs = [out_pool, out_1]

    for rate in atrous_rates:
        out = depth_convolution_block(
            dspp_input, kernel_size=3, dilation_rate=rate)
        outputs.append(out)

    x = Concatenate(axis=-1)(outputs)
    output = convolution_block(x, kernel_size=1)
    return output


def deeplab_v3_plus(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
                    deep_layer=5, shallow_layer=2, atrous_rates=[6, 12, 18],
                    stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                    backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='deeplab_v3_plus'):
    '''
    The base of UNET 3+ with an optional ImagNet-trained backbone.

    unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='deeplab_v3_plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.        
        filter_num_down: a list that defines the number of filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.          
        activation: one of the `tensorflow.keras.layers` or `swiss_army_keras.activations` interfaces, e.g., ReLU                
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.     
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.   

    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with the first tensor has the smallest size. 

    '''

    depth_ = len(filter_num_down)

    X_encoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):

            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

            preprocessing = backbone_.preprocessing

        # for other backbones
        else:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder) + 1

            preprocessing = backbone_.preprocessing

    x = DilatedSpatialPyramidPooling(X_encoder[deep_layer-1], atrous_rates)

    print(input_tensor)

    input_a = UpSampling2D(
        size=(input_tensor.shape[1] // 4 // x.shape[1],
              input_tensor.shape[2] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = X_encoder[shallow_layer-1]

    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(n_labels, kernel_size=(1, 1), padding="same")(x)

    m = Model([input_tensor, ], [model_output, ])

    m.preprocessing = preprocessing

    return m


def deeplab_v3_plus_lite(input_tensor, n_labels, filter_num_down=[64, 128, 256, 512, 1024],
                         deep_layer=5, shallow_layer=2, atrous_rates=[6, 12, 18],
                         stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True,
                         backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='deeplab_v3_plus_lite'):
    '''
    The base of UNET 3+ with an optional ImagNet-trained backbone.

    unet_3plus_2d_base(input_tensor, filter_num_down, filter_num_skip, filter_num_aggregate, 
                       stack_num_down=2, stack_num_up=1, activation='ReLU', batch_norm=False, pool=True, unpool=True, 
                       backbone=None, weights='imagenet', freeze_backbone=True, freeze_batch_norm=True, name='unet3plus')

    ----------
    Huang, H., Lin, L., Tong, R., Hu, H., Zhang, Q., Iwamoto, Y., Han, X., Chen, Y.W. and Wu, J., 2020. 
    UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation. 
    In ICASSP 2020-2020 IEEE International Conference on Acoustics, 
    Speech and Signal Processing (ICASSP) (pp. 1055-1059). IEEE.

    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.        
        filter_num_down: a list that defines the number of filters for each 
                         downsampling level. e.g., `[64, 128, 256, 512, 1024]`.
                         the network depth is expected as `len(filter_num_down)`
        filter_num_skip: a list that defines the number of filters after each 
                         full-scale skip connection. Number of elements is expected to be `depth-1`.
                         i.e., the bottom level is not included.
                         * Huang et al. (2020) applied the same numbers for all levels. 
                           e.g., `[64, 64, 64, 64]`.
        filter_num_aggregate: an int that defines the number of channels of full-scale aggregations.
        stack_num_down: number of convolutional layers per downsampling level/block. 
        stack_num_up: number of convolutional layers (after full-scale concat) per upsampling level/block.          
        activation: one of the `tensorflow.keras.layers` or `swiss_army_keras.activations` interfaces, e.g., ReLU                
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.     
        name: prefix of the created keras model and its layers.

        ---------- (keywords of backbone options) ----------
        backbone_name: the bakcbone model name. Should be one of the `tensorflow.keras.applications` class.
                       None (default) means no backbone. 
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0-7]
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        freeze_backbone: True for a frozen backbone.
        freeze_batch_norm: False for not freezing batch normalization layers.   

    * Downsampling is achieved through maxpooling and can be replaced by strided convolutional layers here.
    * Upsampling is achieved through bilinear interpolation and can be replaced by transpose convolutional layers here.

    Output
    ----------
        A list of tensors with the first/second/third tensor obtained from 
        the deepest/second deepest/third deepest upsampling block, etc.
        * The feature map sizes of these tensors are different, 
          with the first tensor has the smallest size. 

    '''

    depth_ = len(filter_num_down)

    X_encoder = []

    # no backbone cases
    if backbone is None:

        X = input_tensor

        # stacked conv2d before downsampling
        X = CONV_stack(X, filter_num_down[0], kernel_size=3, stack_num=stack_num_down,
                       activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
        X_encoder.append(X)

        # downsampling levels
        for i, f in enumerate(filter_num_down[1:]):

            # UNET-like downsampling
            X = UNET_left(X, f, kernel_size=3, stack_num=stack_num_down, activation=activation,
                          pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i+1))
            X_encoder.append(X)

        preprocessing = dummy_preprocessing

    else:
        # handling VGG16 and VGG19 separately
        if 'VGG' in backbone:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, depth_, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder)

            preprocessing = backbone_.preprocessing

        # for other backbones
        else:
            backbone_ = backbone_zoo(
                backbone, weights, input_tensor, deep_layer, freeze_backbone, freeze_batch_norm)
            # collecting backbone feature maps
            X_encoder = backbone_([input_tensor, ])
            depth_encode = len(X_encoder) + 1

            preprocessing = backbone_.preprocessing

    x = DilatedSpatialPyramidPooling(X_encoder[deep_layer-1], atrous_rates)

    print(input_tensor)

    input_a = UpSampling2D(
        size=(input_tensor.shape[1] // 4 // x.shape[1],
              input_tensor.shape[2] // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = X_encoder[shallow_layer-1]

    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    """x = UpSampling2D(
        size=(input_tensor.shape[1] // x.shape[1],
              input_tensor.shape[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)"""
    model_output = Conv2D(n_labels, kernel_size=(1, 1), padding="same")(x)

    m = Model([input_tensor, ], [model_output, ])

    m.preprocessing = preprocessing

    return m
