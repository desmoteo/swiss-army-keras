from __future__ import absolute_import

import tensorflow as tf

from tensorflow.keras.applications import *
from efficientnet_lite import *
from tensorflow.keras.models import Model
from tensorflow.python.framework.tensor_shape import as_dimension

from swiss_army_keras.utils import freeze_model

import tensorflow_hub as hub

import warnings

keras_layer_cadidates = {
    'MobileNetV2' : ('block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'out_relu'),
    #'MobileNetV3Large' : ('re_lu_2', 're_lu_6', 're_lu_15', 're_lu_29', 're_lu_38'),
    'MobileNetV3Large' : ('re_lu_2', 're_lu_6', 're_lu_12', 're_lu_22', 're_lu_30'),
    'VGG16': ('block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'),
    'VGG19': ('block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'),
    'ResNet50': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'),
    'ResNet101': ('conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block23_out', 'conv5_block3_out'),
    'ResNet152': ('conv1_relu', 'conv2_block3_out', 'conv3_block8_out', 'conv4_block36_out', 'conv5_block3_out'),
    'ResNet50V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block6_1_relu', 'post_relu'),
    'ResNet101V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block4_1_relu', 'conv4_block23_1_relu', 'post_relu'),
    'ResNet152V2': ('conv1_conv', 'conv2_block3_1_relu', 'conv3_block8_1_relu', 'conv4_block36_1_relu', 'post_relu'),
    'DenseNet121': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet169': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'DenseNet201': ('conv1/relu', 'pool2_conv', 'pool3_conv', 'pool4_conv', 'relu'),
    'EfficientNetB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB5': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB6': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetB7': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'top_activation'),
    'EfficientNetLiteB0': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
    'EfficientNetLiteB1': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
    'EfficientNetLiteB2': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
    'EfficientNetLiteB3': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
    'EfficientNetLiteB4': ('block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation', 'block7a_activation'),
    }


keras_normalization_layers = {
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'MobileNetV3Large': tf.keras.applications.mobilenet_v3.preprocess_input,
    'VGG16': tf.keras.applications.vgg16.preprocess_input,
    'VGG19': tf.keras.applications.vgg19.preprocess_input,
    'ResNet50': tf.keras.applications.resnet.preprocess_input,
    'ResNet101': tf.keras.applications.resnet.preprocess_input,
    'ResNet152': tf.keras.applications.resnet.preprocess_input,
    'ResNet50V2': tf.keras.applications.resnet.preprocess_input,
    'ResNet101V2': tf.keras.applications.resnet.preprocess_input,
    'ResNet152V2': tf.keras.applications.resnet.preprocess_input,
    'DenseNet121': tf.keras.applications.densenet.preprocess_input,
    'DenseNet169': tf.keras.applications.densenet.preprocess_input,
    'DenseNet201': tf.keras.applications.densenet.preprocess_input,
    'EfficientNetB0': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB1': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB2': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB3': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB4': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB5': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB6': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB7': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetLiteB0': get_preprocessing_layer(),
    'EfficientNetLiteB1': get_preprocessing_layer(),
    'EfficientNetLiteB2': get_preprocessing_layer(),
    'EfficientNetLiteB3': get_preprocessing_layer(),
    'EfficientNetLiteB4': get_preprocessing_layer(),
}

keras_kwargs = {
    'MobileNetV2' : {},
    'MobileNetV3Large': {'minimalistic': True},
    'VGG16': {},
    'VGG19': {},
    'ResNet50': {},
    'ResNet101': {},
    'ResNet152': {},
    'ResNet50V2': {},
    'ResNet101V2': {},
    'ResNet152V2': {},
    'DenseNet121': {},
    'DenseNet169': {},
    'DenseNet201': {},
    'EfficientNetB0': {},
    'EfficientNetB1': {},
    'EfficientNetB2': {},
    'EfficientNetB3': {},
    'EfficientNetB4': {},
    'EfficientNetB5': {},
    'EfficientNetB6': {},
    'EfficientNetB7': {},
    'EfficientNetLiteB0': {},
    'EfficientNetLiteB1': {},
    'EfficientNetLiteB2': {},
    'EfficientNetLiteB3': {},
    'EfficientNetLiteB4': {},
}


hub_layer_candidates = {
    'EfficientNet-lite0' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite1' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite2' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite3' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),
    'EfficientNet-lite4' : ('reduction_1/expansion_output', 'reduction_2/expansion_output', 'reduction_3/expansion_output', 'reduction_4/expansion_output', 'reduction_5/expansion_output'),

}

backbone_hub_urls = {
    'EfficientNet-lite0' : 'https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2',
    'EfficientNet-lite1' : 'https://tfhub.dev/tensorflow/efficientnet/lite1/feature-vector/2',
    'EfficientNet-lite2' : 'https://tfhub.dev/tensorflow/efficientnet/lite2/feature-vector/2',
    'EfficientNet-lite3' : 'https://tfhub.dev/tensorflow/efficientnet/lite3/feature-vector/2',
    'EfficientNet-lite4' : 'https://tfhub.dev/tensorflow/efficientnet/lite4/feature-vector/2',
}

backbone_hub_signatures = {
    'EfficientNet-lite0' : 'image_feature_vector',
    'EfficientNet-lite1' : 'image_feature_vector',
    'EfficientNet-lite2' : 'image_feature_vector',
    'EfficientNet-lite3' : 'image_feature_vector',
    'EfficientNet-lite4' : 'image_feature_vector',
}

def bach_norm_checker(backbone_name, batch_norm):
    '''batch norm checker'''
    if 'VGG' in backbone_name:
        batch_norm_backbone = False
    else:
        batch_norm_backbone = True
        
    if batch_norm_backbone != batch_norm:       
        if batch_norm_backbone:    
            param_mismatch = "\n\nBackbone {} uses batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
        else:
            param_mismatch = "\n\nBackbone {} does not use batch norm, but other layers received batch_norm={}".format(backbone_name, batch_norm)
            
        warnings.warn(param_mismatch);
        
def backbone_zoo(backbone_name, weights, input_tensor, depth, freeze_backbone, freeze_batch_norm, return_outputs=False):
    '''
    Configuring a user specified encoder model based on the `tensorflow.keras.applications`
    
    Input
    ----------
        backbone_name: the bakcbone model name. Expected as one of the `tensorflow.keras.applications` class.
                       Currently supported backbones are:
                       (1) VGG16, VGG19
                       (2) ResNet50, ResNet101, ResNet152
                       (3) ResNet50V2, ResNet101V2, ResNet152V2
                       (4) DenseNet121, DenseNet169, DenseNet201
                       (5) EfficientNetB[0,7]
                       
        weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                 or the path to the weights file to be loaded.
        input_tensor: the input tensor 
        depth: number of encoded feature maps. 
               If four dwonsampling levels are needed, then depth=4.
        
        freeze_backbone: True for a frozen backbone
        freeze_batch_norm: False for not freezing batch normalization layers.
        
    Output
    ----------
        model: a keras backbone model.
        
    '''
    

    # ----- #
    

    using_hub = False
    if 'hub://' in backbone_name:
        using_hub = True
        backbone_name = backbone_name[6:]
        print(backbone_name)
        backbone_ = hub.KerasLayer(backbone_hub_urls[backbone_name], input_shape=(513, 513, 3), signature=backbone_hub_signatures[backbone_name], signature_outputs_as_dict=True, trainable=False)
        backbone_ = backbone_(input_tensor)
        cadidate = hub_layer_candidates[backbone_name]
        for c in cadidate:
            print(backbone_[c])



    else:
        backbone_func = eval(backbone_name)
        backbone_ = backbone_func(include_top=False, weights=weights, input_tensor=input_tensor, pooling=None, **keras_kwargs[backbone_name])
    
        cadidate = keras_layer_cadidates[backbone_name]
    
    # ----- #
    # depth checking
    depth_max = len(cadidate)
    if depth > depth_max:
        depth = depth_max

    X_skip = []
    
    for i in range(depth):
        if using_hub:
            X_skip.append(backbone_[cadidate[i]])
        else:
            X_skip.append(backbone_.get_layer(cadidate[i]).output)
            
    if return_outputs:
        return X_skip, keras_normalization_layers[backbone_name]
        
    model = Model(inputs=[input_tensor,], outputs=X_skip, name='{}_backbone'.format(backbone_name))
    
    if freeze_backbone:
        model = freeze_model(model, freeze_batch_norm=freeze_batch_norm)
    
    model.preprocessing = keras_normalization_layers[backbone_name]
    return model
