# swiss-army-keras

A library to help with the development of AI models with Keras, with a focus on edge / IoT applications. Based originally on https://github.com/yingkaisha/keras-unet-collection (see the [README.md](https://github.com/waterviewsrl/swiss-army-keras/blob/main/README-keras-unet-collection.md))

## Summary

This library wants you to focus on dataset, model architecture and hyperparameters tuning, without worring about the rest. 

It provides several helper classes which help in the development of CNN based AI models for edge IoT applications, where resources are limited and model quantization is reccomended. 

The main features of the library are the following:

1) Flexible, efficient and scalable Dataset management with augmentation pipelines leveraging albumentations (https://albumentations.ai/) and td-data (https://www.tensorflow.org/guide/data)
2) Training driver with builtin callbacks, configurable backbone unfreezing, and quantized model generation
3) Helper classes to easiliy combine pretrained backbones for Edge AI applications with the desired segmentation and classification architectures
4) Additional loss functions and optimizers which are not part of the Keras distribution, as for now 

## Installation

### Dependencies

tensorflow>=2.4.1 (cpu or gpu) must be installed.

Install the dependency git+https://github.com/waterviewsrl/efficientnet-lite-keras.git (forked from [sebastian-sz/efficientnet-lite-keras](https://github.com/sebastian-sz/efficientnet-lite-keras)  to simplify requirements and solve minor import issues):

    pip3 install git+https://github.com/waterviewsrl/efficientnet-lite-keras.git

### Install with pip

You can install directly by pypi with pip:

    pip3 install swiss-army-keras

