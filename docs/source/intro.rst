Introduction
============

A library to help with the development of AI models with Keras, with a focus on edge / IoT applications. 

Motivation
**********

Building CNN experiments with flexible and scalable augmentation pipelines, with the ability to cobine different encoder and decoder architectures, 
can require a huge initial effort. 
This is even more true in the case of edge-AI models, where models need to be build taking into account more severe constraints and quantization is usually required.

This library wants you to focus on dataset, model architecture and hyperparameters tuning, without worring about the rest.

It provides several helper classes which help in the development of CNN based AI models for edge IoT applications, where resources are limited and model quantization is reccomended.

Features
********

The main features of the library are the following:

1. Flexible, efficient and scalable Dataset management with augmentation pipelines leveraging albumentations (https://albumentations.ai/) and td-data (https://www.tensorflow.org/guide/data)
2. Training driver with builtin callbacks, configurable backbone unfreezing, and quantized model generation
3. Helper classes to easiliy combine pretrained backbones for Edge AI applications with the desired segmentation and classification architectures
4. Additional loss functions and optimizers which are not part of the Keras distribution, as for now

