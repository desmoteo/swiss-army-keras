import os
import cv2
from functools import partial

import albumentations as albu

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import rescale, resize, downscale_local_mean


# helper function for data visualization


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class SegmentationAlbumentationsDataLoader:

    def __init__(self, dataset_path, train_augmentations=None, val_augmentations=None, test_augmentations=None, images_dir='images', masks_dir='annotations', width=512, height=512, batch_size=16, num_classes=2, mask_downsample=None, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, label_shift=0):

        self.ids = os.listdir(os.path.join(dataset_path, images_dir))
        self.num_classes = num_classes
        #self.mask_ids = os.listdir(masks_dir)
        images_frames = [os.path.join(dataset_path, images_dir, image_id)
                         for image_id in self.ids]
        labels_frames = [os.path.join(dataset_path, masks_dir, image_id.split(
            image_id.split('.')[-1])[0]+'png') for image_id in self.ids]

        self.configs = {}
        self.images = images_frames
        self.labels = labels_frames
        self.width = width
        self.height = height
        self.width = width
        self.height = height

        self.buffer_size = buffer_size

        self.train_val_test_split = []

        for val in train_val_test_split:
            self.train_val_test_split.append(int(len(self.images)*val))

        self.batch_size = batch_size

        self.label_shift = label_shift
        
        self.mask_downsample = mask_downsample

        self.augmentations = {}

        self.augmentations['train'] = train_augmentations if train_augmentations else self.get_default_augmentation()
        self.augmentations['val'] = val_augmentations if val_augmentations else self.get_default_augmentation()
        self.augmentations['test'] = test_augmentations if test_augmentations else self.get_default_augmentation()

        self.datasets = {}

        self.datasets['train'] = None
        self.datasets['val'] = None
        self.datasets['test'] = None

        self.assert_dataset()

    def get_default_augmentation(self):
        transform = [albu.Resize(
            self.width, self.height, cv2.INTER_CUBIC, p=1), albu.HorizontalFlip(p=0.5), ]
        return albu.Compose(transform)

    def assert_dataset(self):
        assert len(self.images) == len(self.labels)
        print('Train Images are good to go')

    def __len__(self):
        return len(self.images)

    def aug_function(self, image, mask, set):

        data = {"image": image, 'mask': mask}

        aug_data = self.augmentations[set.decode()](**data)

        aug_img = aug_data["image"]

        aug_msk = aug_data["mask"][:, :, 0]
        
        aug_img = aug_img.astype(np.float32)

        aug_img = aug_img/255.

        if self.mask_downsample is not None:                                   
            aug_msk = rescale(aug_msk, 1.0/self.mask_downsample, anti_aliasing=False, multichannel=False, preserve_range=True, order=0)

        aug_msk = tf.keras.utils.to_categorical(aug_msk-self.label_shift, num_classes=self.num_classes)
        
        return aug_img, aug_msk

    @tf.function(input_signature=[tf.TensorSpec(None, tf.string), tf.TensorSpec(None, tf.string), tf.TensorSpec(None, tf.string)])
    def process_data(self, image, label, set):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        lbl = tf.image.decode_png(tf.io.read_file(label), channels=1)
        aug_img, aug_msk = tf.numpy_function(func=self.aug_function, inp=[
                                             img, lbl, set], Tout=[tf.float32, tf.float32])

        return aug_img, aug_msk, image

    def set_shapes(self, img, label, path):
        img.set_shape((self.width, self.height, 3))
        # label.set_shape((self.width,self.height,self.num_classes))
        if self.mask_downsample is None:
            label.set_shape((self.width, self.height, self.num_classes))
        else:
            label.set_shape((int(self.width/self.mask_downsample) , int(self.height/self.mask_downsample) , self.num_classes))
        return img, label

    def prepare_dataset(self, dataset, set):
        dataset = dataset.map(partial(self.process_data, set=set),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(
            self.batch_size*self.buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    def build_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.images, self.labels)
        )
        dataset = dataset.shuffle(
            len(self.images), reshuffle_each_iteration=False)
        train_dataset = dataset.take(self.train_val_test_split[0])
        val_dataset = dataset.skip(self.train_val_test_split[0]).take(
            self.train_val_test_split[1])
        test_dataset = dataset.skip(self.train_val_test_split[0]).skip(
            self.train_val_test_split[1])

        train_dataset = self.prepare_dataset(train_dataset, 'train')
        val_dataset = self.prepare_dataset(val_dataset, 'val')
        test_dataset = self.prepare_dataset(test_dataset, 'test')

        self.datasets['train'] = train_dataset
        self.datasets['val'] = val_dataset
        self.datasets['test'] = test_dataset

        return train_dataset, val_dataset, test_dataset

    def show_images(self,  num_images=4, set='train',):

        # extract 1 batch from the dataset
        res = next(self.datasets[set].__iter__())

        image = res[0]
        label = res[1]

        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            # print(label[i])
            visualize(
                image=image[i],
                mask=np.argmax(label[i], axis=-1)*255,
            )
            t = image[i]  # tf.cast(image[i], tf.uint8)
            ax = fig.add_subplot(num_images, 5, i+1, xticks=[], yticks=[])
            #tf.numpy_function(func=ax.imshow, inp=[t], Tout=[])
            ax.imshow(image[i])
            #ax.set_title(f"Label: {label[i]}")

            
    def show_results(self, model, num_images=4, set='test', output=None):

        # extract 1 batch from the dataset
        res = next(self.datasets[set].__iter__())

        images = res[0]
        labels = res[1]
        
        preds = model.predict([images])

        if output is not None:
            preds = preds[output]
        
        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            visualize(
                image=images[i],
                predicted_mask=np.argmax(preds[i], axis=-1)*255,
                reference_mask=np.argmax(labels[i], axis=-1)*255,
            )
