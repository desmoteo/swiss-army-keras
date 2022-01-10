import os
import cv2
from functools import partial

import albumentations as albu

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import rescale, resize, downscale_local_mean

from tqdm import tqdm

from multiprocessing import Process, cpu_count

from IronDomo import IDPBroker, IDPAsyncClient, IDPWorker

import logging

import dill

import threading

import time

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


class BrokerProcess(Process):
    class Autorizer(object):
        db = None

        def __init__(self):
            db = None

        def callback(self, domain, key):
            logging.warning('Autorizing: {0}, {1}'.format(domain, key))
            return True

    def __init__(self, clear_url='tcp://127.0.0.1:6555', curve_url='tcp://127.0.0.1:6556'):
        super(BrokerProcess, self).__init__()
        self.clear_url = clear_url
        self.curve_url = curve_url

    def run(self):
        autorizer = BrokerProcess.Autorizer()
        server_public = b"P+S690P{iVPfx<aFJwxfSY^ugFzjuWOnaIh!o7J<"
        server_secret = b":$80ST.hxA5xL7c+@$3YTEohOR^GhrJ2$qzN@bR^"
        self.broker = IDPBroker.IronDomoBroker(self.clear_url, self.curve_url, publisher_connection_string=None, verbose=False, credentials=(
            server_public, server_secret), credentialsCallback=autorizer)
        self.broker.bind()
        self.broker.mediate()


class WorkerProcess(Process):
    class Workload(object):
        pre = None

        def __init__(self, count, augmentations_serialized, width, height, output_width, output_height):
            self.count = count
            self.augmentations = {}
            self.augmentations['train'] = dill.loads(
                augmentations_serialized['train'])
            self.augmentations['val'] = dill.loads(
                augmentations_serialized['val'])
            self.augmentations['test'] = dill.loads(
                augmentations_serialized['test'])
            self.width = width
            self.height = height
            self.output_width = output_width
            self.output_height = output_height

        def do(self, request):

            set = request[0].decode()
            img = np.frombuffer(request[1], dtype=np.uint8).reshape(
                (self.width, self.height, 3))
            lbl = np.frombuffer(request[2], dtype=np.uint8).reshape(
                (self.output_width, self.output_height))

            data = {'image': img, 'mask': lbl}

            aug_data = self.augmentations[set](**data)
            return [aug_data['image'].data, aug_data['mask'].data]

    def __init__(self, count, augmentations_serialized, width, height, output_width, output_height, clear_url='tcp://localhost:6555', service='augmentation'):
        self.count = count
        self.augmentations_serialized = augmentations_serialized
        self.width = width
        self.height = height
        self.output_width = output_width
        self.output_height = output_height
        self.clear_url = clear_url
        self.service = service
        super(WorkerProcess, self).__init__()

    def run(self):
        import random as ra
        import numpy as np
        seed = (os.getpid() * int(time.time())) % 123456789
        ra.seed(seed)
        np.random.seed(seed)

        workload = WorkerProcess.Workload(
            self.count, self.augmentations_serialized, self.width, self.height, self.output_width, self.output_height)
        self.worker = IDPWorker.IronDomoWorker(
            self.clear_url, self.service.encode(), False, workload=workload)
        self.worker.loop()


class SegmentationAlbumentationsDataLoader:

    def __init__(self, dataset_path, precache=False, train_augmentations=None, val_augmentations=None, test_augmentations=None, images_dir='images', masks_dir='annotations', width=512, height=512, batch_size=16, num_classes=2, mask_downsample=1, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, label_shift=0, normalization_range=(0, 1), dinamic_range=255):

        self.ids = os.listdir(os.path.join(dataset_path, images_dir))
        self.num_classes = num_classes
        # self.mask_ids = os.listdir(masks_dir)
        images_frames = [os.path.join(dataset_path, images_dir, image_id)
                         for image_id in self.ids]
        labels_frames = [os.path.join(dataset_path, masks_dir, image_id.split(
            image_id.split('.')[-1])[0]+'png') for image_id in self.ids]

        self.precache = precache

        self.configs = {}
        self.images = images_frames
        self.labels = labels_frames
        self.width = width
        self.height = height
        self.output_width = int(width/mask_downsample)
        self.output_height = int(height/mask_downsample)

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

        self.augmentations_serialized = {}
        self.augmentations_serialized['train'] = dill.dumps(
            self.augmentations['train'])
        self.augmentations_serialized['val'] = dill.dumps(
            self.augmentations['val'])
        self.augmentations_serialized['test'] = dill.dumps(
            self.augmentations['test'])

        self.datasets = {}

        self.datasets['train'] = None
        self.datasets['val'] = None
        self.datasets['test'] = None

        self.normalization_range = normalization_range
        self.dinamic_range = dinamic_range

        self.assert_dataset()

        print('a')
        self.broker = BrokerProcess()
        print('b')
        self.broker.daemon = True
        self.broker.start()
        print('c')

        workers = []
        # for i in range(4):
        for i in range(int(cpu_count()/2)):
            p = WorkerProcess(i, self.augmentations_serialized, self.width,
                              self.height, self.output_width, self.output_height)
            p.daemon = True
            p.start()

            workers.append(p)

        self.client = IDPAsyncClient.IronDomoAsyncClient(
            "tcp://localhost:6555", False, identity="DatasetLoader")

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_default_augmentation(self):
        # transform = [albu.Resize(
        #    self.width, self.height, cv2.INTER_CUBIC, p=1), albu.HorizontalFlip(p=0.5), ]
        #transform = albu.Compose(transform)
        transform = None
        return transform

    def assert_dataset(self):
        assert len(self.images) == len(self.labels)
        print('Train Images are good to go')

    def __len__(self):
        return len(self.images)

    def aug_function(self, image, mask, set):

        data = {"image": image, 'mask': mask}

        aug_data = self.augmentations[set.decode()](**data)

        aug_img = aug_data["image"]

        aug_msk = aug_data["mask"]  # [:, :, 0]

        return aug_img, aug_msk.astype(np.uint8)

    def aug_function_parallel(self, images, masks, set):

        for i in range(images.shape[0]):

            self.client.send(b'augmentation', [
                             set, images[i].data, masks[i].data])

        aug_img = np.empty_like(images)
        aug_msk = np.empty_like(masks)

        for i in range(images.shape[0]):

            aug_data = self.client.recv()

            aug_img[i] = np.frombuffer(aug_data[0], dtype=np.uint8).reshape(
                self.width, self.height, 3)  # aug_data["image"]
            aug_msk[i] = np.frombuffer(aug_data[1], dtype=np.uint8).reshape(
                self.output_width, self.output_height)  # aug_data["image"]

        return aug_img, aug_msk.astype(np.uint8)

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.string)])
    def augment_data(self, image, label, set):

        aug_img, aug_msk = tf.numpy_function(func=self.aug_function_parallel, inp=[
                                             image, label, set], Tout=[tf.uint8, tf.uint8])
        # tf.cast(image, np.uint8), tf.cast(label, np.uint8), set], Tout=[tf.float32, tf.uint8])

        return aug_img, aug_msk

    def open_images(self, image, label):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        lbl = tf.image.decode_png(tf.io.read_file(label), channels=1)

        img = tf.cast(tf.image.resize(
            img, [self.height, self.width]), np.uint8)
        lbl = tf.cast(tf.image.resize(
            lbl, [int(self.height/self.mask_downsample), int(self.width/self.mask_downsample)]), np.uint8)[:, :, 0]

        return img, lbl

    def set_shapes(self, img, label):

        img.set_shape((self.width, self.height, 3))

        label.set_shape((int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample)))

        return img, label

    def normalize(self, img, label):
        img = tf.cast(img, tf.float32)
        img = (self.normalization_range[1] - self.normalization_range[0]
               )*img/self.dinamic_range + self.normalization_range[0]

        img.set_shape((self.batch_size, self.width, self.height, 3))

        label = tf.cast(tf.one_hot(tf.cast(label-self.label_shift, tf.uint8),
                                   self.num_classes), tf.float32)

        label.set_shape((self.batch_size,  int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample), self.num_classes))
        return img, label

    def prepare_dataset(self, dataset, set):

        # Open and resize images
        dataset = dataset.map(
            self.open_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.precache:
            dataset = dataset.cache()
            for _ in tqdm(dataset):
                pass
        # augment data
        dataset = dataset.map(
            self.set_shapes, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(
            self.batch_size*self.buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(
            tf.data.experimental.AUTOTUNE)
        if self.augmentations[set] is not None:
            dataset = dataset.map(partial(self.augment_data, set=set),
                                  num_parallel_calls=1)  # tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
            # tf.numpy_function(func=ax.imshow, inp=[t], Tout=[])
            ax.imshow(image[i])
            # ax.set_title(f"Label: {label[i]}")

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
