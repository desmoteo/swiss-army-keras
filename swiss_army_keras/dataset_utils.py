from codecs import ignore_errors
from swiss_army_keras import __path__ as mypath
from PIL import ImageFont
from PIL import ImageDraw
import pathlib
import os
from base64 import b64encode
import cv2
from functools import partial

import albumentations as albu

import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from skimage.transform import rescale, resize, downscale_local_mean
from skimage.io import imread

from tqdm import tqdm

from multiprocessing import Process, cpu_count

from IronDomo import IDPBroker, IDPAsyncClient, IDPWorker

import logging

import dill

import threading

from PIL import Image

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

    def __init__(self, clear_url='tcp://127.0.0.1:4445', curve_url='tcp://127.0.0.1:4446'):
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

        def __init__(self, count, augmentations_serialized, width, height, output_width, output_height, mode='segmentation'):
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

            self.mode = mode

        def do(self, request):

            dset = request[0].decode()
            img = np.frombuffer(request[1], dtype=np.uint8).reshape(
                (self.width, self.height, 3))

            if self.mode == 'segmentation':

                lbl = np.frombuffer(request[2], dtype=np.uint8).reshape(
                    (self.output_width, self.output_height))
                try:
                    data = {'image': img, 'mask': lbl}

                    aug_data = self.augmentations[dset](**data)
                    return [aug_data['image'].data, aug_data['mask'].data]
                except Exception as e:
                    logging.error(f'Augmentation error: {e}')
                    return [img.data, lbl.data]
            elif self.mode == 'classification':

                data = {'image': img}
                label = request[2]

                try:
                    aug_data = self.augmentations[dset](**data)
                    return [aug_data['image'].data, label]
                except Exception as e:
                    logging.error(f'Augmentation error: {e}')
                    return [img.data, label]

    def __init__(self, count, augmentations_serialized, width, height, output_width, output_height, clear_url='tcp://127.0.0.1:4445', service='augmentation', mode='segmentation'):
        self.count = count
        self.augmentations_serialized = augmentations_serialized
        self.width = width
        self.height = height
        self.output_width = output_width
        self.output_height = output_height
        self.clear_url = clear_url
        self.service = service
        self.mode = mode
        super(WorkerProcess, self).__init__()

    def run(self):
        import random as ra
        import numpy as np
        seed = (os.getpid() * int(time.time())) % 123456789
        ra.seed(seed)
        np.random.seed(seed)

        workload = WorkerProcess.Workload(
            self.count, self.augmentations_serialized, self.width, self.height, self.output_width, self.output_height, mode=self.mode)
        self.worker = IDPWorker.IronDomoWorker(
            self.clear_url, self.service.encode(), False, workload=workload, identity=f'{self.service}_{self.count}')
        self.worker.loop()


class SegmentationAlbumentationsDataLoader:

    def __init__(self, dataset_path, precache=False, train_augmentations=None, val_augmentations=None, test_augmentations=None, images_dir='images', masks_dir='annotations', resize=True, width=512, height=512, batch_size=16, num_classes=2, mask_downsample=1, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, label_shift=0, normalization=(0, 1), dinamic_range=255, ignore_errors=False):

        self.ids = os.listdir(os.path.join(dataset_path, images_dir))
        self.num_classes = num_classes
        # self.mask_ids = os.listdir(masks_dir)
        images_frames = [os.path.join(dataset_path, images_dir, image_id)
                         for image_id in self.ids]
        labels_frames = [os.path.join(dataset_path, masks_dir, image_id.split(
            image_id.split('.')[-1])[0]+'png') for image_id in self.ids]

        self.precache = precache

        self.resize = resize

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

        self.normalization = normalization
        self.dinamic_range = dinamic_range

        if (isinstance(self.normalization, tuple)):
            logging.warn(f'Normalizing in range: {self.normalization}')
            self.normalized_dynamic_range = self.normalization
        else:
            mr = self.normalization(
                np.array([[[[0, self.dinamic_range]]]])).flatten()
            logging.warn(f'Normalizing in range: {mr}')
            self.normalized_dynamic_range = (mr[0], mr[1])

        self.assert_dataset()

        randpart = b64encode(os.urandom(5), altchars=b'01').decode()

        self.socket_clear = f'ipc:///tmp/swiss_army_keras_segmentation_clear_socket_{randpart}'
        self.socket_curve = f'ipc:///tmp/swiss_army_keras_segmentation_curve_socket_{randpart}'

        self.broker = BrokerProcess(
            clear_url=self.socket_clear, curve_url=self.socket_curve)
        self.broker.daemon = True
        self.broker.start()

        workers = []
        # for i in range(4):
        for i in range(int(cpu_count()/2)):
            p = WorkerProcess(i, self.augmentations_serialized, self.width,
                              self.height, self.output_width, self.output_height,
                              clear_url=self.socket_clear)
            p.daemon = True
            p.start()

            workers.append(p)

        self.client = IDPAsyncClient.IronDomoAsyncClient(
            self.socket_clear, False, identity="DatasetLoader")

        self.ignore_errors = ignore_errors

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_class_weights(self):
        h = None
        for f in tqdm(self.labels):
            m = Image.open(f).resize(
                (self.width, self.height), resample=Image.NEAREST)
            if h is None:
                h, _ = np.histogram(m, self.num_classes, (0, self.num_classes))
            else:
                hist, _ = np.histogram(
                    m, self.num_classes, (0, self.num_classes))
                h = hist + h
        h = 1/h
        h = h/sum(h)
        return h

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

    def aug_function(self, image, mask, dset):

        data = {"image": image, 'mask': mask}

        aug_data = self.augmentations[dset.decode()](**data)

        aug_img = aug_data["image"]

        aug_msk = aug_data["mask"]  # [:, :, 0]

        return aug_img, aug_msk.astype(np.uint8)

    def aug_function_parallel(self, images, masks, dset):

        for i in range(images.shape[0]):

            

            self.client.send(b'augmentation', [
                             dset,  images[i].data, masks[i].data])

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
    def augment_data(self, image, label, dset):

        aug_img, aug_msk = tf.numpy_function(func=self.aug_function_parallel, inp=[
                                             image, label, dset], Tout=[tf.uint8, tf.uint8])
        # tf.cast(image, np.uint8), tf.cast(label, np.uint8), dset], Tout=[tf.float32, tf.uint8])

        return aug_img, aug_msk

    def open_images(self, image, label):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        lbl = tf.image.decode_png(tf.io.read_file(label), channels=1)
        
        def fmaker(m):
            def f(): 
                # tf.print(f'Resizing: {m}')
                return tf.cast(tf.image.resize(
                    img, [self.height, self.width], method=m), np.uint8)
            return f


        img = tf.switch_case(
            tf.random.uniform(shape=[], dtype=tf.int32, maxval=4), 
            {
                0: fmaker('nearest'),
                1: fmaker('bilinear'),
                2: fmaker('bicubic'),
                3: fmaker('area'),
            }, 
            default=None, 
            name='switch_case'
        )


        lbl = tf.cast(tf.image.resize(
            lbl, [int(self.height/self.mask_downsample), int(self.width/self.mask_downsample)], antialias=False, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR), np.uint8)[:, :, 0]

        return img, lbl

    def set_shapes(self, img, label):

        img.set_shape((self.width, self.height, 3))

        label.set_shape((int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample)))

        return img, label

    def normalize(self, img, label):
        img = tf.cast(img, tf.float32)

        if (isinstance(self.normalization, tuple)):
            img = (self.normalization[1] - self.normalization[0]
                   )*img/self.dinamic_range + self.normalization[0]
        else:
            img = self.normalization(img)

        img.set_shape((self.batch_size, self.width, self.height, 3))

        label = tf.cast(tf.one_hot(tf.cast(label-self.label_shift, tf.uint8),
                                   self.num_classes), tf.float32)

        label.set_shape((self.batch_size,  int(self.width/self.mask_downsample),
                        int(self.height/self.mask_downsample), self.num_classes))
        return img, label

    def prepare_dataset(self, dataset, dset):

        # Open and resize images
        dataset = dataset.map(
            self.open_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if(self.ignore_errors):
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
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
        if self.augmentations[dset] is not None:
            dataset = dataset.map(partial(self.augment_data, dset=dset),
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

    def show_images(self,  num_images=4, dset='train',):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = (res[0] - self.normalized_dynamic_range[0]) * \
            255.0 / \
            (self.normalized_dynamic_range[1] -
             self.normalized_dynamic_range[0])

        label = res[1]

        for i in range(num_images):

            visualize(
                image=tf.cast(images[i], tf.uint8),
                mask=np.argmax(label[i], axis=-1)*255,
            )

    def show_results(self, model, num_images=4, dset='test', output=None):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = res[0]
        labels = res[1]

        preds = model.predict([images])

        if output is not None:
            preds = preds[output]

        for i in range(num_images):
            image = (images[i] - self.normalized_dynamic_range[0]) * \
                255.0 / \
                (self.normalized_dynamic_range[1] -
                 self.normalized_dynamic_range[0])
            visualize(
                image=tf.cast(image, tf.uint8),
                predicted_mask=np.argmax(preds[i], axis=-1)*255,
                reference_mask=np.argmax(labels[i], axis=-1)*255,
            )


class ClassificationAlbumentationsDataLoader:

    def __init__(self, dataset_path, precache=False, train_augmentations=None, val_augmentations=None, test_augmentations=None, resize=True, width=512, height=512, batch_size=16, train_val_test_split=[0.8, 0.1, 0.1], buffer_size=4, normalization=(0, 1), dinamic_range=255, ignore_errors=False):

        self.data_root = pathlib.Path(dataset_path)

        self.path_parts = len(dataset_path.split(os.sep))

        self.resize = resize

        self.subfolders = [f.path.split(
            '/')[-1] for f in os.scandir(str(self.data_root)) if f.is_dir()]
        self.subfolders.sort()

        filelist = []
        for sf in self.subfolders:
            for dirpath, dnames, fnames in os.walk(f'{dataset_path}/{sf}'):
                for f in fnames:
                    filelist.append(os.path.join(dirpath, f))

        self.files_ds = tf.data.Dataset.from_tensor_slices(filelist)

        self.classes_map = {}
        self.reverse_classes_map = {}

        cnt = 0
        for f in self.subfolders:
            self.classes_map[f] = cnt
            self.reverse_classes_map[cnt] = f
            print(f'Class label {f} will have id: {cnt}')
            cnt = cnt + 1

        self.num_classes = len(self.classes_map.keys())

        # build a lookup table
        self.lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self.classes_map.keys()),
                values=list(self.classes_map.values()),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )

        # build a lookup table
        self.reverse_lookup_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(self.classes_map.values()),
                values=list(self.classes_map.keys()),
            ),
            default_value=tf.constant("unknown"),
            name="class_weight"
        )

        self.precache = precache

        self.configs = {}

        self.width = width
        self.height = height

        self.buffer_size = buffer_size

        self.train_val_test_split = []

        for val in train_val_test_split:
            self.train_val_test_split.append(int(len(self.files_ds)*val))

        print(f'train_val_test_split: {self.train_val_test_split}')

        self.batch_size = batch_size

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

        self.normalization = normalization
        self.dinamic_range = dinamic_range

        if (isinstance(self.normalization, tuple)):
            logging.warn(f'Normalizing in range: {self.normalization}')
            self.normalized_dynamic_range = self.normalization
        else:
            mr = self.normalization(
                np.array([[[[0, self.dinamic_range]]]])).flatten()
            logging.warn(f'Normalizing in range: {mr}')
            self.normalized_dynamic_range = (mr[0], mr[1])

        randpart = b64encode(os.urandom(5), altchars=b'01').decode()

        self.socket_clear = f'ipc:///tmp/swiss_army_keras_classification_clear_socket_{randpart}'
        self.socket_curve = f'ipc:///tmp/swiss_army_keras_classification_curve_socket_{randpart}'

        self.broker = BrokerProcess(
            clear_url=self.socket_clear, curve_url=self.socket_curve)
        self.broker.daemon = True
        self.broker.start()

        workers = []
        # for i in range(4):
        for i in range(int(cpu_count()/2)):
            p = WorkerProcess(i, self.augmentations_serialized,
                              self.width, self.height,
                              0, 0,
                              clear_url=self.socket_clear,
                              mode='classification')
            p.daemon = True
            p.start()

            workers.append(p)

        self.client = IDPAsyncClient.IronDomoAsyncClient(
            self.socket_clear, False, identity="DatasetLoader")

        self.ignore_errors = ignore_errors

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def get_class_weights(self):
        h = None
        for f in tqdm(self.labels):
            m = Image.open(f).resize(
                (self.width, self.height), resample=Image.NEAREST)
            if h is None:
                h, _ = np.histogram(m, self.num_classes, (0, self.num_classes))
            else:
                hist, _ = np.histogram(
                    m, self.num_classes, (0, self.num_classes))
                h = hist + h
        h = 1/h
        h = h/sum(h)
        return h

    def get_default_augmentation(self):
        # transform = [albu.Resize(
        #    self.width, self.height, cv2.INTER_CUBIC, p=1), albu.HorizontalFlip(p=0.5), ]
        #transform = albu.Compose(transform)
        transform = None
        return transform

    def __len__(self):
        return len(self.files_ds)

    def aug_function_parallel(self, images, labels, dset):

        for i in range(images.shape[0]):
            self.client.send(b'augmentation', [
                             dset, images[i].data, labels[i].data])

        aug_img = np.empty_like(images)
        aug_label = np.empty_like(labels)

        for i in range(images.shape[0]):

            aug_data = self.client.recv()

            aug_img[i] = np.frombuffer(aug_data[0], dtype=np.uint8).reshape(
                self.width, self.height, 3)  # aug_data["image"]
            aug_label[i] = np.frombuffer(aug_data[1], dtype=np.int32)

        return aug_img, aug_label

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint8), tf.TensorSpec(None, tf.int32), tf.TensorSpec(None, tf.string)])
    def augment_data(self, image, label, dset):

        aug_img, aug_label = tf.numpy_function(func=self.aug_function_parallel, inp=[
            image, label, dset], Tout=[tf.uint8, tf.int32])

        return aug_img, aug_label

    def open_images(self, image, label):
        img = tf.image.decode_jpeg(tf.io.read_file(image), channels=3)

        def fmaker(m):
            def f(): 
                # tf.print(f'Resizing: {m}')
                return tf.cast(tf.image.resize(
                    img, [self.height, self.width], method=m), np.uint8)
            return f


        img = tf.switch_case(
            tf.random.uniform(shape=[], dtype=tf.int32, maxval=4), 
            {
                0: fmaker('nearest'),
                1: fmaker('bilinear'),
                2: fmaker('bicubic'),
                3: fmaker('area'),
            }, 
            default=None, 
            name='switch_case'
        )

        return img, label

    def set_shapes(self, img, label):

        img.set_shape((self.width, self.height, 3))

        return img, label

    def normalize(self, img, label):
        img = tf.cast(img, tf.float32)

        if (isinstance(self.normalization, tuple)):
            img = (self.normalization[1] - self.normalization[0]
                   )*img/self.dinamic_range + self.normalization[0]
        else:
            img = self.normalization(img)

        img.set_shape((self.batch_size, self.width, self.height, 3))

        label = tf.cast(tf.one_hot(tf.cast(label, tf.uint8),
                                   self.num_classes), tf.float32)

        label.set_shape((self.batch_size, self.num_classes))

        return img, label

    def prepare_dataset(self, dataset, dset):

        # Open and resize images
        dataset = dataset.map(
            self.open_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if(self.ignore_errors):
            dataset = dataset.apply(tf.data.experimental.ignore_errors())
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
        if self.augmentations[dset] is not None:
            dataset = dataset.map(partial(self.augment_data, dset=dset),
                                  num_parallel_calls=1)  # tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_datasets(self):

        dataset = self.files_ds

        def process_path(file_path):
            intermedio = tf.strings.split(file_path, os.sep)[self.path_parts]
            #lbl = tf.cast(self.classes_map[intermedio], tf.float32)
            lbl = tf.cast(self.lookup_table.lookup(intermedio), tf.int32)
            #lbl = tf.strings.split(file_path, os.sep)[-2]
            return file_path, lbl

        dataset = dataset.map(process_path)

        dataset = dataset.shuffle(
            len(self.files_ds), reshuffle_each_iteration=False)
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

    def show_images(self,  num_images=4, dset='train',):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = (res[0] - self.normalized_dynamic_range[0]) * \
            255.0 / \
            (self.normalized_dynamic_range[1] -
             self.normalized_dynamic_range[0])

        for i in range(num_images):
            image = tf.cast(images[i], tf.uint8)
            pimg = Image.fromarray(image.numpy())
            draw = ImageDraw.Draw(pimg)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype(
                f'{mypath}/DejaVuSansMono.ttf', int(self.height/20))
            # draw.text((x, y),"Sample Text",(r,g,b))

            shape = [(0, 0), (int(self.width/2), int(1.25*self.height/20))]
            draw.rectangle(shape, (0, 0, 0))
            #draw.text((0, 0), str(res[1][i].numpy()),(255,255,255), font=font)
            draw.text((0, 0), self.reverse_classes_map[np.argmax(
                res[1][i].numpy(), axis=-1)], (255, 255, 255), font=font)
            # print(label[i])
            visualize(
                image=pimg,
                #mask=np.argmax(label[i], axis=-1)*255,
            )

    def show_results(self, model, num_images=4, dset='test', output=None):

        # extract 1 batch from the dataset
        res = next(self.datasets[dset].__iter__())

        images = res[0]
        labels = res[1]

        preds = model.predict([images])

        if output is not None:
            preds = preds[output]

        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            image = (images[i] - self.normalized_dynamic_range[0]) * \
                255.0 / \
                (self.normalized_dynamic_range[1] -
                 self.normalized_dynamic_range[0])
            image = tf.cast(image, tf.uint8)
            predimg = Image.fromarray(image.numpy())
            refimg = Image.fromarray(image.numpy())
            preddraw = ImageDraw.Draw(predimg)
            refdraw = ImageDraw.Draw(refimg)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = ImageFont.truetype(
                f'{mypath}/DejaVuSansMono.ttf', int(self.height/20))
            # draw.text((x, y),"Sample Text",(r,g,b))

            shape = [(0, 0), (int(self.width/2), int(1.25*self.height/20))]
            preddraw.rectangle(shape, (0, 0, 0))
            refdraw.rectangle(shape, (0, 0, 0))
            #draw.text((0, 0), str(res[1][i].numpy()),(255,255,255), font=font)
            preddraw.text((0, 0), self.reverse_classes_map[np.argmax(
                preds[i], axis=-1)], (255, 255, 255), font=font)
            refdraw.text((0, 0), self.reverse_classes_map[np.argmax(
                labels[i], axis=-1)], (255, 255, 255), font=font)
            # print(label[i])

            visualize(
                predicted=predimg,
                reference=refimg,
            )
