import tensorflow as tf
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt


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


def standard_totflite(val):
    return val.numpy().astype(np.float32)


def symmetric_totflite(val):
    return val.numpy().astype(np.float32)


def ui8_totflite(val):
    return (val*255).numpy().astype(np.uint8)


totflite_dict = {}
totflite_dict[-1] = ui8_totflite
totflite_dict[0] = standard_totflite
totflite_dict[1] = symmetric_totflite


class Quantizer():
    def __init__(self, dataset, model, name, batches=1):
        self.dataset = dataset
        self.model = model
        self.name = f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_'
        self.batches = batches

        self.tflite_ui8_model = None
        self.tflite_f16_model = None
        self.normalization = 0

    def quantize(self):

        def representative_data_gen():
            # for input_value in tf.data.Dataset.from_tensor_slices(test_input).batch(1).take(1):
            #    yield [tf.cast(input_value, tf.float32) /255.]
            for i in range(self.batches):
                vals = self.dataset.__iter__().next()[0]
                for val in vals:
                    yield [tf.expand_dims(tf.cast(val, tf.float32), axis=0)]

        self.model.trainable = False

        if isinstance(self.model, str):
            converter = tf.lite.TFLiteConverter.from_saved_model(self.model)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        # Ensure that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Set the input and output tensors to uint8 (APIs added in r2.3)
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        self.tflite_ui8_model = converter.convert()

        with open(f'{self.name}ui8.tflite', 'wb') as f:
            f.write(self.tflite_ui8_model)

        if isinstance(self.model, str):
            converter = tf.lite.TFLiteConverter.from_saved_model(self.model)
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Ensure that if any ops can't be quantized, the converter throws an error
        self.tflite_f16_model = converter.convert()

        with open(f'{self.name}f16.tflite', 'wb') as f:
            f.write(self.tflite_f16_model)

    def vizualize_ui8_results(self, num_images):

        self.vizualize_results(num_images, self.tflite_ui8_model, -1)

    def vizualize_f16_results(self, num_images):

        self.vizualize_results(
            num_images, self.tflite_f16_model, self.normalization)

    def vizualize_results(self, num_images, model, normalization):

        interpreter = tf.lite.Interpreter(model_content=model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print(input_details)
        output_details = interpreter.get_output_details()
        print(output_details)

        it = next(self.dataset.__iter__())

        images = it[0]
        labels = it[1]

        fig = plt.figure(figsize=(22, 22))
        for i in range(num_images):
            interpreter.set_tensor(input_details[0]['index'], np.expand_dims(
                totflite_dict[normalization](images[i]), axis=0))
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data, axis=3)[0]
            visualize(
                image=images[i],
                predicted_mask=prediction*255,
                reference_mask=np.argmax(labels[i], axis=-1)*255,
            )
