import datetime
from matplotlib import pyplot as plt

import tensorflow as tf

import logging

from swiss_army_keras.utils import unfreeze_model
from swiss_army_keras.quantization_utils import Quantizer


class ModelBuilder:

    def __init__(self, width, height, channels):
        self.width = width
        self.height = height
        self.channels = channels

    def build(self, model):

        input_tensor = tf.keras.layers.Input(
            shape=(self.width, self.height, self.channels))

        out = model(input_tensor)
        res = tf.keras.models.Model([input_tensor, ], out)
        res.preprocessing = model.preprocessing
        res.summary()

        return res


class TrainingDriver():

    def __init__(self, model, model_name, optimizer, loss, metrics, train_set, val_set, test_set, epochs, unfreezed_epochs=-1, callbacks=[], quant_batches=1, plot_metrics=True):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.epochs = epochs
        self.unfreezed_epochs = epochs if unfreezed_epochs == -1 else unfreezed_epochs
        self.quant_batches = quant_batches
        self.plot_metrics = plot_metrics

        self.callbacks = []

        self.datestr = datetime.datetime.utcnow().strftime("_%Y-%m-%d_%H-%M-%S")

        self.logsname = 'logs_'+model_name+self.datestr
        self.checkpoint_name = model_name + self.datestr + '.h5'
        self.quantizer_name = model_name + self.datestr

        self.callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=self.logsname)
        )
        self.callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_name,
                                               save_best_only=True)
        )
        self.callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
                restore_best_weights=True,
            )
        )

        for c in callbacks:
            self.callbacks.append(c)

    def run(self):

        self.model.compile(loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=self.metrics)

        self.model.summary()

        model_history = self.model.fit(self.train_set,
                                       epochs=self.epochs,
                                       validation_data=self.val_set,
                                       callbacks=self.callbacks,
                                       )

        if self.unfreezed_epochs > 0:

            logging.warning('Unfreezing Model')

            self.model = unfreeze_model(self.model)

            self.model.compile(loss=self.loss,
                               optimizer=self.optimizer,
                               metrics=self.metrics)

            self.model.summary()

            model_history = self.model.fit(self.train_set,
                                           epochs=self.unfreezed_epochs,
                                           validation_data=self.val_set,
                                           callbacks=self.callbacks,
                                           )

        if self.plot_metrics:

            try:

                key_names = ['loss', 'acc', 'iou']
                for k in key_names:
                    legend = []
                    for metric in model_history.history:
                        if k.upper() in str(metric).upper():
                            plt.plot(model_history.history[metric])
                            legend.append(metric)

                    plt.title(k)
                    plt.xlabel('epochs')
                    plt.legend(legend, loc='upper left')
                    plt.savefig(self.checkpoint_name+'_'+k+'_.pdf')
                    plt.clf()

            except Exception as e:
                logging.error(e)


        logging.warning('Quantizing Model')

        q = Quantizer(self.test_set,
                      self.checkpoint_name,
                      self.quantizer_name,
                      batches=self.quant_batches,
                      append_datetime=False)

        q.quantize()
