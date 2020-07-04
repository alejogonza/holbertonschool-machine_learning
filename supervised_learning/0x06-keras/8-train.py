#!/usr/bin/env python3
"""
Updates the funtion train model to analyze validation data
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    def l_r_decay(epoch):
        """
        a function that performs step decay
        :param epoch:
        :return:
        """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if early_stopping and validation_data:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                   monitor="val_loss"))
    if learning_rate_decay and validation_data:
        callbacks.append(K.callbacks.LearningRateScheduler(l_r_decay,
                                                           verbose=1))
    if save_best and validation_data:
        callbacks.append(K.callbacks.ModelCheckpoint(filepath=filepath,
                                                     save_best_only=True))
    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, validation_data=validation_data,
                          shuffle=shuffle, callbacks=callbacks)
    return history