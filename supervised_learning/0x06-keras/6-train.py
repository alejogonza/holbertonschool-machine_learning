#!/usr/bin/env python3
"""
Train
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent.
    Args:
        network (keras model): model to train.
        data (np.ndarray): matrix of shape (m, nx) containing the input data.
        labels (np.ndarray): one hot matrix of shape (m, classes) containing
                             the labels of data.
        batch_size (int): size of the batch used for mini-batch gradient
                          descent.
        epochs (int): number of passes through data for mini-batch gradient
                      descent.
        validation_data (tuple): data to validate the model with, if not None.
        early_stopping(bool): indicates whether early stopping should be used.
        patiente (int): the patience used for early stopping.
        verbose (bool): determines if output should be printed during training.
        shuffle (bool): determines whether to shuffle the batches every epoch.
        Normally, it is a good idea to shuffle, but for reproducibility, we
        have chosen to set the default to False.
    Returns:
        The History object generated after training the model.
    """
    callbacks = []
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                   patience=patience))

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       callbacks=callbacks)
