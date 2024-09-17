"""Provides functions for training models.
"""

import csv
import math
import os
import tensorflow as tf
from utils.checkpoint import ResumeModelCheckpoint

def train(model, X, y, epochs=10, batch_size=32, save_name=None, save_freq=5, resume=True):
    """Trains a model using :param:`model.fit()`.

    :param model: Model to be trained.
    :type model: Keras.Model
    :param X: Input data
    :type X: See `Keras.Model.fit()`.
    :param y: Target data
    :type y: See `Keras.Model.fit()`.
    :param epochs: Number of epochs to train, defaults to 10
    :type epochs: int, optional
    :param batch_size: Batch size, defaults to 32
    :type batch_size: int, optional
    :param save_name: Folder to save checkpoints in within /checkpoints. 
        If this is not specified, checkpoints will not be saved. Defaults to None
    :type save_name: str, optional
    :param save_freq: How many epochs between which checkpoints are made., defaults to 5
    :type save_freq: int, optional
    :param resume: Whether to overwrite (false) or resume (true) existing logs and checkpoints, defaults to True
    :type resume: bool, optional
    """    
    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    callbacks = []

    if save_name is not None:
        start_epochs = 0
        log_path = f"checkpoints/{save_name}/log.csv"
        hist_callback = tf.keras.callbacks.CSVLogger(log_path, append=resume)
        if resume:
            # Look for existing log to grab start_epochs from
            try:
                with open(log_path) as f:
                    data = list(csv.DictReader(f))
                    start_epochs = len(list(data))
            except FileNotFoundError:
                # Make the directory and file if it doesn't exist
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                f = open(log_path, 'x')

        # TODO: Get this to work properly with save_weights_only=True
        checkpoint_path = "checkpoints/" + save_name + "/cp-{epoch:04d}.keras"
        n_batches = math.ceil(len(y) / batch_size)
        cp_callback = ResumeModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=False,
            save_freq=int(save_freq * n_batches),
            initial_epoch=start_epochs
        )
        
        callbacks.extend([cp_callback, hist_callback])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)
