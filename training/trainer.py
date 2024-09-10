"""Provides functions for training models.
"""

import math
import tensorflow as tf

def train(model, X, y, epochs=10, batch_size=32, save_name=None, save_freq=5):
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
    """    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    callbacks = []
    if save_name is not None:
        # TODO: Get this to work properly with save_weights_only=True
        n_batches = math.ceil(len(y) / batch_size)
        checkpoint_path = "checkpoints/" + save_name + "/cp-{epoch:04d}.keras"
        # TODO: Continuity of epochs in the save name between training sessions
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=False,
            save_freq=int(save_freq * n_batches)
        )
        callbacks.append(cp_callback)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=callbacks)
