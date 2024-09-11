"""Utilities for saving and loading model checkpoints.
"""

import keras
import os
import re

def load_latest(save_name):
    """Gets the latest model checkpoint for the given save name.

    :param save_name: The folder within checkpoints/ to search.
    :type save_name: str
    :raises FileNotFoundError: If the directory cannot be found or the latest save cannot be determined.
    :return: The latest saved model checkpoint
    :rtype: Keras.Model
    """
    checkpoint_dir = os.path.normpath(f"checkpoints/{save_name}/")
    latest = latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"The model save in folder '{checkpoint_dir}' could not be found.")
    return keras.models.load_model(latest)

def latest_checkpoint(checkpoint_dir):
    """Searches through the given directory for the latest checkpoint.
    These checkpoints are assumed to be saved using utils/trainer. 
    The checkpoint filenames are used to determine the latest save, so if these filenames
    are changed, this function may not work as intended.

    :param checkpoint_dir: The directory to search
    :type checkpoint_dir: str
    :return: The filepath for the latest save, or None if it could not be determined.
    :rtype: str | None 
    """
    list_files = os.listdir(checkpoint_dir)
    valid = filter(lambda f: f.endswith(".keras"), list_files)

    found_numbers = {}
    # The final characters in the filename before the extension are the numbers
    pattern = re.compile("\d+$")
    for f in valid:
        without_ext = f.split('.')[0]
        matches = pattern.findall(without_ext)
        if matches:
            # Remove leading zeros, convert to int, find highest match
            num = max(matches, key=lambda x: int(x.lstrip('0'))) 
            found_numbers[f] = num

    if found_numbers:
        latest = max(found_numbers, key=lambda f: found_numbers[f])
        return os.path.join(checkpoint_dir, latest)
    return None

class ResumeModelCheckpoint(keras.callbacks.ModelCheckpoint):
    '''
    This subclass allows us to specify a `initial_epoch` for the ModelCheckpoint callback
    so that the `epoch` parameter that's passed to the checkpoint path string format
    can take into account a previous training session. 
    '''
    def __init__(self,
        filepath,
        monitor="val_loss",
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        save_freq="epoch",
        initial_epoch=0,
        initial_value_threshold=None
    ):
        self.initial_epoch = initial_epoch
        super().__init__(filepath, 
                         monitor=monitor, 
                         verbose=verbose, 
                         save_best_only=save_best_only, 
                         save_weights_only=save_weights_only, 
                         mode=mode, save_freq=save_freq, 
                         initial_value_threshold=initial_value_threshold)

    def _save_model(self, epoch, batch, logs):
        return super()._save_model(self.initial_epoch + epoch, batch, logs)