"""Provides classes for encoding text.
"""

import numpy as np
import tensorflow as tf

class Encoder():
    def __init__(self, data):
        self.data = data

class Characters(Encoder):
    def __init__(self, data, padding_char=' ', encoding="UTF-8"):
        super().__init__(data)
        self.vocab = sorted(set(self.data))
        # Add placeholder character for vocab
        if padding_char in self.vocab:
            self.vocab.remove(padding_char)
        self.vocab.insert(0, padding_char)
        self.data = tf.strings.unicode_split(self.data, input_encoding=encoding, errors="ignore")
        self.char_to_code = idsFromCharsLayer(list(self.vocab))
        self.code_to_char = charsFromIdsLayer(list(self.vocab))
        self.vocab_size = len(self.get_vocab())
        self.int_encoded = self.char_to_code(self.data)
        self.encoded_dataset = tf.data.Dataset.from_tensor_slices(self.int_encoded)

    def next_char_training_data(self, input_length, n):
        ids = self.int_encoded.numpy()
        start_indices = np.random.choice(len(ids) - input_length, n, replace=False)
        sequences = np.array([ids[start:start+input_length + 1] for start in start_indices])
        X = sequences[:,:input_length]
        y = sequences[:,-1]
        return (X, y) 
    
    def shifted_sequence_training_data(self, sequence_length, n):
        ids = self.int_encoded.numpy()
        start_indices = np.random.choice(len(ids) - sequence_length, n, replace=False)
        sequences = np.array([ids[start:start+sequence_length + 1] for start in start_indices])
        X = sequences[:,:-1]
        y = sequences[:,1:]
        return X, y
    
    # TODO: Dataset for other training data types, remove old methods
    def tf_shifted_sequence_training_data(self, seq_length, n=None, batch_size=64, BUFFER_SIZE=10000):
        ids = self.int_encoded.numpy()
        if n is None:
            sequences = self.encoded_dataset.batch(seq_length+1, drop_remainder=True)
        else:
            start_indices = np.random.choice(len(ids) - seq_length, n, replace=False)
            sequences = np.array([ids[start:start+seq_length + 1] for start in start_indices])
            sequences = tf.data.Dataset.from_tensor_slices(sequences)

        dataset = sequences.map(lambda seq: (seq[:-1], seq[1:]))
        dataset = (dataset
                   .shuffle(BUFFER_SIZE)
                   .batch(batch_size, drop_remainder=True)
                   .prefetch(tf.data.experimental.AUTOTUNE))
        return dataset
    
    def random_train_data_raw(self, input_length, n):
        start_indices = np.random.choice(len(self.data) - input_length, n, replace=False)
        sequences = np.array([self.data[start:start+input_length + 1] for start in start_indices])
        X = sequences[:,:input_length]
        y = sequences[:,-1]
        return (X, y) 
    
    def normalise_encoded(self, text_encode = None):
        if text_encode is None:
            return self.int_encoded / (self.vocab_size - 1)
        else:
            return text_encode / (self.vocab_size - 1)
        
    def encode(self, text_encode = None):
        if text_encode is None:
            return self.int_encoded
        else:
            text_encode = tf.strings.unicode_split(text_encode, input_encoding='UTF-8', errors="ignore")
            return self.char_to_code(text_encode)
        
    def decode(self, text_decode = None):
        if text_decode is None:
            chars = self.code_to_char(self.int_encoded)
        else:
            text_decode = np.atleast_1d(text_decode)
            chars = self.code_to_char(text_decode)
        return tf.strings.reduce_join(chars, axis=-1).numpy().decode("utf-8")
    
    def get_vocab(self):
        return self.char_to_code.get_vocabulary()

class idsFromCharsLayer(tf.keras.layers.StringLookup):
    def __init__(self, vocab):
        super().__init__(vocabulary=list(vocab), mask_token=None)

    def call(self, inputs):
        return super().call(inputs)
    
class charsFromIdsLayer(tf.keras.layers.StringLookup):
    def __init__(self, vocab):
        super().__init__(vocabulary=list(vocab), mask_token=None, invert=True)

    def call(self, inputs):
        return super().call(inputs)
