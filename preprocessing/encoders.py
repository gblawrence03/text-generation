import numpy as np
import tensorflow as tf

class Encoder():
    def __init__(self, data):
        self.data = data

class Characters(Encoder):
    def __init__(self, data):
        super().__init__(data)
        vocab = sorted(set(self.data))
        self.data = tf.strings.unicode_split(self.data, input_encoding='UTF-8', errors="ignore")
        self.vocab_size = len(vocab)
        self.char_to_code = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
        self.code_to_char = tf.keras.layers.StringLookup(
            vocabulary = self.char_to_code.get_vocabulary(), mask_token=None, invert=True)
        self.int_encoded = self.char_to_code(self.data) - 1 # StringLookup indexes start at 0 annoyingly 

    def random_train_data_enc(self, input_length, n):
        ids = self.int_encoded.numpy()
        start_indices = np.random.choice(len(ids) - input_length, n, replace=False)
        sequences = np.array([ids[start:start+input_length + 1] for start in start_indices])
        X = sequences[:,:input_length]
        y = sequences[:,-1]
        return (X, y) 
    
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
            return self.char_to_code(text_encode) - 1
        
    def decode(self, text_decode = None):
        if text_decode is None:
            chars = self.code_to_char(self.int_encoded + 1)
        else:
            text_decode = np.atleast_1d(text_decode)
            chars = self.code_to_char(text_decode + 1)
        return tf.strings.reduce_join(chars, axis=-1).numpy().decode("utf-8")
