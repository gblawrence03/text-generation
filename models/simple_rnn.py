import tensorflow as tf
from keras import Model, layers
import numpy as np

class SimpleRNNModel(Model):
    """Simple recurrent generation model. 
    Inputs correspond to characters
    in the form of character indices between 0 and :param:`vocab_size - 1`. 

    :param vocab_size: Number of characters in the text vocabulary
    :type vocab_size: int
    :param embedding_dim: Number of dimensions in the embedding layer
    :type embedding_dim: int
    :param rnn_units: Number of recurrent units
    :type rnn_units: int
    """
    def __init__(self, vocab_size, embedding_dim, rnn_units, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.recurrent = layers.LSTM(rnn_units,
                                     return_sequences=True,
                                     return_state=True)
        self.dense = layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=None, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        r = self.recurrent(x, initial_state=states, training=training)
        x, states = r[0], r[1:]
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
        
    def predict_next(self, inputs, temperature=0, states=None):
        """Predict a single character

        :param inputs: Encoded input sequence. Must have the same length as inputs in training data. 
        :type inputs: Array-like 
        :param temperature: Softmax temperature, used to introduce variation, defaults to 0
        :type temperature: int, optional
        :param states: Initial states to apply to the recurrent layer, defaults to None.
        :type states: tensors
        :return: Encoded output character
        :rtype: Int
        """
        inputs = np.array(inputs).reshape(1, -1)

        preds, states = self(inputs, states=states, return_state=True)
        
        preds = preds[:,-1,:]
        
        if temperature != 0:
            preds /= temperature

            predicted_ids = tf.random.categorical(preds, num_samples=1)
            c = tf.squeeze(predicted_ids, axis=-1)
            
        else:
            c = np.argmax(preds)
        return c, states
    
    def generate_next(self, start_inputs, generation_length=50, temperature=0):
        """Predict a sequence of characters

        :param start_inputs: Starting string of characters to be used as input
        :type start_inputs: array-like
        :param generation_length: Number of characters to generate, defaults to 50
        :type generation_length: int, optional
        :param temperature: Softmax temperature, used to introduce variation, defaults to 0
        :type temperature: int, optional
        :return: Tuple containing start inputs and generated characters
        :rtype: Tuple of arrays
        """
        start_inputs = np.array(start_inputs).flatten()

        generated=start_inputs
        c=start_inputs
        states = None
        for _ in range(generation_length):
            c, states = self.predict_next(c, temperature=temperature, states=states)
            generated = np.append(generated, c)

        return start_inputs, generated[-generation_length:]
    
    def get_config(self):
        config = {'vocab_size': self.vocab_size, 'embedding_dim': self.embedding_dim, 'rnn_units': self.rnn_units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        vocab_size = config.get('vocab_size')
        embedding_dim = config.get('embedding_dim')
        rnn_units = config.get('rnn_units')
        return cls(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)