import tensorflow as tf
import numpy as np

class SimpleRNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.recurrent = tf.keras.layers.GRU(rnn_units, 
                                             return_sequences=True, 
                                             return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    @tf.function
    def call(self, inputs, states=None, return_state=None, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x, states = self.recurrent(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
        
    def predict_next(self, inputs, temperature=0, states=None):
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
    def from_config(cls, config):
        vocab_size = config.get('vocab_size')
        embedding_dim = config.get('embedding_dim')
        rnn_units = config.get('rnn_units')
        return cls(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)