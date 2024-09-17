import tensorflow as tf

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

    def call(self, inputs, states=None, return_state=None, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        x, states = self.recurrent(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
    
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