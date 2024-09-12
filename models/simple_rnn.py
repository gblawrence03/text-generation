import tensorflow as tf

class SimpleRNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
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