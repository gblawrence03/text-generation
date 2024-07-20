import tensorflow as tf
import numpy as np
from keras import ops
from keras import layers

class WrapCharEmbed():
    def __init__(self, encoder):
        self.encoder = encoder

    def create(self, dense_units):
        self.char_embed = CharEmbed(vocab_size=self.encoder.unique_chars, dense_units=dense_units)
        self.inference_model = tf.keras.Sequential([
                                          self.char_embed, 
                                          tf.keras.layers.Softmax()])
        
    def load(self, filename):
        self.char_embed = CharEmbed()
        self.char_embed.load(self.encoder, filename)
        self.inference_model = tf.keras.Sequential([
                                    self.char_embed, 
                                    tf.keras.layers.Softmax()])
    
    def train(self, X_train, y_train, epochs):
        X_train = self.encoder.char_to_code(X_train)
        y_train = self.encoder.char_to_code(y_train)
        self.char_embed.fit(X_train, y_train, epochs=epochs, verbose=2)
    
    def evaluate(self, X_test, y_test):
        X_test = self.encoder.char_to_code(X_test)
        y_test = self.encoder.char_to_code(y_test)
        test_loss, test_acc = self.char_embed.evaluate(X_test,  y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

    # Predicts a single character. input_sequence is assumed to be encoded
    def single_step(self, input_sequence, temperature=1.0):
        input_sequence = self.encoder.char_to_code(input_sequence)
        print(len(input_sequence))
        predictions = self.inference_model.predict(input_sequence, verbose=0)
        if (temperature != 0):
            predictions = np.log(predictions + 1e-10) / temperature
            exp_predictions = np.exp(predictions)
            predictions = exp_predictions / np.sum(exp_predictions)
            return np.random.choice(len(predictions), p=predictions)
        return np.argmax(predictions)

    def save(self, model_file=None):
        self.char_embed.save(filename=model_file)

class CharEmbed(tf.keras.Model):
    def __init__(self, vocab_size=None, dense_units=None):
        super().__init__()
        if vocab_size and dense_units:
            self.create(vocab_size, dense_units)


    def create(self, vocab_size, dense_units):
        self.embedding = EmbedSqueeze(vocab_size + 1)
        self.dense = layers.Dense(dense_units, activation="relu")
        self.dense_output = layers.Dense(vocab_size + 1, activation="relu")
        self.model = tf.keras.Sequential([self.embedding, self.dense, self.dense_output])
        self.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.inf_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        
    def call(self, inputs, training=False): 
        x = inputs
        x = self.embedding(x)#x = tf.keras.ops.squeeze(self.embedding(inputs, training=training))
        
        x = self.dense(x, training=training)
        x = self.dense_output(x, training=training)
        return x
    
    def build(self):
        self.model.build()

    def generate_text(self, input_sequence, output_length, input_length = None, temperature=1.0):
        model = tf.keras.Sequential([self.model, layers.Softmax()])
        if not input_length:
            input_length = len(input_sequence)
        for i in range(output_length):
            input_text = np.array([self.encoder.normalise_encoded(input_sequence[-input_length:])])
            predictions = model.predict(input_text, verbose=0)[0]
            if (temperature != 0):
                predictions = np.log(predictions + 1e-10) / temperature
                exp_predictions = np.exp(predictions)
                predictions = exp_predictions / np.sum(exp_predictions)
                c = np.random.choice(len(predictions), p=predictions)
            else:
                c = np.argmax(predictions)
            input_sequence = np.append(input_sequence, c)

        new_text = self.encoder.decode(input_sequence)
        print(new_text[:input_length] + '|' + new_text[output_length:])

    def save(self, filename=None):
        if filename is None: 
            self.model.save("models/charembedmodel.keras")
        else:
            self.model.save(filename)

    def load(self, encoder, filename=None):
        if filename is None:
            self.model = tf.keras.models.load_model("models/naivemodel.keras")
        else:
            self.model = tf.keras.models.load_model(filename)
        self.embedding = self.model.get_layer(index=0)
        self.dense = self.model.get_layer(index=1)
        self.dense_output = self.model.get_layer(index=2)
        self.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.encoder = encoder

class EmbedSqueeze(layers.Layer):
    def __init__(self, dim=32, trainable=False, dtype='float32'):
        super().__init__()
        self.embedding = layers.Embedding(dim, 1)
        self.reshape = layers.Reshape((-1,))  # Reshape to (-1,) removes any dimension of size 1
    
    def build(self, input_shape):
        # Call build on the embedding layer to ensure its weights are created
        self.embedding.build(input_shape)

    def call(self, inputs):
        x = self.embedding(inputs)
        return self.reshape(x)