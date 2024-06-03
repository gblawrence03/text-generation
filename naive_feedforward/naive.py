import tensorflow as tf
import numpy as np

class Naive:
    def __init__(self):
        return
    
    def create(self, model, encoder):
        self.model = model
        self.input_length = model.layers[0].units
        self.model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.encoder = encoder
    
    def train(self, X_train, y_train, epochs):
        self.model.fit(X_train, y_train, epochs=epochs, verbose=2)

    def evaluate(self, X_test, y_test):
        test_loss, test_acc = self.model.evaluate(X_test,  y_test, verbose=2)
        print('\nTest accuracy:', test_acc)

    def generate_text(self, input_sequence, output_length, input_length = None, temperature=1.0):
        model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
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
            self.model.save("models/naivemodel.keras")
        else:
            self.model.save(filename)

    def load(self, encoder, filename=None):
        if filename is None:
            self.model = tf.keras.models.load_model("models/naivemodel.keras")
        else:
            self.model = tf.keras.models.load_model(filename)
        self.input_length = self.model.layers[0].units
        self.model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        self.encoder = encoder