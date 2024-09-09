import tensorflow as tf
import sklearn.model_selection as sk
from preprocessing import encoders    
from datasets import datasets
from naive_feedforward.naive import Naive 

def train(epochs=10, model_file=None):
    text = datasets.shakespeare.raw_text

    encoder = encoders.Characters(text)

    input_length = 50

    (X, y) = encoder.random_train_data_enc(input_length, 2000000)
    X = encoder.normalise_encoded(X)

    X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=0.2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(input_length, activation='sigmoid'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(encoder.unique_chars+1)
    ])
    
    naive_model = Naive()
    naive_model.create(model, encoder)

    naive_model.train(X_train, y_train, epochs)
    naive_model.evaluate(X_test, y_test)
    if model_file is None:
        naive_model.save()
    else:
        naive_model.save(model_file)