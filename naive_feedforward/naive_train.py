import tensorflow as tf
import sklearn.model_selection as sk
from common import datasets, encoders    
from naive import Naive 

text = datasets.shakespeare.raw_text
encoder = encoders.Characters(text)

input_length = 50

(X, y) = encoder.random_train_data(input_length, 1000000)
X = encoder.normalise_encoded(X)

X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=0.33)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_length, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(encoder.unique_chars)
])

naive_model = Naive()
naive_model.create(model, encoder)

naive_model.train(X_train, y_train, 2)
naive_model.evaluate(X_test, y_test)
naive_model.save("models/test_model.keras")