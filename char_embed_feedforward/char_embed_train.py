import tensorflow as tf
import sklearn.model_selection as sk
from common import datasets, encoders    
from char_embed_feedforward.char_embed import WrapCharEmbed



def train(epochs=10, model_file=None):
    text = datasets.shakespeare.raw_text

    encoder = encoders.Characters(text)

    input_length = 50

    (X, y) = encoder.random_train_data_raw(input_length, 2000000)

    X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=0.2)

    model = WrapCharEmbed(encoder)
    model.create(128)

    model.train(X_train, y_train, epochs)
    model.evaluate(X_test, y_test)
    if model_file is None:
        model.save()
    else:
        model.save(model_file)