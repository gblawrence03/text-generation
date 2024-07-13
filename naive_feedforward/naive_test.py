from common import datasets, encoders
from naive_feedforward.naive import Naive 

def run(samples, model_file=None):
    text = datasets.shakespeare.raw_text
    encoder = encoders.Characters(text)

    input_length = 50

    (X_test, y_test) = encoder.random_train_data_enc(input_length, 10000)
    X_test = encoder.normalise_encoded(X_test)

    naive_model = Naive()
    naive_model.load(encoder, "models/test_model.keras")
    if model_file is not None:
        naive_model.load(encoder, "models/test_model_2.keras")
    naive_model.evaluate(X_test, y_test)
    for i in range(samples):
        naive_model.generate_text(encoder.random_train_data(input_length, 1)[0][0], 50, input_length=input_length, temperature=0.4)