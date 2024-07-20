from common import datasets, encoders
from char_embed_feedforward.char_embed import WrapCharEmbed

text = datasets.shakespeare.raw_text
encoder = encoders.Characters(text)

input_length = 50

(X_test, y_test) = encoder.random_train_data_raw(input_length, 10000)

model = WrapCharEmbed(encoder)

def run(samples, model_file=None):
    if model_file is not None:
        model.load("models/test_model_3.keras")
    # model.evaluate(X_test, y_test)
    # for i in range(samples):
    #    naive_model.generate_text(encoder.random_train_data(input_length, 1)[0][0], 50, input_length=input_length, temperature=0.4)
    print(model.single_step(encoder.random_train_data_raw(input_length, 1)[0][0]))