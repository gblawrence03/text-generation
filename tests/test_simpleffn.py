import unittest
import numpy as np
import keras
from sklearn.model_selection import train_test_split

class TestSimpleFFNModel(unittest.TestCase):
    def __init__(self, method):
        self.text = shakespeare.raw_text
        self.encoder = Characters(self.text)
        super().__init__(method)

    @unittest.skip
    def test_simple_ffn_rescale(self):
        vocab_size = 4
        input_length = 3
        model = SimpleFFNModel(vocab_size, hidden_units=8)
        self.run_rescale_test(model, vocab_size, [1, 2, 3])
        self.run_rescale_test(model, vocab_size, [0, 1, 3])
        self.run_rescale_test(model, vocab_size, [0, 2, 3])

    @unittest.skip
    def test_simple_ffn(self):
        vocab_size = 4
        model = SimpleFFNModel(vocab_size, hidden_units=8)
        model.compile(loss=keras.losses.categorical_crossentropy)
        model(np.array([[3,1,0]]))
        model.fit(np.array([[3, 1, 0],[1,2,3]]), np.array([[0, 0, 1, 0], [0, 1, 0, 0]]), verbose=0)
        model(np.array([[3,1,0]]))

    def test_train_simple_ffn(self):
        print("Getting sequences...")
        input_length = 32
        X, y = self.encoder.next_char_training_data(input_length, 50000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = SimpleFFNModel(self.encoder.vocab_size, hidden_layers_units = [128, 256, 128])

        print("Training...")
        train(model, X_train, y_train, epochs=5, batch_size=32, save_name="test-simpleffn")

        print("Loading...")
        model = load_latest("test-simpleffn")
        print(type(model))

        print("Testing...")
        model.evaluate(X_test, y_test, verbose=2)

        print("Retraining...")
        train(model, X_train, y_train, epochs=5, batch_size=32, save_name="test-simpleffn")

        print("Retesting...")
        model.evaluate(X_test, y_test, verbose=2)

        # Test single character prediction with correct and incorrect input lengths
        c = model.predict_next(range(input_length), temperature=1) 
        print(self.encoder.decode(c))
        c = model.predict_next(range(input_length + 2), input_length=input_length, temperature=0)
        print(self.encoder.decode(c))

        # Test sequence generation
        input_text = "Hello! This is a cool test."
        input_encoded = self.encoder.encode(input_text)
        _, generated = model.generate_next(input_encoded, input_length=input_length, temperature=0.5)
        decoded = self.encoder.decode(generated)
        print(f"{input_text}|{decoded}")
    
    @unittest.skip
    def run_rescale_test(self, model, vocab_size, test_input):
        expected_output = [x / (vocab_size) for x in test_input]
        self.assertListEqual(list(model.rescale(test_input)), expected_output)
    
"""Deal with stupid annoying relative imports 
(thanks to Paolo Rovelli's answer at
https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py"""
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from models.simple_ffn import SimpleFFNModel
        from training.trainer import train
        from datasets.datasets import shakespeare
        from preprocessing.encoders import Characters
        from utils.checkpoint import load_latest
    else:
        from ..models.simple_ffn import SimpleFFNModel
        from ..training.trainer import train
        from ..datasets.datasets import shakespeare
        from ..preprocessing.encoders import Characters
        from ..utils.checkpoint import load_latest
    unittest.main()
