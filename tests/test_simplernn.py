import unittest
import numpy as np
import keras
from sklearn.model_selection import train_test_split

class TestSimpleFFNModel(unittest.TestCase):
    def __init__(self, method):
        self.text = shakespeare.raw_text
        self.encoder = Characters(self.text)
        super().__init__(method)

    def test_simple_ffn(self):
        vocab_size = 4
        model = SimpleRNNModel(vocab_size, 256, 1024)
        # model.compile(loss=keras.losses.categorical_crossentropy)
        model(np.array([[3,1,0], [1,2,3]]))
        #model.fit(np.array([[[3, 1, 0],[1,2,3]]]), np.array([[0, 0, 1, 0], [0, 1, 0, 0]]), verbose=0)
       # model(np.array([[3,1,0]])) 

    #@unittest.skip
    def test_train_simple_rnn(self):
        print("Getting sequences...")
        input_length = 24
        X, y = self.encoder.new_random_train_data_enc(input_length, 100000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
        model = SimpleRNNModel(self.encoder.vocab_size, 128, 512)

        print("Training...")
        train(model, X_train, y_train, epochs=5, batch_size=32, save_name="test")

        print("Loading...")
        model = load_latest("test")
        print(type(model))

        print("Testing...")
        model.evaluate(X_test, y_test, verbose=2)

        print("Retraining...")
        train(model, X_train, y_train, epochs=5, batch_size=32, save_name="test")

        print("Retesting...")
        model.evaluate(X_test, y_test, verbose=2)

        # Test single character prediction with correct and incorrect input lengths
        """
        c = predict_next(model, range(input_length), temperature=1) 
        print(self.encoder.decode(c))
        c = predict_next(model, range(input_length + 2), input_length=input_length, temperature=1)
        print(self.encoder.decode(c))

        # Test sequence generation
        input_text = "Hello! This is a cool test."
        input_encoded = self.encoder.encode(input_text)
        _, generated = generate_next(model, input_encoded, input_length=input_length, temperature=0.7)
        decoded = self.encoder.decode(generated)
        print(f"{input_text}|{decoded}")"""
    
"""Deal with stupid annoying relative imports 
(thanks to Paolo Rovelli's answer at
https://stackoverflow.com/questions/11536764/how-to-fix-attempted-relative-import-in-non-package-even-with-init-py"""
if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path
        sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        from models.simple_rnn import SimpleRNNModel
        from training.trainer import train
        from datasets.datasets import shakespeare
        from preprocessing.encoders import Characters
        from inference.generator import predict_next, generate_next
        from utils.checkpoint import load_latest
    else:
        from ..models.simple_rnn import SimpleRNNModel
        from ..training.trainer import train
        from ..datasets.datasets import shakespeare
        from ..preprocessing.encoders import Characters
        from ..inference.generator import predict_next, generate_next
        from ..utils.checkpoint import load_latest
    unittest.main()
