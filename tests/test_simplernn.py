import unittest
import numpy as np

class TestSimpleFFNModel(unittest.TestCase):
    def __init__(self, method):
        self.text = shakespeare.raw_text
        self.encoder = Characters(self.text)
        super().__init__(method)

    @unittest.skip
    def test_simple_rnn(self):
        vocab_size = 4
        model = SimpleRNNModel(vocab_size, 256, 1024)
        model(np.array([[3,1,0], [1,2,3]]))

    @unittest.skip
    def test_train_simple_rnn(self):
        print("Getting sequences...")
        input_length = 100
        batch_size = 64
        dataset = self.encoder.tf_shifted_sequence_training_data(input_length, 100000, batch_size=batch_size)
        dataset_size = 100000 // batch_size
        train_size = int(0.8 * dataset_size)
        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)
        model = SimpleRNNModel(self.encoder.vocab_size, 256, 512)

        print("Training...")
        train(model, train_dataset, epochs=2, save_name="test-simplernn", save_freq=2)

        print("Loading...")
        model = load_latest("test")
        print(type(model))

        print("Testing...")
        model.evaluate(test_dataset, verbose=2)

        print("Retraining...")
        train(model, train_dataset, epochs=2, save_name="test-simplernn", save_freq=2)

        print("Retesting...")
        model.evaluate(test_dataset, verbose=2)

        print("Testing single character generation...")
        c, _ = model.predict_next(self.encoder.encode("ROMEO: "), temperature=1) 
        print(self.encoder.decode(c))

        print("Testing sequence generation...")
        input_text = "ROMEO: "
        input_encoded = self.encoder.encode(input_text)
        _, generated = model.generate_next(input_encoded, temperature=0.7, generation_length=200)
        decoded = self.encoder.decode(generated)
        print(f"{input_text}|{decoded}")

    #@unittest.skip
    def test_inference_simple_rnn(self):
        model = load_latest("test-simplernn")
        input_length = 100
        
        dataset = self.encoder.tf_shifted_sequence_training_data(input_length, 5000)
        model.evaluate(dataset, verbose=2)

        input_encoded = self.encoder.encode("Hello! ")
        c, _ = model.predict_next(input_encoded, temperature=0) 
        print(self.encoder.decode(c))

        # Test sequence generation
        input_text = "Thou"
        input_encoded = self.encoder.encode(input_text)
        _, generated = model.generate_next(input_encoded, temperature=1, generation_length=200)
        decoded = self.encoder.decode(generated)
        print(f"{input_text}|{decoded}")

    
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
        from utils.checkpoint import load_latest
    else:
        from ..models.simple_rnn import SimpleRNNModel
        from ..training.trainer import train
        from ..datasets.datasets import shakespeare
        from ..preprocessing.encoders import Characters
        from ..utils.checkpoint import load_latest
    unittest.main()
