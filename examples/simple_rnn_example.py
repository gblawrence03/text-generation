"""Example usage of SimpleRNN
"""

import sys
from os import path
import tensorflow as tf

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from models.simple_rnn import SimpleRNNModel
from training.trainer import train
from datasets.datasets import shakespeare
from preprocessing.encoders import Characters
from utils.checkpoint import load_latest

def main():
    model_name = "example-rnn"

    # Grab the Shakespeare text dataset
    dataset = shakespeare.raw_text

    # Create the character encoder for the dataset
    encoder = Characters(dataset)

    batch_size = 64

    # Using TF dataset
    
    dataset = encoder.tf_shifted_sequence_training_data(200, batch_size=batch_size)
    dataset_size = int(tf.data.experimental.cardinality(dataset)) # Number of batches
    train_size = int(0.8 * dataset_size)
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)

    '''
    # Using numpy
    n_sequences = 50000
    dataset_size = n_sequences // batch_size
    X, y = encoder.shifted_sequence_training_data(100, n_sequences)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_split=0.2)
    '''

    try:
        print("Looking for existing model...")
        model = load_latest(model_name)
    except FileNotFoundError:
        # Create one if not found
        print("Model not found, creating new one...")
        model = SimpleRNNModel(encoder.vocab_size, 256, 1024)

    # Train the model on the training data and save checkpoints
    train(model, train_dataset, epochs=1, batch_size=batch_size, save_name=model_name, save_freq=1, progress_bar="tqdm")
    #train(model, X_train, y_train, epochs=10, batch_size=batch_size, save_name=model_name)

    print("Evaluating!")
    model.evaluate(test_dataset, verbose=1)
    #model.evaluate(X_test, y_test, verbose=1)

    # Encode an input string for inference testing
    input_text = "ROMEO:"
    input_encoded = encoder.encode(input_text)

    # Generate and decode output from the model
    _, generated = model.generate_next(input_encoded, temperature=0.7, generation_length=1000)
    decoded = encoder.decode(generated)

    # Print and format output, separating the input from the generated output
    print(f"{input_text}|{decoded}")

if __name__ == "__main__":
    main()
