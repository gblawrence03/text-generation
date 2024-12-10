import sys
from os import path

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from models.simple_rnn import SimpleRNNModel
from training.trainer import train
from datasets.datasets import shakespeare
from preprocessing.encoders import Characters
from utils.checkpoint import load_latest

model_name = "example-rnn"

# Grab the Shakespeare text dataset
dataset = shakespeare.raw_text

# Create the character encoder for the dataset
encoder = Characters(dataset)                    

# Grab some random train data in encoded format and split
n_sequences = 10000
batch_size = 64
dataset = encoder.tf_shifted_sequence_training_data(50, n_sequences, batch_size=batch_size)
dataset_size = n_sequences // batch_size
train_size = int(0.8 * dataset_size)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Try to load existing SimpleFFN model
try:
    print("Looking for existing model...")
    model = load_latest(model_name)
except FileNotFoundError as e:
    # Create one if not found
    print("Model not found, creating new one...")
    model = SimpleRNNModel(encoder.vocab_size, 128, 256)

# Train the model on the training data and save checkpoints
train(model, train_dataset, epochs=10, batch_size=32, save_name=model_name)

# Evaluate the model
model.evaluate(test_dataset, verbose=2)

# Encode an input string for inference testing
input_text = "Thou liest, thou shag-hair'd villain!"
input_encoded = encoder.encode(input_text)

# Generate and decode output from the model
_, generated = model.generate_next(input_encoded, temperature=0.7)
decoded = encoder.decode(generated)

# Print and format output, separating the input from the generated output
print(f"{input_text}|{decoded}")