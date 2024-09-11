import sys
from os import path
from sklearn.model_selection import train_test_split

sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from models.simple_ffn import SimpleFFNModel
from training.trainer import train
from datasets.datasets import shakespeare
from preprocessing.encoders import Characters
from inference.generator import generate_next
from utils.checkpoint import load_latest

model_name = "example-ffn"

# Grab the Shakespeare text dataset
dataset = shakespeare.raw_text

# Create the character encoder for the dataset
encoder = Characters(dataset)
                           
# Define an input length (or context length) for the model
input_length = 32

# Grab some random train data in encoded format and split
X, y = encoder.random_train_data_enc(input_length, 100000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Try to load existing SimpleFFN model
try:
    print("Looking for existing model...")
    model = load_latest(model_name)
except FileNotFoundError as e:
    # Create one if not found
    print("Model not found, creating new one...")
    model = SimpleFFNModel(encoder.vocab_size, hidden_layers_units=[128, 128, 128])

# Train the model on the training data and save checkpoints
train(model, X_train, y_train, epochs=10, batch_size=32, save_name=model_name)

# Evaluate the model
model.evaluate(X_test, y_test, verbose=2)

# Encode an input string for inference testing
input_text = "Thou liest, thou shag-hair'd villain!"
input_encoded = encoder.encode(input_text)

# Generate and decode output from the model
_, generated = generate_next(model, input_encoded, input_length=input_length, temperature=0.7)
decoded = encoder.decode(generated)

# Print and format output, separating the input from the generated output
print(f"{input_text}|{decoded}")