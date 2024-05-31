import tensorflow as tf
import numpy as np

# Define the file path
file_path = 'common/alllines.txt'

# Initialize an empty list to hold the processed lines
processed_lines = []

# Open and read the file
with open(file_path, 'r') as file:
    for line in file:
        # Strip the line of leading and trailing whitespace (including newlines)
        stripped_line = line.strip()
        # Remove the quotes from the start and end of the line
        if stripped_line.startswith('"') and stripped_line.endswith('"'):
            stripped_line = stripped_line[1:-1]
        # Add the processed line to the list
        processed_lines.append(stripped_line)

# Concatenate all the lines with a space between each
result = ' '.join(processed_lines)
result = np.array(list(result))

# We have a dataset with 76 unique characters. 

# Print the result
chars = len(set(result))
print(f"The dataset has {chars} unique characters, and {len(result)} characters in total.")

char_to_code = {char: idx for idx, char in enumerate(sorted(set(result)))}
code_to_char = {idx: char for char, idx in char_to_code.items()}

encoded_result = [char_to_code[char] for char in result]
num_classes = len(char_to_code)
normalized_encoded = [code / (num_classes - 1) for code in encoded_result]

input_length = 50

def generate_data(input_length, n):
    start_indices = np.random.choice(len(result) - input_length, n, replace=False)
    sequences_array = np.array([encoded_result[i:i + input_length + 1] for i in start_indices])
    X = sequences_array[:,:input_length]
    y = sequences_array[:,-1]
    return (X, y)

def generate_text(model, input_sequence, output_length, input_length = None, temperature=1.0):
    model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    if not input_length:
        input_length = len(input_sequence)
    for i in range(output_length):
        input_text = np.array([input_sequence[-input_length:] / (num_classes - 1)])
        predictions = model.predict(input_text, verbose=0)[0]
        predictions = np.log(predictions + 1e-10) / temperature
        exp_predictions = np.exp(predictions)
        predictions = exp_predictions / np.sum(exp_predictions)
        c = np.random.choice(len(predictions), p=predictions)
        input_sequence = np.append(input_sequence, c)

    text = ''.join([code_to_char[code] for code in input_sequence])
    print(text[:input_length] + '|' + text[output_length:])


use = 1000000
split = 0.8
split_point = int(split * use)

(X, y) = generate_data(input_length, use)
X = X / (num_classes - 1)

X_train = X[:split_point]
y_train = y[:split_point]

X_test = X[split_point:]
y_test = y[split_point:]

# print(X_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_length, activation='sigmoid'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, verbose=2)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

for i in range(10):
    generate_text(model, generate_data(input_length, 1)[0][0], 50, input_length=input_length, temperature=0.7)

# print(tf.one_hot([0, 3, 5, 2], chars))