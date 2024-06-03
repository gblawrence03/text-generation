import numpy as np

class Encoder():
    def __init__(self, data):
        self.data = data

class Characters(Encoder):
    def __init__(self, data):
        super().__init__(data)
        self.unique_chars = len(set(self.data))
        self.char_to_code = {char: idx for idx, char in enumerate(sorted(set(self.data)))}
        self.code_to_char = {idx: char for char, idx in self.char_to_code.items()}
        self.int_encoded = [self.char_to_code[char] for char in self.data]

    def random_train_data(self, input_length, n):
        start_indices = np.random.choice(len(self.data) - input_length, n, replace=False)
        sequences_array = np.array([self.int_encoded[i:i + input_length + 1] for i in start_indices])
        X = sequences_array[:,:input_length]
        y = sequences_array[:,-1]
        return (X, y) 
    
    def normalise_encoded(self, text_encode = None):
        if text_encode is None:
            return self.int_encoded / (self.unique_chars - 1)
        else:
            return text_encode / (self.unique_chars - 1)
        
    def decode(self, text_decode = None):
        if text_decode is None:
            return ''.join([self.code_to_char[code] for code in self.int_encoded])
        else:
            return ''.join([self.code_to_char[code] for code in text_decode])