import numpy as np 
import os

class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_data()

    def load_data(self):
        processed_lines = []
        with open(self.filepath, 'r') as file:
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
        self.raw_text = np.array(list(result))
    
shakespeare = Dataset(os.path.join(__file__, '../alllines.txt'))