import numpy as np 
import os
from pathlib import Path

class Dataset:
    """Holds a raw text dataset
    """
    def __init__(self, filepath, encoding="UTF-8"):
        self.encoding = encoding
        self.filepath = filepath
        self.load_data()

    def load_data(self):
        processed_lines = []
        with open(self.filepath, 'r', encoding=self.encoding) as file:
            for line in file:
                # Strip the line of leading and trailing whitespace (including newlines)
                stripped_line = line.strip()
                # Remove the quotes from the start and end of the line
                if stripped_line.startswith('"') and stripped_line.endswith('"'):
                    stripped_line = stripped_line[1:-1]
                # Add the processed line to the list
                processed_lines.append(stripped_line)

        # Concatenate all the lines with a space between each
        self.raw_text = '\n'.join(processed_lines)
    
data_path = os.path.join(os.path.dirname(__file__), "/data")
shakespeare = Dataset(os.path.join(os.path.dirname(__file__), 'data/shakespeare_alllines.txt'))
