from keras import Model, layers
import numpy as np
import tensorflow as tf
import warnings

class SimpleFFNModel(Model):
    """Simple feedfoward generation model. 
    Inputs correspond to characters
    in the form of character indices between 0 and :param:`vocab_size - 1`. 

    :param vocab_size: Number of characters in the text vocabulary
    :type vocab_size: int
    :param hidden_units: Number of units in the hidden layer, defaults to 128. 
        If :param:`hidden_layers_units` is specified, this is ignored.
    :type hidden_units: int, optional
    :param hidden_layers_units: List of sizes of hidden layers, defaults to None
    :type hidden_layers_units: List[int], optional
    """
    def __init__(self, vocab_size, hidden_units=128, hidden_layers_units=None, **kwargs):
        """Contructor"""
        super().__init__()
        self.vocab_size = vocab_size

        self.hidden_layers_units = hidden_layers_units

        if self.hidden_layers_units is None:
            self.hidden_layers_units = [hidden_units]
        if not hidden_layers_units:
            raise ValueError(("At least one of 'hidden_units' or "
                             "'hidden_layers_units' must be specified."))

        self.rescale = layers.Rescaling(1./vocab_size)
        self.hidden_layers = []
        for units in hidden_layers_units:
            self.hidden_layers.append(layers.Dense(units, activation="relu"))
        self.output_layer = layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        """Inputs are expected to be in the size (batch_size, :param:`size(inputs)`)
        """
        x = self.rescale(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def get_config(self): 
        config = {'vocab_size': self.vocab_size, 'hidden_layers_units': self.hidden_layers_units}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def predict_next(self, inputs, temperature=0, input_length=None):
        inputs = np.array(inputs).reshape(1, -1)

        if input_length is None:
            input_length = inputs.size

        if input_length > inputs.size:
            inputs = np.pad(inputs, ((0, 0), (input_length - inputs.size, 0)), 'constant', constant_values=(0, 0))

        inputs = inputs[:, -input_length:]

        preds = self.predict(inputs[:, -input_length:], verbose=0).flatten()

        # TODO: Find out why RNN temperature method doesn't work here
        if temperature != 0:
            preds = np.log(preds + 1e-10) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            c = np.random.choice(len(preds), p=preds)
        else:
            c = np.argmax(preds)
        return c
    
    def generate_next(self, start_inputs, input_length=None, generation_length=50, temperature=0):
        start_inputs = np.array(start_inputs).flatten()

        if (input_length == None):
            warnings.warn("Inferring model input length from start_inputs. It's recommended to specify the model's input length.")
            input_length = start_inputs.size
    
        # If the input is too short, pad it with the encoder's padding token
        if input_length > start_inputs.size:
            start_inputs = np.pad(start_inputs, (input_length - start_inputs.size, 0), 'constant', constant_values=(0, 0))

        generated=start_inputs
        c=start_inputs
        for _ in range(generation_length):
            c = self.predict_next(generated, input_length=input_length, temperature=temperature)
            generated = np.append(generated, c)

        return start_inputs, generated[-generation_length:]
    
    @classmethod 
    def from_config(cls, config):
        vocab_size = config.get('vocab_size')
        hidden_layers_units = config.get('hidden_layers_units')
        return cls(vocab_size=vocab_size, hidden_layers_units=hidden_layers_units)
