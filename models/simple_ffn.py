from keras import Model, layers

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
    def __init__(self, vocab_size, hidden_units=128, hidden_layers_units=None):
        """Contructor"""
        super().__init__()
        self.vocab_size = vocab_size

        if hidden_layers_units is None:
            hidden_layers_units = [hidden_units]
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
