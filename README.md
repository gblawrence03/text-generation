# What is this?
This is a library for creating, training, testing, and running various models for text generation. 
I'm using it for my own learning in text generation with Python, but it could also be used in projects. 
This library provides
- Loading and preprocessing of datasets for training
- Simple pipelines for creating and training models with checkpointing
- Model inference for producing strings of text of variable length

# What models are provided?
Currently, this is a very minimal library, and the only provided model is a simple feedforward model - SimpleFFN - which takes encoded characters.
The goal is to expand my text generation knowledge with models of increasing complexity, including RNNs, LSTMs, and eventually including transformers. 
The minimal setup of the library so far will make it much easier to create, train, and test new models. 
