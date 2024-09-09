import numpy as np
import warnings 

def predict_next(model, inputs, input_length=None, temperature=0):
    inputs = np.array(inputs).reshape(1, -1)

    if input_length is None:
        input_length = inputs.size

    if input_length > inputs.size:
        inputs = np.pad(inputs, ((0, 0), (input_length - inputs.size, 0)), 'constant', constant_values=(0, 0))
        
    preds = model.predict(inputs[:, -input_length:], verbose=0).flatten()

    if temperature != 0:
        preds = np.log(preds + 1e-10) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        c = np.random.choice(len(preds), p=preds)
    else:
        c = np.argmax(preds)
    return c

def generate_next(model, start_inputs, input_length=None, generation_length=50, temperature=0):
    start_inputs = np.array(start_inputs).flatten()

    if (input_length == None):
        warnings.warn("Inferring model input length from start_inputs. It's recommended to specify the model's input length.")
        input_length = start_inputs.size
    
    if input_length > start_inputs.size:
        start_inputs = np.pad(start_inputs, (input_length - start_inputs.size, 0), 'constant', constant_values=(0, 0))

    generated=start_inputs
    for _ in range(generation_length):
        c = predict_next(model, generated, input_length=input_length, temperature=temperature)
        generated = np.append(generated, c)

    return generated