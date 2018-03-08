# numpy implementation of a simple RNN
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps, input_features))

print(inputs)
state_t = np.zeros((output_features,))

W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []

# input_t is a vector of shape (input_features,)
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t

print(successive_outputs)
print(len(successive_outputs))

final_output_sequence = np.concatenate(successive_outputs, axis=0)
print(final_output_sequence)
print(len(final_output_sequence))