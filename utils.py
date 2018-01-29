import numpy as np


# decoding newswires back to text
def decoding_newswires(dataset, sequence):
    word_index = dataset.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_result = ' '.join([reverse_word_index.get(i-3, '?') for i in sequence])
    return decode_result

# encoding the data
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# to_one_hot labels
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# create validation_set
def create_validation_set(train_data, train_labels, num_set):
    x_val = train_data[:num_set]
    partial_x_train = train_data[num_set:]
    y_val = train_labels[:num_set]
    partial_y_train = train_labels[num_set:]
    return x_val, y_val, partial_x_train, partial_y_train

