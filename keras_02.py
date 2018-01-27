# classify movie reviews: a binary classification
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

# load imdb data set, keep 10000 most frequently occurring words in the training data.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# train_data, a list of word indices.
print(train_data[0])

# train_labels, 0 for negative and 1 for positive.
print(train_labels[0])

# top 10000 most frequent words.
print(max([max(sequence) for sequence in train_data]))

# decode one of these reviews back to English.
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# the indices are offset by 3 because 0, 1, and 2 are reserved indices for "padding"
# "start of sequence", and "unknown".
decode_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

print(decode_review)


# encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# preprocess data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = train_labels.astype('float32')
y_test = test_labels.astype('float32')

# # the model
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # compile the model
# # model.compile(optimizer='rmsprop',
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
#
# # compile2
# # model.compile(optimizer=optimizers.RMSprop(lr=0.001),
# #               loss='binary_crossentropy',
# #               metrics=['accuracy'])
#
# # compile3
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])
#
#
# setting aside a validation set
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]
#
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#
# # plotting the training and validation loss
# history_dict = history.history  # dict


# a new model to avoid overfitting
model_ = models.Sequential()
model_.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model_.add(layers.Dense(32, activation='relu'))
model_.add(layers.Dense(16, activation='relu'))
model_.add(layers.Dense(1, activation='sigmoid'))

model_.compile(optimizer=optimizers.Adam(lr=0.001),
               loss=losses.binary_crossentropy,
               metrics=[metrics.binary_accuracy])

history = model_.fit(partial_x_train, partial_y_train, epochs=2, batch_size=512, validation_data=(x_val, y_val))

# history_dict = history.history
#
# print("history_dict's keys: ", history_dict.keys())
# loss_value = history_dict['loss']
# val_loss_value = history_dict['val_loss']
#
# epochs = range(1, len(loss_value) + 1)
# plt.plot(epochs, loss_value, 'bo', label='Training loss')
# plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # plotting the training and validation accuracy
# plt.clf()  # clears the figure
# acc_value = history_dict['binary_accuracy']
# val_acc_value = history_dict['val_binary_accuracy']
# plt.plot(epochs, acc_value, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_value, 'b', label='Validation acc')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


loss, acc = model_.evaluate(x_test, y_test)

print("acc: ", acc, " loss: ", loss)