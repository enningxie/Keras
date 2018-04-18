# classifying newswires, a multiclass classification examples
# for text classification
from keras.datasets import reuters
from utils import decoding_newswires, vectorize_sequences, to_one_hot, create_validation_set
from keras.utils.np_utils import to_categorical
from keras import models, layers
from keras import optimizers, losses, metrics

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# trian_dataset's length
print(len(train_data))

# test_dataset's length
print(len(test_data))

# decoding a sequence
print(decoding_newswires(reuters, train_data[0]))

# the label is an integer between 0 and 45.
# preparing the data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# set the labels to one_hot encoding
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# there is a built-in way to do this in keras
one_hot_train_labels_ = to_categorical(train_labels)
one_hot_test_labels_ = to_categorical(test_labels)


# model definition
def model(x, y, x_val, y_val):
    model_ = models.Sequential()
    model_.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
    model_.add(layers.Dense(64, activation='relu'))
    model_.add(layers.Dense(46, activation='softmax'))
    # compile
    model_.compile(optimizer=optimizers.Adam(lr=0.001),
                   loss=losses.categorical_crossentropy,
                   metrics=[metrics.categorical_accuracy])
    history = model_.fit(x, y, epochs=9, batch_size=512, validation_data=(x_val, y_val))
    return history, model_


# create validation set
x_val, y_val, x_partial, y_partial = create_validation_set(x_train, one_hot_train_labels_, 1000)

history, network = model(x_partial, y_partial, x_val, y_val)

history_dict = history.history

print(history_dict)

loss, acc = network.evaluate(x_test, one_hot_test_labels_)

print("loss: ", loss, " acc: ", acc)

# lalal

