# train
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers, losses, metrics

# load the mnist data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
train_data = train_data.reshape((60000, 28, 28, 1))
train_data = train_data.astype('float32') / 255
test_data = test_data.reshape((10000, 28, 28, 1))
test_data = test_data.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

# adding a classifier on top of the convnet
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# train_op
model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.categorical_crossentropy, metrics=[metrics.categorical_accuracy])
model.fit(train_data, train_labels, epochs=5, batch_size=128)

# test_op
test_loss, test_acc = model.evaluate(test_data, test_labels)

# display the net's architecture.
print(model.summary())

# print
print(test_loss, test_acc)
