# start begin mnist
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical


# load mnist dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print dataset's info
print(train_images.shape)
print(len(train_labels))
print(test_images.shape)
print(len(test_labels))

# construct networks
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# set networks
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy']
                )

# resize dataset and normalize it
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# check what's to_categorical work
print('---------------------------')
print(train_labels[:5])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels[:5])

# train
print('--begin--')
network.fit(train_images, train_labels, epochs=5, batch_size=128)
print('--end--')

# test
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc: ', test_acc)