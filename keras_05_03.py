# a small convnet for dog vs cat classification
from keras import layers
from keras import models
from keras import optimizers, losses, metrics
from keras.preprocessing.image import ImageDataGenerator
# from keras_05_02 import train_dir, validation_dir
import matplotlib.pyplot as plt
train_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data/train'
validation_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data/validation'

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

# compile_op
model.compile(loss=losses.binary_crossentropy,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=[metrics.binary_accuracy])

# preprocessing the imagedata
# rescales all images
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150, 150),  # resize all images to 150x150
    batch_size=20,
    class_mode='binary'  # binary
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# fitting the model using a batch generator
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50
)

print(history.history)
# saving the model
model.save('cats_and_dogs_small_1.h5')



