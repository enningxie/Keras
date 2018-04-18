# feature extraction with data augmentation
from keras.applications import VGG16
from keras import models, layers, metrics, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator

train_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data/train'
validation_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data/validation'
test_dir = '/home/cooli/Documents/DataSets/cat_vs_dog/data/test'

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

model = models.Sequential()

model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())
print('...')
print(len(model.trainable_weights))

conv_base.trainable = False
print('...')
print(len(model.trainable_weights))

model.compile(loss=losses.binary_crossentropy,
              optimizer=optimizers.Adam(lr=0.001),
              metrics=[metrics.binary_accuracy])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

print(history.history)