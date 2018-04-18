# Predicting house prices: a regression example
from keras.datasets import boston_housing
from utils import normalize_data
from keras import models, layers, optimizers, losses, metrics

# loading the Boston housing dataset
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()
# dataset info
print(train_data.shape)
print(test_data.shape)

# preparing the data
# Normalizing the data
train_data = normalize_data(train_data)
test_data = normalize_data(test_data)

# build your network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.Adam(lr=0.001), loss=losses.mse, metrics=[metrics.mae])
    return model

network = build_model()
# train
network.fit(train_data, train_labels, epochs=80)
loss, acc = network.evaluate(test_data, test_labels)
print(loss, acc)