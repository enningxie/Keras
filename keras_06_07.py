# rnn in keras
# preparing the IMDB data
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.models import Sequential
from keras import losses, optimizers, metrics

# number of words to consider as features
max_features = 10000
maxlen=500
batch_size = 32

print('loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_trian shape:', input_train.shape)
print('input_test shape:', input_test.shape)

# training the model with embedding and simpleRnn layers
model = Sequential()
model.add(Embedding(max_features, 32))
# model.add(SimpleRNN(32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

print(history.history)