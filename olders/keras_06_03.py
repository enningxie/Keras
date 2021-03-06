# processing the labels of the raw IMDB data
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Bidirectional
from keras import optimizers, metrics, losses

import pandas as pd
import numpy as np
import pickle
data_path = '/home/enningxie/Documents/Datasets/comment/process_01.csv'

def process_data(path):
    raw_data = pd.read_csv(path)
    train_data = np.asarray(raw_data['comment_pure'])
    raw_data = raw_data.drop(['id', 'comment_text', 'comment_pure'], axis=1)
    labels = np.asarray(raw_data)
    return train_data, labels

process=process_data(data_path)
# imdb_dir = '/home/enningxie/Documents/DataSets/aclImdb/aclImdb'
# train_dir = os.path.join(imdb_dir, 'train')
#
# labels = []
# texts = []
#
# for label_type in ['neg', 'pos']:
#     dir_name = os.path.join(train_dir, label_type)
#     for fname in os.listdir(dir_name):
#         if fname[-4:] == '.txt':
#             f = open(os.path.join(dir_name, fname))
#             texts.append(f.read())
#             f.close()
#             if label_type == 'neg':
#                 labels.append(0)
#             else:
#                 labels.append(1)
#
#
# print(texts[0])
# print(labels[0])
# print(len(texts[0]))
# Tokenizing the text of the raw IMDB data
maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(process[0])
sequences = tokenizer.texts_to_sequences(process[0])  # list [word_index, word_index...]
word_index = tokenizer.word_index  # dict {word: index}


print('Found %s unique tokens.' % len(word_index))
print('-----')
data = pad_sequences(sequences, maxlen=maxlen)


labels = np.asarray(process[1])
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
#
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
#
# x_train = data[:training_samples]
# y_train = labels[:training_samples]
# x_val = data[training_samples: training_samples + validation_samples]
# y_val = labels[training_samples: training_samples + validation_samples]
#
# Parsing the Glove word-embeddings file
glove_dir = '/home/enningxie/Documents/Codes/xz/comment'
#
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
     values = line.split()
     word = values[0]
     coefs = np.asarray(values[1:], dtype='float32')
     embeddings_index[word] = coefs
f.close()


print('Found %s word vectors.' % len(embeddings_index))

 # preparing the GloVe word-embeddings matrix
embedding_dim = 100
#
embedding_matrix = np.zeros((max_words, embedding_dim))
#
for word, i in word_index.items():
     if i < max_words:
         embedding_vector = embeddings_index.get(word)
         if embedding_vector is not None:
             embedding_matrix[i] = embedding_vector  # words not found in embedding index will be all zeros.

 # define a model
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
# model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True))
# model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.5))
model.add(Bidirectional(LSTM(64, dropout=0.1, recurrent_dropout=0.5)))
model.add(Dense(6, activation='sigmoid'))
model.summary()

# loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# train and evaluating the model
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss = losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

history = model.fit(data, labels,
                    epochs=2,
                    verbose=1,
                    batch_size=128,
                    validation_split=0.2)

print(history.history)
#
# # save trained model
# model.save('/home/enningxie/Documents/Models/pre_trained_glove_model.h5')

