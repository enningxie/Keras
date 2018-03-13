# processing the labels of the raw IMDB data
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, Reshape, SpatialDropout1D, Conv2D, MaxPool2D, Concatenate, Dropout
from keras import optimizers, metrics, losses
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keras import Input

import pandas as pd
import numpy as np
train_data_path = '/home/enningxie/Documents/Datasets/comment/process_train_.csv'
test_data_path = '/home/enningxie/Documents/Datasets/comment/process_test_.csv'
submission = pd.read_csv('/home/enningxie/Documents/Datasets/comment/sample_submission.csv')

def process_train_data(path):
    raw_data = pd.read_csv(path, dtype={'comment_pure': str})
    train_data = raw_data['comment_pure'].fillna('nan').values
    labels = raw_data.drop(['id', 'comment_text', 'comment_pure'], axis=1).values
    return train_data, labels

def process_test_data(path):
    raw_data = pd.read_csv(path, dtype={'comment_pure': str})
    test_data = raw_data['comment_pure'].fillna('nan').values
    return test_data

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))

train_data, train_labels=process_train_data(train_data_path)
test_data = process_test_data(test_data_path)


maxlen = 200
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(list(train_data)+list(test_data))

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)# list [word_index, word_index...]
word_index = tokenizer.word_index  # dict {word: index}



train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)


print('Shape of train_data tensor:', train_data.shape)
print('Shape of test_data tensor:', test_data.shape)
print('Shape of label tensor:', train_labels.shape)

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


filter_sizes = [1, 2, 3, 5]
num_filters = 32
# define a model
inp = Input(shape=(maxlen,))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False)(inp)
x = SpatialDropout1D(0.4)(x)
x = Reshape((maxlen, embedding_dim, 1))(x)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), kernel_initializer='normal',
                activation='elu')(x)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), kernel_initializer='normal',
                activation='elu')(x)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), kernel_initializer='normal',
                activation='elu')(x)
conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), kernel_initializer='normal',
                activation='elu')(x)

maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)

z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])
z = Flatten()(z)
z = Dropout(0.1)(z)

outp = Dense(6, activation="sigmoid")(z)

model = Model(inputs=inp, outputs=outp)
model.summary()



# train and evaluating the model
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss = losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

X_tra, X_val, y_tra, y_val = train_test_split(train_data, train_labels, train_size=0.95, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

history = model.fit(X_tra, y_tra,
                    epochs=2,
                    verbose=1,
                    batch_size=128,
                    validation_data=(X_val, y_val),
                    callbacks=[RocAuc])

print(history.history)



predict_data = model.predict(test_data, batch_size=256, verbose=1)

print(predict_data.shape)

submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = predict_data
submission.to_csv('submission.csv', index=False)
#
# # save trained model
# model.save('/home/enningxie/Documents/Models/pre_trained_glove_model.h5')

