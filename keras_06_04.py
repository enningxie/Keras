import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import optimizers, losses, metrics


imdb_dir = '/home/enningxie/Documents/DataSets/aclImdb/aclImdb'
maxlen = 100
max_words = 10000
embedding_dim = 100
# tokenizing the data of the test set
test_dir = os.path.join(imdb_dir, 'test')

labels_test = []
texts_test = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts_test.append(f.read())
            f.close()
            if label_type == 'neg':
                labels_test.append(0)
            else:
                labels_test.append(1)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts_test)
sequences_test = tokenizer.texts_to_sequences(texts_test)
x_test = pad_sequences(sequences_test, maxlen=maxlen)
y_test = np.asarray(labels_test)

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.load_weights('/home/enningxie/Documents/Models/pre_trained_glove_model.h5')

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss = losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

result = model.evaluate(x_test, y_test)
print(result)
