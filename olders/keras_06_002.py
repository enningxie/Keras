# using keras for word-level one-hot encoding
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# create a tokenizer, configured to only take into account the 1000 most common words
tokenizer = Tokenizer(num_words=1000)
# builds the word index
tokenizer.fit_on_texts(samples)

# turns strings into lists of integer indices
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print(len(one_hot_results[0]))

word_index = tokenizer.word_index
print(word_index)