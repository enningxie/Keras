# classify movie reviews: a binary classification
from keras.datasets import imdb

# load imdb data set, keep 10000 most frequently occurring words in the training data.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# train_data, a list of word indices.
print(train_data[0])

# train_labels, 0 for negative and 1 for positive.
print(train_labels[0])

# top 10000 most frequent words.
print(max([max(sequence) for sequence in train_data]))

# decode one of these reviews back to English.
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# the indices are offset by 3 because 0, 1, and 2 are reserved indices for "padding"
# "start of sequence", and "unknown".
decode_review = ' '.join([reverse_word_index.get(i-3, '?') for i in train_data[0]])

print(decode_review)

# todo