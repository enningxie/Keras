# Functional API implementation of a two-input question-answering model
from keras.models import Model
from keras import layers
from keras import Input
from keras import losses, optimizers, metrics
import numpy as np

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

text_input = Input(shape=(None,), dtype='int32', name='text')

embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)

encoded_text = layers.LSTM(32)(embedded_text)

question_input = Input(shape=(None,), dtype='int32', name='question')

embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)

encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)

answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)

model = Model([text_input, question_input], answer)
model.summary()
model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.categorical_accuracy])

# feeding data to a multi-input model
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, answer_vocabulary_size))
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))
# history = model.fit([text, question], answers, epochs=10, batch_size=128)

history = model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

print(history.history)