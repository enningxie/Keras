# Functional API implementation of a three-output model
from keras.models import Model
from keras import layers, losses, optimizers, metrics
from keras import Input

vocabulary_size = 50000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(vocabulary_size, 256)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dense(128, activation='relu')(x)

# Note that the output layers are given names
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
geder_prediciton = layers.Dense(1, activation='sigmoid', name='gender')(x)

model = Model(posts_input, [age_prediction, income_prediction, geder_prediciton])

model.compile(optimizer=optimizers.Adam(lr=0.001),
              loss=[losses.mse, losses.categorical_crossentropy, losses.binary_crossentropy],
              loss_weights=[0.25, 1., 10.])

model.compile(optimizer=optimizers.Adam(0.001),
              loss={'age': losses.mse,
                    'income': losses.categorical_crossentropy,
                    'gender': losses.binary_crossentropy},
              loss_weights={'age': 0.25,
                            'income': 1.,
                            'gender': 10.})

# feeding data to a multi-output model
model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)

model.fit(posts, {'age': age_targets,
                  'income': income_targets,
                  'gender': gender_targets}, epochs=10, batch_size=64)

