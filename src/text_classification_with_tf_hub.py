import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("TF Version: {}".format(tf.__version__))
print("Eager mode: {}".format(tf.executing_eagerly()))
print("Hub version: {}".format(hub.__version__))
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "Not AVAILABLE")

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(name="imdb_reviews",
                                                     split=(train_validation_split, tfds.Split.TEST),
                                                     as_supervised=True)
# Explore the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
for i, (train_example, train_label) in enumerate(zip(train_examples_batch, train_labels_batch)):
    print('\nID: {} \nSentence: {} \nLabel: {}'.format(i, train_example, train_label))

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
print('Embedding features: {}'.format(hub_layer(train_examples_batch[:3])))
print('Shape of embedding features: {}'.format(hub_layer(train_examples_batch[:3]).shape))

# Let's now build the full model:
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# Loss function and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# Evaluate the model
results = model.evaluate(test_data.batch(512), verbose=1)

print("Test resutls...")
for name, value in zip(model.metrics_names, results):
    print("{}: {:.2%}".format(name, value))
