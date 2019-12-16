import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# noinspection PyUnresolvedReferences
print('TF version: {}'.format(tf.__version__))
print('TFDS version: {}'.format(tfds.__version__))

# Download the IMDB dataset
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k',
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the 'info' structure.
    with_info=True)

# Try the encoder
encoder = info.features['text'].encoder
print('Vocabulary size: {}'.format(encoder.vocab_size))

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print('The original string: "{}"'.format(original_string))

assert original_string == sample_string

for ts in encoded_string:
    print('{} -----> {}'.format(ts, encoder.decode([ts])))

# Explore the data
for train_example, train_label in train_data.take(1):
    print('Encoded text: {}'.format(train_example[:10].numpy()))
    print('Label: {}'.format(train_label.numpy()))
    print('Decode: \n{}'.format(encoder.decode(train_example)))

# Prepare the data for training
BUFFER_SIZE = 1000

train_batches = (train_data.shuffle(BUFFER_SIZE).padded_batch(32, train_data.output_shapes))
test_batches = (test_data.padded_batch(32, train_data.output_shapes))

for example_batch, label_batch in train_batches.take(2):
    print("Batch shape: {}".format(example_batch.shape))
    print("label shape: {}".format(label_batch.shape))

# Build the model
# Caution: This model doesn't use masking, so the zero-padding is used as part of the input, so the padding length
# may affect the output. To fix this, see the masking and padding guide.

model = keras.Sequential([
    keras.layers.Embedding(encoder.vocab_size, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(1, activation='sigmoid')])
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_batches,
                    epochs=30,
                    validation_data=test_batches,
                    validation_steps=30)

# Evaluate the model
loss, accuracy = model.evaluate(test_batches)
print("Loss: {:.3f}".format(loss))
print('Accuracy: {:.2%}'.format(accuracy))

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo-', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()  # clear figure
plt.plot(epochs, acc, 'bo-', label='Training acc')
plt.plot(epochs, val_acc, 'ro-', label='validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
