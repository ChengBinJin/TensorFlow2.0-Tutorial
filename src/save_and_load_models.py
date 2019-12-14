import os
import tensorflow as tf
from tensorflow import keras

# noinspection PyUnresolvedReferences
print('TF version: {}'.format(tf.version.VERSION))

# Get an example dataset
# noinspection PyUnresolvedReferences
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

print('train_labels shape: {}'.format(train_labels.shape))
print('test_labels shape: {}'.format(test_labels.shape))
print('train_images shape: {}'.format(train_images.shape))
print('test_images shape: {}'.format(test_images.shape))

# Define a model
# Define a simple sequential model
# noinspection PyUnresolvedReferences
def create_model():
    model_ = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model_.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    return model_

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

# Save checkpoints during training
# Checkpoint callback usuage
checkpoint_path = "../training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# noinspection PyUnresolvedReferences
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

# Create a baisc model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2%}".format(acc))

# Loades the weights
model.load_weights(checkpoint_path)

# Re-evaluate teh model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2%}".format(acc))

# Include the epoch in the file name (use 'str.format')
checkpoint_path = "../training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir_ = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
# noinspection PyUnresolvedReferences
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    peroid=5)

# Creat a new model instance
model = create_model()

# Save the weights using the 'checkpoint_path' format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images,
          train_labels,
          epochs=50,
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# noinspection PyUnresolvedReferences
latest = tf.train.latest_checkpoint(checkpoint_dir_)
print('\nLatest model name: {}'.format(latest))
# Note: the default tensorflow format only saves the 5 most recent checkpoints.

# Create a new model instance
model = create_model()

# Load the prevously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2%}".format(acc))

# Note: the default tensorflow format only saves the 5 most recent checkpoints

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2%}".format(acc))

# Manually save weights
# Save the weights
model.save_weights('../checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore teh weights
model.load_weights('../checkpoints/my_checkpoint')

# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2%}".format(acc))

# Save the entire model
# Call model.save to save the a model's architecture, weights, and training configuration in a single file/folder.

# HDF5 format
# Create and train a new model instance
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

print("Now, recreate the model from that file:")
# Recreate the exact same model, including its weights and the optimizer
# noinspection PyUnresolvedReferences
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

# Check its accuracy
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2%}'.format(acc))

# SavedModel format
# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entimre model as a SavedModel.
model.save('saved_model/my_model')

# noinspection PyUnresolvedReferences
new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# Evaluate teh restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2%}'.format(acc))

print('new_model.predict(test_images).shape: {}'.format(new_model.predict(test_images).shape))