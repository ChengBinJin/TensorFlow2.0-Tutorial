import numpy as np
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = \
    fashion_mnist.load_data()

# expand the last dimension for the convolutional layer
train_imgs = np.expand_dims(train_imgs, axis=-1)
test_imgs = np.expand_dims(test_imgs, axis=-1)


print('train_imgs shape: {}'.format(train_imgs.shape))
print('train_labels shape: {}'.format(train_labels.shape))
print('test_imgs shape: {}'.format(test_imgs.shape))
print('test_labels shape: {}'.format(test_labels.shape))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(28, 28, 1),
                           kernel_initializer=keras.initializers.he_normal()),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           kernel_initializer=keras.initializers.he_normal()),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_initializer=keras.initializers.he_normal()),
    tf.keras.layers.Dense(10, activation='softmax',
                          kernel_initializer=keras.initializers.he_normal())
])
model.compile(optimizer='Adam',
              loss=tf.losses.sparse_categorical_crossentropy)
model.fit(train_imgs, train_labels, epochs=5, verbose=1)
model.evaluate(test_imgs, test_labels, verbose=2)
