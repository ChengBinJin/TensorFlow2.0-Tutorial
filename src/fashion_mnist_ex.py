import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print('trian_images shape: {}'.format(train_images.shape))
print('train_labels shape: {}'.format(train_labels.shape))
print('test_images shape: {}'.format(test_images.shape))
print('test_labels shape: {}'.format(test_labels.shape))

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=128, activation=tf.nn.relu,
                       kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(units=10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.losses.sparse_categorical_crossentropy)
model.fit(train_images, train_labels, epochs=5, verbose=2)
model.evaluate(test_images, test_labels, verbose=2)

# predictions = model.predict(my_images)