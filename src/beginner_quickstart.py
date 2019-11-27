import tensorflow as tf

mnist = tf.keras.datasets.mnist

# Load and prepare the MNIST dataset. Convert the samples from integers to floating-point numbers:
# noinspection PyUnresolvedReferences
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255., x_test / 255.

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('x_test shape: {}'.format(x_test.shape))
print('y_test shape: {}'.format(y_test.shape))

# Build the tf.keras.Sequential model by stacking layers.
# Choose an optimizer and loss function for training:
# noinspection PyUnresolvedReferences
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# noinspection PyUnresolvedReferences
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# Train and evalaute the model:
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)