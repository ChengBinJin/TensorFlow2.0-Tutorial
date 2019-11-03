import tensorflow as tf

# Load and prepare the MNIST dataset. Convert the
# samples from integers to floating-point numbers:
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

print('x_test shape: {}'.format(x_test.shape))
print('y_test shape: {}'.format(y_test.shape))

x_train, x_test = x_train / 255., x_test / 255.

# Build the tf.keras.Sequential model by stacking
# layers. Choose an optimizer and loss function
# for training:
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train and evaluate the model
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

