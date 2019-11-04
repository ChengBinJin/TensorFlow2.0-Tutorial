# Import dependencies
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os

print('Tensorflow version: {}'.format(tf.__version__))

# Download the dataset
datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)
mnist_train = datasets['train']
mnist_test = datasets['test']

# print('mnist_train shape: {}'.format(mnist_train.shape))
# print('mnist_test shape: {}'.format(mnist_test.shape))

strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(
    strategy.num_replicas_in_sync))

# Setup input pipeline

# You can also do info.splits.total_num_exapmles
# to get the total number of examples in
# the dataset.
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 100000
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label

train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)

# Create the model
# with strategy.scope():
#    model = tf.keras.Sequential([
#        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
#        tf.keras.layers.MaxPooling2D(),
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(64, activation='relu'),
#        tf.keras.layers.Dense(10, activation='softmax')
#    ])

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Define the checkpoint directory to store the checkpoints
checkpoint_dir = '../training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

# Function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif (epoch >= 3) and (epoch < 7):
        return 1e-4
    else:
        return 1e-5

# Callback for printing the LR at the end of each epoch.
class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(
            epoch + 1, model.optimizer.lr.numpy()))

callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='../logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                       save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
]

# Train and evaluate
model.fit(train_dataset, epochs=12, callbacks=callbacks)










