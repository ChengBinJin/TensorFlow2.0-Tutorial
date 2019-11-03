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






