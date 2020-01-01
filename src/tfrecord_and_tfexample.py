# Note: While useful, these structures are optional. There is no need to convert existing code to use TFRecords, unless
# are using tf.data and reading data is still the bottleneck to training. See Data Input Pipeline Performance for
# dataset performance tips.

# Setup
import numpy as np
import tensorflow as tf

print('Tensorflow version: {}'.format(tf.version.VERSION))

# tf.Example
# The following functions can be used to convert a value to a type compatible
# with tf.Exapmle.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a flaot / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum/ int / unit."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Note: To stay simple, this example only uses scalar inputs. The simplest way to handle non-scalar features is to use
# tf.serialize_tensor to convert tensors to binary-strings. Strings are scalars in tensorflow. Use tf.parse_tensor to
# convert the binary-string back to a tensor.

print('_bytes_feature(b''test_string''): {}\n'.format(_bytes_feature(b'test_string')))
print('_bytes_feature(u''test_bytes''.encode(''utf-8''))): {}\n'.format(_bytes_feature(u'test_bytes'.encode('utf-8'))))

print('_float_feature(np.exp(1)): {}'.format(_float_feature(np.exp(1))))
print('_int64_feature(True): {}'.format(_int64_feature(True)))
print('_int64_feature(1): {}'.format(_int64_feature(1)))

feature = _float_feature(np.exp(1))
print('feature.SerializeToString(): {}'.format(feature.SerializeToString()))

# Creating a tf.Example message
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)

def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _int64_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _float_feature(feature3),
    }

    # Create a Feature message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# This is an example observation from the dataset.
example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print('\nserialized_example: {}'.format(serialized_example))

example_proto = tf.train.Example.FromString(serialized_example)
print('\nexample_proto: {}'.format(example_proto))

# TFRecrods format details
# Note: There is no requirement to use tf.Example in TFRecord files. tf.Example is just a method of serializing
# dictionaries to byte-strings. Lines of text, encoded image data, or serialized tensors (using tf.io.serialize_tensor,
# and tf.io.parse_tensor when loading). See the tf.io module for more options.

# TFRecord files using tf.data
# Writing a TFRecord file

print('\ntf.data.Dataset.from_tensor_slices(feature1): {}'.format(tf.data.Dataset.from_tensor_slices(feature1)))

features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print('\nfeatures_dataset: {}'.format(features_dataset))

# Use 'take(1)' to only pull one example from the dataset.
for f0, f1, f2, f3 in features_dataset.take(1):
    print('f0: {}'.format(f0))
    print('f1: {}'.format(f1))
    print('f2: {}'.format(f2))
    print('f3: {}'.format(f3))
