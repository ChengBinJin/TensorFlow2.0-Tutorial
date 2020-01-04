# Note: While useful, these structures are optional. There is no need to convert existing code to use TFRecords, unless
# are using tf.data and reading data is still the bottleneck to training. See Data Input Pipeline Performance for
# dataset performance tips.

# Setup
import cv2
import numpy as np
from IPython.display import display, Image
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
n_observations = int(1e3)

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


def tf_serialize_example(f0, f1, f2, f3):
    tf_string = tf.py_function(
        serialize_example,
        (f0, f1, f2, f3),   # pass these args to the above function
        tf.string)          # the return type is 'tf.string'
    return tf.reshape(tf_string, ())  # The result is a scalar

# Use 'take(1)' to only pull one example from the dataset.
for f0, f1, f2, f3 in features_dataset.take(1):
    print('f0: {}'.format(f0))
    print('f1: {}'.format(f1))
    print('f2: {}'.format(f2))
    print('f3: {}'.format(f3))
    print(tf_serialize_example(f0, f1, f2, f3))

serialized_features_dataset = features_dataset.map(tf_serialize_example)
print(serialized_features_dataset)


def generator():
    for features in features_dataset:
        yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

print(serialized_features_dataset)

filename = './test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)

# Reading a TFRecrod file
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('raw_dataset: {}'.format(raw_dataset))

# Note: iterating over a tf.data.Dataset only works with eager execution enabled
for raw_record in raw_dataset.take(10):
    print(repr(raw_record))

# Create a description of the feature.
feature_description = {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}

def _parse_function(example_proto):
    # Parse the input 'tf.Example' proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, feature_description)

parsed_dataset = raw_dataset.map(_parse_function)
print('\nparsed_dataset: \n{}'.format(parsed_dataset))

for parsed_record in parsed_dataset.take(10):
    print(repr(parsed_record))

# TFRecrod files in Python
# Writing a TFRecord file
with tf.io.TFRecordWriter(filename) as writer:
    for i in range(n_observations):
        example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
        writer.write(example)

# Reading a TFRecrod file
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print('\nraw_dataset: \n{}'.format(raw_dataset))

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    print('\nexample: \n{}'.format(example))

# Walkthrough: Reading and writing image data
# Fetch the images
cat_in_snow = tf.keras.utils.get_file(
    '320px-Felis_catus-cat_on_snow.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')

print('\ncat_in_snow: \n{}'.format(cat_in_snow))
williamsburg_bridge = tf.keras.utils.get_file(
    '194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
print('\nwilliamsburg_bridege: \n{}'.format(williamsburg_bridge))

img = cv2.imread(cat_in_snow)
print('img(cat) shape: {}'.format(img.shape))
cv2.imshow('Cat in Snow', img)
cv2.waitKey(0)

img = cv2.imread(williamsburg_bridge)
print('img(bridge) shape: {}'.format(img.shape))
cv2.imshow('Bridge', img)
cv2.waitKey(0)

# Write the TFRecord file
image_labels = {
    cat_in_snow: 0,
    williamsburg_bridge: 1
}

# This is an example, just using the cat iamge.
image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

# Create a dictionary with features that may be relevant.
def image_example(image_string, label):
    image_shape = tf.image.decode_jpeg(image_string).shape

    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for line in str(image_example(image_string, label)).split('\n')[:15]:
    print(line)
print('...')

# Write the raw image files to 'images.tfrecrods'.
# First, process the two images into 'tf.Example' messages.
# Then, write to a '.tfrecords' file.
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
    for filename, label in image_labels.items():
        image_string = open(filename, 'rb').read()
        tf_example = image_example(image_string, label)
        writer.write(tf_example.SerializeToString())

# Read the TFRecord file
raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')

# Create a dictionary describing the features.
image_feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    # Parse the inptu tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
print('\nparsed_image_dataset: \n{}'.format(parsed_image_dataset))

for index, image_features in enumerate(parsed_image_dataset):
    image_raw = image_features['image_raw'].numpy()
    img = Image(data=image_raw)
    display(img)
