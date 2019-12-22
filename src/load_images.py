import os
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

print('TensorFlow version: {}'.format(tf.version.VERSION))

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Retrieve the images
data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
                                   fname='flower_photos', untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

print('data_dir: {}'.format(data_dir))
print('image_count: {}'.format(image_count))

CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
print('CLASS_NAMES: {}'.format(CLASS_NAMES))

roses = list(data_dir.glob('roses/*'))
for image_path in roses[:3]:
    img = cv2.imread(str(image_path))
    cv2.imshow("Rose", img)
    cv2.waitKey(0)

# Load using keras.preprocessing
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes=list(CLASS_NAMES))

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')

    plt.show()

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

# Load using tf.data
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D unit8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use 'convert_image_dtype' to convert to floats in the [0, 1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the iamge to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set 'num_parallel_calls' so multiple images are loaded/processed in parallel
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
    print("Image shape: {}".format(image.numpy().shape))
    print("Label: {}".format(label.numpy()))
    print("Min value: {}".format(np.min(image.numpy())))
    print("Max value: {}".format(np.max(image.numpy())))

# Basic methods for training
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use '.cache(filename)' to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # 'prefetch' lets the datase fetch batches in the background while the model
    # is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))

show_batch(image_batch.numpy(), label_batch.numpy())

# Performance
# Note: This section just shows a couple of easy tricks that may help performance. For an in depth guide see Input
# Pipeline Performance

import time
default_timeit_steps = 1000

def timeit(ds, steps=default_timeit_steps):
    start = time.time()
    it = iter(ds)

    for i in range(steps):
        batch = next(it)
        if i % 10 == 0:
            print('.', end='')
    print()
    end = time.time()

    duration = end - start
    print("{} batches: {} s".format(steps, duration))
    print("{:0.5f} Images/s".format(BATCH_SIZE * steps / duration))

# 'keras.preprocessing'
print("\nkeras.preprocessing:")
timeit(train_data_gen)

# 'tf.data'
print("\nCached tf.data:")
timeit(train_ds)

# A large part of the performance gain comes from the use of .cache
uncached_ds = prepare_for_training(labeled_ds, cache=False)
print('\nUncached tf.data:')
timeit(uncached_ds)

# If the dataset doesn't fit in memory use a cache file to maintain some of the advantages:
filecache_ds = prepare_for_training(labeled_ds, cache="./flowers.tfcache")
print("\nFilecached tf.data:")
timeit(filecache_ds)


