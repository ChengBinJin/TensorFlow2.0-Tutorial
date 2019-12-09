import os
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from IPython import display
# import shutil

# noinspection PyUnresolvedReferences
print('TF version: {}'.format(tf.__version__))

# logdir = pathlib.Path(tempfile.mkdtemp())/"./tensorboard_logs"
logdir = Path("../tensorboard_logs/overfit_and_underfit")
if not logdir.exists():
    os.makedirs(logdir)

# The Higgs Dataset
# noinspection PyUnresolvedReferences
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz')

FEATURES = 28

# noinspection PyUnresolvedReferences
ds = tf.data.experimental.CsvDataset(gz, [float(),] * (FEATURES + 1), compression_type="GZIP")

def pack_row(*row):
    label_ = row[0]
    # noinspection PyUnresolvedReferences
    features_ = tf.stack(row[1:], axis=1)
    return features_, label_

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features, label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins=101)
    plt.show()

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()

print('train_ds: {}'.format(train_ds))
print('validate_ds: {}'.format(validate_ds))

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

# Demonstrate overfitting
# Train procedure
# noinspection PyUnresolvedReferences
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)

def get_optimizer():
    # noinspection PyUnresolvedReferences
    return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0, 100000)
lr = lr_schedule(step)
plt.figure(figsize=(8, 6))
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0, max(plt.ylim())])
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

def get_callbacks(name):
    # noinspection PyUnresolvedReferences
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        tf.keras.callbacks.TensorBoard(logdir/name)
    ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
    if optimizer is None:
        optimizer = get_optimizer()

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
    model.summary()

    history = model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=max_epochs,
        validation_data=validate_ds,
        callbacks=get_callbacks(name),
        verbose=0)

    return history

# Tiny model
# noinspection PyUnresolvedReferences
tiny_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

size_histories = dict()
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

# noinspection PyUnresolvedReferences
plotter = tfdocs.plots.HistoryPlotter(metric='binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()

# Small model
# noinspection PyUnresolvedReferences
small_model = tf.keras.Sequential([
    # 'input_shape' is only required here so that '.summary' works.
    tf.keras.layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(16, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

# Medium model
# noinspection PyUnresolvedReferences
medium_model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
size_histories['Medium'] = compile_and_fit(medium_model, "sizes/Medium")

# Large model
# noinspection PyUnresolvedReferences
large_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(512, activation='elu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
size_histories['Large'] = compile_and_fit(large_model, "sizes/Large")

# Plot the training and validation losses
plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()

# Note: All the above training runs used the callbacks.EarlyStopping to end the training once it was clear the model was
# not making progress.

# View in TensorBoard
# If you want to share TensorBoard results you can upload the logs to TensorBoard.dev by copying the follwing into
# a code-cell