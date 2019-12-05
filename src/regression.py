import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow import keras


# noinspection PyUnresolvedReferences
print('TF version: {}'.format(tf.__version__))

# Get the data
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print('dataset_path: {}'.format(dataset_path))

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# Clearn the data
print("Before dropping Nan")
print(dataset.isna().sum(), '\n')

dataset = dataset.dropna()
print("After dropping Nan")
print(dataset.isna().sum(), '\n')

dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print('Num. of train_dataset: {}'.format(len(train_dataset)))
print('Num. of test_dataset: {}'.format(len(test_dataset)))

# Inspect the data
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print('\n', train_stats)

# Split features from labels
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Note: Alghough we intentionally generate these statistics from only the training dataset, these statistics will also
# be used to normalize the test dataset. We need to do that to project the test dataset into the same distribution that
# the model has been trained on.

# Normalize the data
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Caution: The statistics used to normalize the inputs (mean and standard deviation) need to be applied to any other
# dat that is fed to the model, along with the one-hot encoding that we did earlier. That includes the test set as well
# as live data when the model is used in predocution.

# The model
# Build the model
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # noinspection PyUnresolvedReferences
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    # noinspection PyUnresolvedReferences
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

model = build_model()

# Inspect the model
model.summary()

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print('example_result: {}'.format(example_result))

# Train the model
EPOCHS = 1000

########################################################################################################################
# Basic
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])

# Visualize the model's training progress using the stats stored in the history object
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# MAE plot
plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

# MSE plot
plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.show()
########################################################################################################################

########################################################################################################################
# Improved
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data, train_labels,
                          epochs=EPOCHS, validation_split=0.2, verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

# MAE plot
plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

# MSE plot
plotter.plot({'Early Stopping': early_history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.show()
########################################################################################################################

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Make predictions
test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
plt.ylabel("Count")
plt.show()