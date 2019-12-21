# Read data using pandas

import pandas as pd
import tensorflow as tf

csv_file = tf.keras.utils.get_file('heart.csv', 'https://storage.googleapis.com/applied-dl/heart.csv')
df = pd.read_csv(csv_file)

print('Before:')
print(df.head())
print(df.dtypes)

df['thal'] = pd.Categorical(df['thal'])
df['thal'] = df.thal.cat.codes

print('After:')
print(df.head())
print(df.dtypes)

# Load data using tf.data.Dataset
target = df.pop('target')

dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

for feat, targ in dataset.take(5):
    print('Features: {}, Target: {}'.format(feat, targ))

print('tf.constant(df[''thal'']): {}'.format(tf.constant(df['thal'])))
train_dataset = dataset.shuffle(len(df)).batch(1)

# Create and train a model
def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrices=[tf.keras.metrics.BinaryAccuracy()])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs=15)

# Alternative to feature columns
inputs = {key: tf.keras.layers.Input(shape=(), name=key) for key in df.keys()}
x = tf.stack(list(inputs.values()), axis=-1)

x = tf.keras.layers.Dense(10, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model_func = tf.keras.Model(inputs=inputs, outputs=output)

model_func.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

for dict_slice in dict_slices.take(1):
    print(dict_slice)

model_func.fit(dict_slices, epochs=15)