import pandas as pd
import tensorflow as tf
import numpy as np
import argparse

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# Feature Engineering for the Model
feature_columns = []
for column in train.columns:
    feature_columns.append(tf.feature_column.numeric_column(key=column))


def make_input_fn(features, labels, training=False, batch_size=256):
    def input_fn():
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # Shuffle and repeat if you are in training mode.
        if training:
            dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

    return input_fn


estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    hidden_units=[30, 10]
)

train_input_fn = make_input_fn(train, train_y, training=True)
eval_input_fn = make_input_fn(test, test_y)

# Train the Model.
estimator.train(input_fn=train_input_fn, steps=5000)
# Evaluate the Model
eval_result = estimator.evaluate(input_fn=eval_input_fn)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
