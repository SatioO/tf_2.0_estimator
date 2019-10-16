import pandas as pd
import tensorflow as tf
import numpy as np
import argparse

# parse arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--model-dir', type=str,
#                     default='./saved_model', help='model dir')
# parser.add_argument('--batch-size', type=int,
#                     default=32, help='batch size')
# args = parser.parse_args()

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

feature_columns = []

for column in train.columns:
    feature_columns.append(tf.feature_column.numeric_column(key=column))

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=3,
    hidden_units=[30, 10]
)


def input_fn(features, labels, training=True, batch_size=256):
     # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)


estimator.train(input_fn=lambda: input_fn(train, train_y))
