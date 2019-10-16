import pandas as pd
import tensorflow as tf
import numpy as np

tf.random.set_seed(0)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

train = pd.read_csv("./iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv("./iris_test.csv", names=CSV_COLUMN_NAMES, header=0)

train_y = train.pop('Species')
test_y = test.pop('Species')

# Feature Engineering for the Model
feature_columns = []
for column in train.columns:
    feature_columns.append(tf.feature_column.numeric_column(key=column))


def make_input_fn(features, labels, training=False, batch_size=32):
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

for feature_batch, label_batch in train_input_fn().take(1):
    print('Some feature keys:', list(feature_batch.keys()))
    print('A batch of class:', feature_batch['SepalLength'].numpy())
    print('A batch of Labels:', label_batch.numpy())

# Train the Model.
estimator.train(input_fn=train_input_fn, steps=5000)
# Evaluate the Model
eval_result = estimator.evaluate(input_fn=eval_input_fn)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
