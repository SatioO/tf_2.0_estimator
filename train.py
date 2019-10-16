import pandas as pd
import tensorflow as tf
import numpy as np
import argparse

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', type=str,
                    default='./saved_model', help='model dir')
parser.add_argument('--batch-size', type=int,
                    default=32, help='batch size')
args = parser.parse_args()


class Model(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def input_fn(self):
        pass

    def model_fn(self, features, labels, mode):
        if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
            loss = None
        else:
            loss = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = None
        else:
            train_op = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = None
        else:
            predictions = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op)


def main():
    # init model class
    model = Model(args.batch_size)
    # create classifier
    classifier = tf.estimator.Estimator(
        model_dir=args.model_dir, model_fn=model.model_fn)

    classifier.train(input_fn=model.input_fn)


if __name__ == "__main__":
    main()
