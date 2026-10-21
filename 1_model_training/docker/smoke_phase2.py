"""Executed inside the image build, not on the user's CPU notebook kernel."""

import tensorflow as tf
from object_detection import inputs
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2

assert tf.__version__ == "2.13.0"
assert inputs._LABEL_OFFSET == 1
assert callable(model_builder.build)
assert pipeline_pb2.TrainEvalPipelineConfig()
tf.debugging.assert_equal(
    tf.one_hot(tf.constant([1, 2, 4]) - inputs._LABEL_OFFSET, 4),
    [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]],
)
print("TensorFlow/Object Detection API imports and sparse-label one-hot check passed.")
