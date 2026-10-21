"""CPU-only container preflight: parse config and verify sparse label slots."""

import argparse
from pathlib import Path

from google.protobuf import text_format
import tensorflow as tf
from object_detection.builders import model_builder
from object_detection.protos import pipeline_pb2
from object_detection.utils import label_map_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--data", action="store_true")
    args = parser.parse_args()
    config = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Parse(Path(args.config).read_text(), config)
    kind = config.model.WhichOneof("model")
    count = getattr(config.model, kind).num_classes
    assert count == 4, "IDs 1,2,4 require four class slots."
    assert config.train_config.num_steps == 2000
    assert not config.train_config.use_bfloat16
    assert config.train_config.fine_tune_checkpoint_type == "detection"
    expected_batch = 8 if kind == "ssd" else 2
    assert config.train_config.batch_size == expected_batch
    assert config.train_config.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate.total_steps == 2000
    encoded = tf.one_hot(tf.constant([1, 2, 4]) - 1, depth=count)
    tf.debugging.assert_equal(tf.reduce_sum(encoded, axis=1), [1., 1., 1.])
    # Construct on CPU; no gradient step, dataset consumption, or local GPU needed.
    model_builder.build(config.model, is_training=True)
    checkpoint = config.train_config.fine_tune_checkpoint
    assert Path(checkpoint + ".index").is_file()
    assert list(Path(checkpoint).parent.glob("ckpt-0.data-*"))
    assert tf.train.list_variables(checkpoint), "Checkpoint index must be readable."
    if args.data:
        for reader in [config.train_input_reader, *config.eval_input_reader]:
            labels = label_map_util.load_labelmap(reader.label_map_path)
            assert {item.id for item in labels.item} == {1, 2, 4}
            assert tf.io.gfile.glob(reader.tf_record_input_reader.input_path[0])
    print("Config/model/label-slot preflight passed; this is not a GPU memory test.")


if __name__ == "__main__":
    main()
