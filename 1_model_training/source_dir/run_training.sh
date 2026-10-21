#!/bin/bash
set -euo pipefail
trap 'printf "Training pipeline failed at line %s; export will not continue.\n" "$LINENO" >&2' ERR

MODEL_DIR="${SM_HP_MODEL_DIR:?Missing model directory}"
PIPELINE_CONFIG_PATH="${SM_HP_PIPELINE_CONFIG_PATH:?Missing pipeline config}"
NUM_TRAIN_STEPS="${SM_HP_NUM_TRAIN_STEPS:?Missing step count}"
SAMPLE_1_OF_N_EVAL_EXAMPLES="${SM_HP_SAMPLE_1_OF_N_EVAL_EXAMPLES:-1}"
NUM_WORKERS="${SM_NUM_GPUS:-0}"
if (( NUM_WORKERS < 1 )); then
    printf 'GPU training was requested but no GPU is available.\n' >&2
    exit 1
fi
if [[ "$NUM_TRAIN_STEPS" != "2000" ]]; then
    printf 'Unreviewed training step count; refusing to run.\n' >&2
    exit 1
fi

python validate_pipeline.py "$PIPELINE_CONFIG_PATH" --data
mkdir -p "$MODEL_DIR" /opt/ml/model

printf '=== TRAINING ===\n'
python model_main_tf2.py \
    --pipeline_config_path "$PIPELINE_CONFIG_PATH" \
    --model_dir "$MODEL_DIR" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --num_workers "$NUM_WORKERS" \
    --checkpoint_every_n 500 \
    --alsologtostderr

printf '=== EVALUATION (same instance; no parallel job) ===\n'
python model_main_tf2.py \
    --pipeline_config_path "$PIPELINE_CONFIG_PATH" \
    --model_dir "$MODEL_DIR" \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --sample_1_of_n_eval_examples "$SAMPLE_1_OF_N_EVAL_EXAMPLES" \
    --checkpoint_dir "$MODEL_DIR" \
    --eval_timeout 10 \
    --alsologtostderr

# Preserve the trained pipeline/checkpoints/events for backup and later analysis.
# TODO: Evaluate intermediate checkpoints sequentially if validation curves are needed.
cp "$PIPELINE_CONFIG_PATH" "$MODEL_DIR/pipeline.config"
printf '=== EXPORT ===\n'
EXPORT_DIR="$(mktemp -d /tmp/od-export-XXXXXX)"
python exporter_main_v2.py \
    --input_type image_tensor \
    --trained_checkpoint_dir "$MODEL_DIR" \
    --pipeline_config_path "$PIPELINE_CONFIG_PATH" \
    --output_directory "$EXPORT_DIR"
test -f "$EXPORT_DIR/saved_model/saved_model.pb"
mv "$EXPORT_DIR/saved_model" /opt/ml/model/1
