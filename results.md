# Object detection experiment results

All metrics below come from the SageMaker training job logs in CloudWatch
(log group `/aws/sagemaker/TrainingJobs`). Evaluation runs once, after training,
on the full validation set (`sample_1_of_n_eval_examples=1`), so each model has a
single validation point, not a validation curve. Upstream Model Zoo COCO scores
are not project results.

## Experiment settings

| Model | Training steps | Batch | Image presentations (steps × batch) | Class slots / label IDs | Instance |
|---|---:|---:|---:|---|---|
| SSD MobileNet V2 FPNLite 640 | 2,000 | 8 | 16,000 | 4 / {1,2,4} | 1 × ml.g4dn.xlarge |
| Faster R-CNN ResNet50 V1 640 | 2,000 | 2 | 4,000 | 4 / {1,2,4} | 1 × ml.g4dn.xlarge |

Both models use the same optimizer step count, but MobileNet sees four times as
many image presentations, so differences in mAP/loss cannot be attributed to
architecture alone. The unused ID-3 output slot preserves the public cyclist ID 4
without rewriting the TFRecords; there are still only three labeled categories.

Configs: `1_model_training/source_dir/pipeline_ssd_mobilenet.config`,
`1_model_training/source_dir/pipeline_faster_rcnn.config`.

## Measured comparison

| Model | mAP | mAP@.50 | mAP@.75 | mAP small | mAP medium | mAP large | AR@100 | Train total loss (last logged step) | Train total loss (mean of last 3–4 logged steps) | Val total loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SSD MobileNet V2 FPNLite | 0.066 | 0.133 | 0.058 | 0.022 | 0.245 | 0.383 | 0.094 | ≈ 0.534 | ≈ 0.58 | 1.256 |
| Faster R-CNN ResNet50 V1 | **0.082** | **0.172** | **0.068** | **0.032** | **0.295** | **0.650** | **0.141** | 0.757 | ≈ 1.05 | 1.076 |

| Model | Train time | Training compute cost |
|---|---|---|
| SSD MobileNet V2 FPNLite | 19 min (console duration) | ≈ $0.23 (19 min × $0.736/h; confirm with BillableTimeInSeconds) |
| Faster R-CNN ResNet50 V1 | ≈ 18 min (from log timestamps; confirm in console) | ≈ $0.22 (estimate; confirm with BillableTimeInSeconds) |

**Loss totals are not directly comparable across the two models.** MobileNet's
total includes an L2 regularization term of ≈ 0.151; Faster R-CNN's config uses an
L2 weight of 0.0, so its regularization term is 0.000. The two architectures also
sum different components (SSD: classification + localization; Faster R-CNN: RPN
localization + RPN objectness + box-classifier classification + box-classifier
localization). Compare mAP across models, and training vs. validation loss within
each model.

**Single-step training losses are noisy,** especially for Faster R-CNN at batch 2
(its total jumped between 0.71 and 1.67 within the last four logged steps). The
mean of the last few logged steps is the more reliable training figure.

## SSD MobileNet V2 FPNLite — details (`od-ssd-mobilenet-01`)

Source: CloudWatch log stream `od-ssd-mobilenet-01/algo-1-1792611173`.

### COCO detection metrics (evaluation at step 2000, logged 2026-10-21T19:50:14Z)

| Metric | Value |
|---|---:|
| DetectionBoxes_Precision/mAP | 0.065678 |
| DetectionBoxes_Precision/mAP@.50IOU | 0.132971 |
| DetectionBoxes_Precision/mAP@.75IOU | 0.058035 |
| DetectionBoxes_Precision/mAP (small) | 0.022296 |
| DetectionBoxes_Precision/mAP (medium) | 0.245496 |
| DetectionBoxes_Precision/mAP (large) | 0.383211 |
| DetectionBoxes_Recall/AR@1 | 0.016647 |
| DetectionBoxes_Recall/AR@10 | 0.068440 |
| DetectionBoxes_Recall/AR@100 | 0.093675 |
| DetectionBoxes_Recall/AR@100 (small) | 0.046951 |
| DetectionBoxes_Recall/AR@100 (medium) | 0.319015 |
| DetectionBoxes_Recall/AR@100 (large) | 0.648669 |

### Losses

| Loss | First logged training step (19:36:46Z) | Last logged training step (19:49:12Z) | Validation (19:50:14Z) |
|---|---:|---:|---:|
| Classification | 0.468 | 0.182 | 0.498 |
| Localization | 0.781 | 0.201 | 0.606 |
| Regularization | 0.151 | 0.151 | 0.151 |
| Total | ≈ 1.400 | ≈ 0.534 | 1.256 |

Training totals were truncated in the console view, so they are the sum of the
three logged components. Last three logged training totals (19:47:54Z, 19:48:33Z,
19:49:12Z): ≈ 0.555, ≈ 0.643, ≈ 0.534 (mean ≈ 0.58). The validation total (1.256) is
the logged `Loss/total_loss` value.

## Faster R-CNN ResNet50 V1 — details (`od-faster-rcnn-01`)

Source: CloudWatch log stream `od-faster-rcnn-01/algo-1-1792615367`.

### COCO detection metrics (evaluation at step 2000, logged 2026-10-21T20:55:33Z)

Values from the COCO summary table (three decimals as printed).

| Metric | Value |
|---|---:|
| AP @[IoU=0.50:0.95, all] (mAP) | 0.082 |
| AP @[IoU=0.50, all] | 0.172 |
| AP @[IoU=0.75, all] | 0.068 |
| AP @[IoU=0.50:0.95, small] | 0.032 |
| AP @[IoU=0.50:0.95, medium] | 0.295 |
| AP @[IoU=0.50:0.95, large] | 0.650 |
| AR @[maxDets=1] | 0.040 |
| AR @[maxDets=10] | 0.104 |
| AR @[maxDets=100] | 0.141 |
| AR @[maxDets=100, small] | 0.080 |
| AR @[maxDets=100, medium] | 0.508 |
| AR @[maxDets=100, large] | 0.761 |

### Losses

| Loss | Last 4 logged training steps (20:52:53Z / 20:53:17Z / 20:53:40Z / 20:54:04Z) | Validation (20:55:33Z) |
|---|---|---:|
| BoxClassifier classification | 0.161 / 0.252 / 0.143 / 0.224 | 0.173 |
| BoxClassifier localization | 0.349 / 0.529 / 0.350 / 0.243 | 0.246 |
| RPN localization | 0.514 / 0.801 / 0.184 / 0.256 | 0.587 |
| RPN objectness | 0.046 / 0.089 / 0.036 / 0.035 | 0.070 |
| Regularization | 0.0 / 0.0 / 0.0 / 0.0 | 0.000 |
| **Total** | **1.069 / 1.670 / 0.712 / 0.757** (mean ≈ 1.05) | **1.076** |

Learning rate at the last logged step was 0.0 (end of the 2,000-step cosine
schedule).

## Deployment selection

Faster R-CNN is selected for deployment because its measured mAP is higher
(0.082 vs. 0.066). Its artifact was verified against
`DescribeTrainingJob("od-faster-rcnn-01").ModelArtifacts.S3ModelArtifacts`:
`s3://obj-detection-proj/artifacts/od-faster-rcnn-01/od-faster-rcnn-01/output/model.tar.gz`.

## TODO
- Record BillableTimeInSeconds and console duration for both jobs.
