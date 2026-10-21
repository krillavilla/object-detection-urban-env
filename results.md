# Object detection experiment results

No Phase 2 notebook or training job has been executed by the coding agent.
All measured fields remain **Not measured** until logs, eval output, or TensorBoard
event files are provided in Phase 3. Upstream Model Zoo COCO scores are not
project results.

## Requested experiment settings, not measured results

| Model | Training steps | Batch | Requested image presentations | Class slots / label IDs |
|---|---:|---:|---:|---|
| SSD MobileNet V2 FPNLite 640 | 2,000 | 8 | 16,000 | 4 / {1,2,4} |
| Faster R-CNN ResNet50 V1 640 | 2,000 | 2 | 4,000 | 4 / {1,2,4} |

Sources: `1_model_training/source_dir/pipeline_ssd_mobilenet.config`,
`1_model_training/source_dir/pipeline_faster_rcnn.config`, and experiment settings
in `1_model_training/phase2.py`. Presentations are **requested steps × batch**,
not unique images, completed examples, or a measured dataset count.

The models have the same requested optimizer-step count, but MobileNet receives
four times as many image presentations. Differences in mAP/loss cannot be
attributed solely to architecture. Training time and compute are also not matched.
The unused ID-3 output slot preserves public cyclist ID 4 without remapping
TFRecords; there are still only three labeled semantic categories.

## Measured comparison — pending Phase 3

| Model | mAP | mAP@.5 | mAP small | Final train loss | Final val loss | Train time | Cost | Artifact S3 path |
|---|---|---|---|---|---|---|---|---|
| SSD MobileNet V2 FPNLite | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not available |
| Faster R-CNN ResNet50 V1 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not available |

TODO: Add mAP@.75, mAP medium/large, classification/localization/regularization/
total train and eval losses with source files, event tags, and step numbers.
Faster R-CNN has RPN and second-stage loss components; preserve those rather than
pretending its raw loss is identical in definition to SSD's.
TODO: Copy actual artifact URIs from DescribeTrainingJob, and cite training time
and BillableTimeInSeconds separately. Training compute cost is billable seconds
times the verified on-demand instance rate; endpoint/notebook/storage costs are
separate. Estimates in COST GUARD cells must not be reported as actual costs.

The original script evaluates after training. A single final validation point is
not a curve and cannot establish its shape or an overfitting trend.
An optional EfficientDet result must come from a separate guarded copy with
`num_classes: 4`; the original baseline notebook/config are unchanged references.
