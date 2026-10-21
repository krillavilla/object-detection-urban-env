# Object detection in an urban environment

## Public repository safety

This is a standalone coursework copy of the
[Udacity starter](https://github.com/udacity/cd2688-object-detection-in-urban-environment-project),
retaining its original history and license. Original training/deployment notebooks
and the EfficientNet pipeline configuration are unchanged during repository setup.

After cloning, enable the repository's credential checks (repeat on the Notebook Instance):

```bash
git config --local core.hooksPath .githooks
python3 scripts/check_aws_secrets.py
```

The pre-commit hook checks all staged files, including saved notebook outputs.
The pre-push hook checks reachable Git history. Both reject likely AWS access-key
IDs and AWS secret/session credential field names without printing their values.
Hooks are local, can be bypassed, and pattern checks are not a guarantee: review
notebooks and diffs before committing. No server-side credential check is
configured. Never put AWS keys or tokens in this repository; the Notebook Instance should use
its execution role.

The ignore rules exclude local credentials, environment files, TFRecords,
downloaded checkpoints, exported models, and video files. Existing starter images
and the example animation remain tracked. Keep generated videos and large model
artifacts outside Git and download them separately for backup.

Training and deployment launch paid AWS resources. Repository setup does not
launch any resources. **TODO:** Complete the read-only budget review before
running either original notebook; use us-east-1, generic resource names, at most one
training job and one endpoint, and remove billable resources promptly.

## Phase 2: use the guarded model copies

The supplied Amazon Linux 2023 environment is a **SageMaker Notebook Instance**, kernel
`conda_python3`, not Studio. Its Git repository setting clones here:
`/home/ec2-user/SageMaker/object-detection-urban-env`.
Use the supplied `AmazonSageMaker-ExecutionRole`; do not configure manual keys.
The role is expected to have AmazonSageMakerFullAccess, AmazonS3FullAccess,
and AmazonEC2ContainerRegistryFullAccess. Lab restrictions still take precedence.
No notebook cell creates IAM roles/policies or the data bucket.

Open, in this order:

1. `1_model_training/1_train_model_ssd_mobilenet.ipynb`
2. `1_model_training/1_train_model_faster_rcnn.ipynb`, only after the first job terminates.

**Do not Run All.** Build and training have separate one-shot switches, both off
by default. Enter the unspent portion of the $15 project cap and remaining
planned active notebook hours when instructed; do not count already-spent
notebook time again in the future reserve.
Existing completed/active jobs are inspected without resubmission; failed/stopped
jobs require review, not automatic retries. Keep the local
`1_model_training/.phase2/run_records.json` journal for session recovery and backup.
It is ignored by Git; don't delete it to work around an ambiguous submission.
Only one controller may launch work; local locking and AWS status checks cannot
prevent another user/console from racing the account's quotas.

The original `1_train_model.ipynb` and `source_dir/pipeline.config` are preserved
byte-for-byte as **reference-only, unsafe-to-run baseline files**. The original
launch cell is the exception to adding a COST GUARD because that would modify
the protected notebook. An optional EfficientDet experiment must use a separate
copy with the same label-slot fix and the reviewed runtime guards.
The deployment notebook has a warning cell, but its actual deployment/cleanup
implementation is deferred to Phase 6. Do not run it yet.

### Model configs and differences

Both new configs retain the Model Zoo architecture and augmentation options.
They use the public data/label maps, **four class slots** for three categories
with IDs **1, 2, 4**, 2,000 steps, one T4 GPU, and detection checkpoint restore.
Original Model Zoo COCO mAP comments are upstream reference claims, not results
on this project. Authentic references, SHA-256 provenance, and full unified diffs:

- `1_model_training/model_zoo/provenance.json`
- `1_model_training/model_zoo/ssd_mobilenet.diff`
- `1_model_training/model_zoo/faster_rcnn.diff`

One-line reasons for every adaptation:

| Change | MobileNet | Faster R-CNN | Reason |
|---|---|---|---|
| Class slots | 90 → 4 | 90 → 4 | Preserve public sparse IDs 1,2,4 without rewriting data. |
| Batch size | 128 → 8 | 64 → 2 | Starting memory estimates for a 16 GB T4; not measured maxima. |
| Checkpoint path | `checkpoint/ckpt-0` | Same | Bundle only this model's downloaded COCO detection weights. |
| Checkpoint type | classification → detection | Same | Use the detection Model Zoo tarball, not ImageNet-only weights. |
| Synchronous replicas | true/8 → false/1 | Same | One training instance with one GPU, not TPU-8. |
| Train and cosine steps | 50,000 → 2,000 | 25,000 → 2,000 | Enforce the reviewed short experiment. |
| Base LR | 0.08 → 0.005 | 0.04 → 0.00125 | Linear batch scaling from the reference global batch. |
| Warmup LR | 0.026666 → 0.001666667 | 0.013333 → 0.000416667 | Approximately one-third of the scaled base LR. |
| Warmup steps | 1,000 → 100 | 2,000 → 100 | Use 5% of the shortened training schedule. |
| bfloat16 | Explicit false | true → false | Avoid TPU-only precision on the T4. |
| Train/val label and record paths | `/opt/ml/input/data/{train,val}/...` | Same | Match the starter's SageMaker data channels. |
| Eval batch | Explicit 1 | Already 1 | Keep validation memory predictable. |

Batch 8 vs 2 means 16,000 vs 4,000 requested image presentations at 2,000
steps: this is **not an equal-exposure or equal-compute comparison**. See
`results.md`; no measured metrics have been populated.

### Sparse-label verification

The inspected local TensorFlow Models checkout was clean at revision
`5f846b08bedd4866ee5da6f93a2c07d521cf2e02`; image preparation fetches that same
revision rather than unpinned master.

- `research/object_detection/inputs.py:48`: `_LABEL_OFFSET = 1`.
- `inputs.py:355–363`: subtracts that offset, then calls
  `tf.one_hot(zero_indexed_groundtruth_classes, num_classes)`.
- IDs `[1,2,4]` become indices `[0,1,3]`; depth 4 preserves each label.
  Depth 3 would encode cyclist index 3 as all zeros.
- `research/object_detection/exporter_lib_v2.py:124–129` adds one back to
  detection classes. Cyclist remains output ID 4.
- ID 3 is an unused output slot, **not a fourth labeled semantic category**.
  COCO evaluation uses the public label-map categories; the deployment
  `category_index` stays `{1,2,4}`.

### Notebook hardware, Docker disk, and costs

Nothing in the new notebooks requires a **local GPU**. The m5.xlarge CPU
Notebook Instance builds the GPU-capable image and controls remote jobs.
Local Docker preflight uses CPU only; training runs remotely on g4dn.xlarge.

The configured **30 GB persistent volume** backs `/home/ec2-user/SageMaker`.
Docker's data root was moved to `/home/ec2-user/SageMaker/docker`; re-apply that
relocation after every Notebook Instance restart before building or pulling images.
The disk cell queries `docker info --format '{{.DockerRootDir}}'` to check the
actual location rather than assuming the default `/var/lib/docker`.
The preflight requires 20 GiB free on Docker's filesystem, 3 GiB for source/
checkpoint files, and 2 GiB for temporary files, summed for shared filesystems.
A fresh 30 GB volume might hold the project files, but **we cannot certify the
Docker build fits without this notebook-side check and a real build**. Its 20 GiB
Docker requirement estimates build headroom only; it does not account for an
already-built image occupying the Docker data root.
There is no automatic cleanup, resizing, daemon restart, or Docker relocation.
Image setup disables pip caches and limits the Docker build context.
The original Dockerfile/build script remain as references; use `Dockerfile.phase2`.

AWS explains the separate Docker root storage here:
https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html

| Resource | On-demand rate, us-east-1 | Planning envelope |
|---|---|---|
| MobileNet job, batch 8 | $0.736/hour | Estimated 1–2 hours ($0.74–$1.47), max_run 7,200 seconds |
| Faster R-CNN job, batch 2 | $0.736/hour | Estimated 1.5–3 hours ($1.10–$2.21), max_run 14,400 seconds |
| m5.xlarge notebook | $0.23/hour | 8 active hours: $1.84 |
| g5.xlarge notebook fallback | $1.41/hour | **4**, not 8, active hours: $5.64 |
| Future endpoint | $0.736/hour | At most 1 hour: $0.736 |

Rates come from the AWS public SageMaker us-east-1 catalog published
2026-10-16: https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonSageMaker/current/us-east-1/index.json
Runtime values are estimates, not measurements. At both training runtime limits,
plus one endpoint hour, $1 storage/log allowance and $3 contingency, the planned
envelope is **$10.99 with m5 / $14.79 with the four-hour g5 fallback**.
This is not a hard billing cap. The 30 GB volume continues charging while stopped;
storage allowance assumes a short project window. No optional baseline or
augmentation is approved within the g5 fallback envelope.

You create the bucket `obj-detection-proj` in us-east-1 before running these
notebooks. S3 bucket names are globally unique: if unavailable, stop and review
one consistent generic replacement rather than silently using another bucket.
Artifacts, TensorBoard events, source code and resumable checkpoints use distinct
generic prefixes in this bucket. The notebooks only check it exists/in-region.
Checkpoint weights reach jobs inside each isolated uploaded source bundle, the
same mechanism as the EfficientDet starter.

### Validation without paid jobs

Run `python3 -m unittest discover -s tests -v` locally for fake-client guard,
notebook, artifact-provenance, disk, and shell-failure checks. Protobuf parsing
can additionally be enabled by setting `OD_PROTO_PATH` to generated Model Garden
protos. Container smoke tests and CPU model/config checks are supplied but are
**not evidence of a successful image build or GPU run**.
TODO: Review image build output and the first job's memory/throughput logs before
changing batch sizes, retrying a failed job, or authorizing optional work.

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/). At the end of this project, you will be able to generate videos such as the one below. 

<p align="center">
    <img src="data/animation.gif" alt="drawing" width="600"/>
</p>

## Installation

Refer to the **Setup Instructions** page in the classroom to setup the Sagemaker Notebook instance required for this project.

>Note: The `conda_tensorflow2_p310` kernel contains most of the required packages for this project. The notebooks contain lines for manual installation when required.

## Usage

This repository contains two notebooks:
* [1_train_model](1_model_training/1_train_model.ipynb): this notebook is used to launch a training job and create tensorboard visualizations. 
* [2_deploy_model](2_run_inference/2_deploy_model.ipynb): this notebook is used to deploy your model, run inference on test data and create a gif similar to the one above.

First, run `1_train_model.ipynb` to train your model. Once the training job is complete, run `2_deploy_model.ipynb` to deploy your model and generate the animation.

Each notebook contains the instructions for running the code, as well the requirements for the writeup. 
>Note: Only the first notebook requires a write up. 

## Useful links
* The Tensorflow Object Detection API tutorial is a great resource to debug your code. This [section](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) in particular will teach you how to edit the `pipeline.config` file to update
your training job.

* [This blog post](https://aws.amazon.com/blogs/machine-learning/training-and-deploying-models-using-tensorflow-2-with-the-object-detection-api-on-amazon-sagemaker/) teaches how to label data, train and deploy a model with the Tensorflow Object Detection API and AWS Sagemaker.
