#!/usr/bin/env python3
"""Generate reviewable Phase 2 assets from authentic, pinned Model Zoo references."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "1_model_training"
sys.path.insert(0, str(TRAINING))
from deployment_phase6 import configure_deployment_notebook
from phase2 import EXPERIMENTS, MODEL_GARDEN_REVISION, experiment


def cell(kind, text):
    result = {
        "cell_type": kind, "metadata": {},
        "id": hashlib.sha256((kind + text.strip()).encode()).hexdigest()[:12],
        "source": text.strip().splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(spec):
    key = spec["key"]
    title = "SSD MobileNet V2 FPNLite" if key == "ssd_mobilenet" else "Faster R-CNN ResNet50 V1"
    cells = [
        cell("markdown", f"""# {title} — guarded sequential experiment
Use `conda_python3` on the supplied Amazon Linux 2023 **Notebook Instance**, not Studio.
Open this copy; the original EfficientDet notebook/config remain reference-only.
Public labels stay {{1,2,4}}; four output slots preserve cyclist ID 4.
Batch **{spec['batch_size']}**, **2,000 steps**, requested image exposures
**{spec['batch_size'] * 2000:,}**. Different batches make this an unequal-exposure comparison.
Nothing here needs a local GPU: the image build/smoke tests use CPU; only the
remote training job needs one T4. Run one notebook/controller at a time.

**Do not Run All.** Build and training switches default off and reset after use.
Do not proceed after a failed check. No AWS keys or tokens belong in cells.
The existing bucket `obj-detection-proj` must belong to this lab account and be in us-east-1.
"""),
        cell("code", """# Install only the controller SDK; do not replace the kernel's TensorFlow.
%pip install "sagemaker==2.243.2"
# TODO: Restart this kernel if the SDK version changed, then continue below.
"""),
        cell("code", f"""from pathlib import Path
import json
import os
import sys
import subprocess
import boto3
import sagemaker

# The Notebook Instance Git repository setting places the checkout here.
REPO = Path("/home/ec2-user/SageMaker/object-detection-urban-env")
TRAINING = REPO / "1_model_training"
if not (TRAINING / "phase2.py").is_file():
    raise RuntimeError("Expected Notebook Instance checkout is missing.")
os.chdir(TRAINING)
sys.path.insert(0, str(TRAINING))
from phase2 import (
    REGION, BUCKET, INPUTS, MODEL_GARDEN_REVISION, experiment, check_disk,
    prepare_model_garden, download_checkpoint, prepare_source, check_bucket,
    config_hash, launch_or_reuse, describe_job,
)
from framework import CustomFramework
from build_phase2 import build_or_reuse, image_tag
SPEC = experiment("{key}")
STATE_PATH = TRAINING / ".phase2/run_records.json"
"""),
        cell("markdown", """## Controller and region preflight — read-only AWS calls
The kernel name does not tell us the hardware. Confirm the instance type below
matches the lab console. m5.xlarge is $0.23/hour; g5.xlarge fallback is $1.41/hour.
Budget at most 8 active hours on m5 or **4 active hours on g5** across the project.
Stop the Notebook Instance between sessions; closing the browser is insufficient.
"""),
        cell("code", """NOTEBOOK_INSTANCE_TYPE = "ml.m5.xlarge"  # TODO: Set to ml.g5.xlarge only if necessary.
REMAINING_PROJECT_BUDGET_USD = None  # TODO: Enter unspent portion of the $15 project cap.
NOTEBOOK_ACTIVE_HOURS_REMAINING = None  # TODO: Enter remaining active hours (max 8 m5 / 4 g5).
if NOTEBOOK_INSTANCE_TYPE not in ("ml.m5.xlarge", "ml.g5.xlarge"):
    raise RuntimeError("Unreviewed Notebook Instance type.")
configured_region = boto3.Session().region_name
if configured_region not in (None, REGION):
    raise RuntimeError("Wrong region: stop and use us-east-1.")
aws_session = boto3.Session(region_name=REGION)
sm = aws_session.client("sagemaker")
s3 = aws_session.client("s3")
check_bucket(s3)  # No bucket creation or permission changes.
role = sagemaker.get_execution_role()
sm_session = sagemaker.Session(boto_session=aws_session, default_bucket=BUCKET)
print("Region and existing bucket checks passed. No resources created.")
"""),
        cell("markdown", """## Pre-build disk-space check
The **30 GB** persistent volume holds `/home/ec2-user/SageMaker`. Docker normally
uses `/var/lib/docker` on the separate root filesystem, not that volume.
`docker info` below resolves the actual location (including any existing relocation).
Require 20 GiB free for Docker, 3 GiB workspace, and 2 GiB temporary headroom;
requirements are added together when paths share a filesystem.
This is a conservative heuristic, **not proof of the unbuilt image's peak size**.
If it fails, stop and report the output; do not automatically prune images, resize
volumes, restart Docker, or move its data directory.
"""),
        cell("code", """docker_data_root = check_disk(TRAINING)
"""),
        cell("markdown", """## Prepare pinned Object Detection API source
Download only; this does not build an image or call AWS. Existing modified
Model Garden checkouts are rejected rather than overwritten.
"""),
        cell("code", """model_garden = prepare_model_garden(TRAINING)
print("Pinned Model Garden revision:", MODEL_GARDEN_REVISION)
"""),
        cell("markdown", """## COST GUARD — optional image build and ECR push
No training job or endpoint is created here. Build CPU time is billed on the
already-running Notebook Instance: reserve **0.5–1 hour**, approximately
**$0.12–$0.23 on m5.xlarge** or **$0.71–$1.41 on g5.xlarge**, plus ECR storage.
These are estimates, not measured build times. Reuse the same image for both models.
Build failure stops before ECR creation. Existing matching ECR tags are reused.
"""),
        cell("code", """ALLOW_IMAGE_BUILD = False  # TODO: Arm only after reviewing the disk check and budget.
"""),
        cell("code", """try:
    if ALLOW_IMAGE_BUILD:
        container = build_or_reuse(TRAINING, aws_session)
    else:
        print("Build disabled. Reuse .phase2/image.json if already prepared.")
finally:
    ALLOW_IMAGE_BUILD = False
"""),
        cell("markdown", """## Model Zoo checkpoint download
The archive URL is HTTPS. Only regular checkpoint files are extracted; path
traversal and links are not extracted. Completed extractions are reused.
Like the original EfficientDet workflow, the checkpoint is included in an
isolated `source_dir` bundle; `fit()` uploads that bundle to our explicit S3
code prefix. **No separate checkpoint S3 upload or training job occurs here.**
"""),
        cell("code", """checkpoint_dir = download_checkpoint(SPEC, TRAINING)
source_dir = prepare_source(SPEC, TRAINING)
print("Prepared one model's source and checkpoint:", source_dir)
"""),
        cell("markdown", """## Validate the built image/config locally — CPU only
This runs an existing local Docker image without GPU flags, network, or AWS access.
It checks API imports, protobuf parsing, label one-hot slots, model construction,
and checkpoint-index readability through TensorFlow's checkpoint reader. It does not train
or establish a maximum GPU batch size. If the image is missing after a notebook
restart, re-arm image preparation to reuse its ECR tag, then explicitly pull it.
"""),
        cell("code", """PREFLIGHT_CONFIG_SHA256 = None
image_record = json.loads((TRAINING / ".phase2/image.json").read_text())
if image_record["model_garden_revision"] != MODEL_GARDEN_REVISION:
    raise RuntimeError("Stale image record.")
if image_record["tag"] != image_tag(TRAINING):
    raise RuntimeError("Image build inputs changed; prepare the reviewed image again.")
container = image_record["image_uri"]
subprocess.run(["docker", "image", "inspect", container], check=True, stdout=subprocess.DEVNULL)
subprocess.run([
    "docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
    "-e", "CUDA_VISIBLE_DEVICES=", "-v", f"{source_dir}:/opt/ml/code:ro",
    "-w", "/opt/ml/code", container, "validate_pipeline.py", SPEC["config"],
], check=True)
PREFLIGHT_CONFIG_SHA256 = config_hash(SPEC, source_dir)
"""),
        cell("markdown", """## Arm exactly one training submission
Existing completed/active jobs are inspected, not relaunched. Failed/stopped jobs
are not automatically retried. Permission/credential/network errors fail closed.
An account-wide active-job check and a shared local submission journal enforce
sequential use by this controller. They are not a distributed lock: do not launch
from other consoles/controllers. Preserve `.phase2/run_records.json` for backup.
An ambiguous submission error requires manual reconciliation, not deleting the journal.
"""),
        cell("code", """ALLOW_NEW_JOB = False  # TODO: Set True only for the first reviewed submission.
"""),
        cell("markdown", f"""## COST GUARD — {spec['job_name']}
**ONE ml.g4dn.xlarge**, on-demand us-east-1, **$0.736/hour**.
**2,000 steps**, batch **{spec['batch_size']}**.
Estimated full job **{spec['expected_hours'][0]}–{spec['expected_hours'][1]} hours**,
approximately **${spec['expected_hours'][0] * .736:.2f}–${spec['expected_hours'][1] * .736:.2f}**.
`max_run={spec['max_run']}` limits algorithm runtime to {spec['max_run'] // 3600} hours
(**${spec['max_run'] / 3600 * .736:.3f}** at the instance rate).
Provisioning/billing overhead, storage and notebook compute are additional;
this is not an all-inclusive dollar cap. A timeout may prevent export.
Training → full validation → export happen **sequentially in this same job**.
No Spot, tuning, retries, warm pools, managed TensorBoard, or extra Debugger jobs.
Checkpoint synchronization still writes checkpoints and TensorBoard event files to
`s3://obj-detection-proj/checkpoints/<job>/`; no TensorBoardOutputConfig is used.
The original tiny-batch runtime estimates are not measurements for these batches.
"""),
        cell("code", """def training_preflight():
    if PREFLIGHT_CONFIG_SHA256 != config_hash(SPEC, source_dir):
        raise RuntimeError("Run the local image/config validation cell successfully first.")
    if not isinstance(REMAINING_PROJECT_BUDGET_USD, (int, float)):
        raise RuntimeError("Enter the remaining project budget before arming a job.")
    max_hours = 8 if NOTEBOOK_INSTANCE_TYPE == "ml.m5.xlarge" else 4
    if not isinstance(NOTEBOOK_ACTIVE_HOURS_REMAINING, (int, float)) or not 0 < NOTEBOOK_ACTIVE_HOURS_REMAINING <= max_hours:
        raise RuntimeError("Enter remaining active notebook hours within the reviewed limit.")
    notebook_rate = .23 if NOTEBOOK_INSTANCE_TYPE == "ml.m5.xlarge" else 1.41
    notebook_reserve = notebook_rate * NOTEBOOK_ACTIVE_HOURS_REMAINING
    other_training_reserve = 2.944 if SPEC["key"] == "ssd_mobilenet" else 0
    reserve = SPEC["max_run"] / 3600 * .736 + other_training_reserve + notebook_reserve + .736 + 1 + 3
    if not 0 < REMAINING_PROJECT_BUDGET_USD <= 15 or REMAINING_PROJECT_BUDGET_USD < reserve:
        raise RuntimeError(f"Conservative remaining budget required: ${reserve:.2f}.")
    check_bucket(s3)

def submit_training(job_name):
    # This closure is only called after existing/concurrent-job and budget guards.
    digest = config_hash(SPEC, source_dir)
    estimator = CustomFramework(
        role=role, image_uri=container, entry_point="run_training.sh",
        source_dir=str(source_dir), sagemaker_session=sm_session,
        code_location=f"s3://{BUCKET}/code/{SPEC['job_name']}",
        output_path=SPEC["output_path"],
        hyperparameters={
            "model_dir": "/opt/ml/checkpoints",
            "pipeline_config_path": SPEC["config"],
            "num_train_steps": "2000",
            "sample_1_of_n_eval_examples": "1",
            "batch_size": str(SPEC["batch_size"]),
            "config_sha256": digest,
            "model_garden_revision": MODEL_GARDEN_REVISION,
        },
        instance_count=1, instance_type="ml.g4dn.xlarge",
        max_run=SPEC["max_run"],
        use_spot_instances=False,
        checkpoint_s3_uri=f"s3://{BUCKET}/checkpoints/{SPEC['job_name']}",
        checkpoint_local_path="/opt/ml/checkpoints",
        debugger_hook_config=False, disable_profiler=True,
        base_job_name=SPEC["job_name"], tags=[],
    )
    estimator.fit(INPUTS, job_name=job_name, wait=False, logs=False)

try:
    job = launch_or_reuse(
        sm, SPEC, source_dir, container, STATE_PATH,
        submit_training, allow_new=ALLOW_NEW_JOB, preflight=training_preflight,
    )
finally:
    ALLOW_NEW_JOB = False  # Even a timeout/expired session consumes the switch.
"""),
        cell("markdown", """## Inspect results without launching anything
After a restart, the status cell below needs only the controller/setup cells,
not Docker, a checkpoint download, or a launch switch. Re-run the guarded launch
cell separately if you want to refresh the durable journal. A Completed job's artifact
path comes from `DescribeTrainingJob`, never from an invented URI.
Events/checkpoints live under this experiment's S3 prefixes. Evaluation is after
training, so there may be only one validation point—not a validation curve.
`BillableTimeInSeconds × $0.736 / 3600` is training compute only, not total project cost.
TODO: Provide the saved journal plus logs/events in Phase 3. Do not fill metrics
in results.md until those sources exist. Stop the Notebook Instance when finished.
"""),
        cell("code", """job = describe_job(sm, SPEC["job_name"])  # Read-only; never submits.
if job is not None:
    print("Status:", job["TrainingJobStatus"])
    if job["TrainingJobStatus"] == "Completed":
        print("Artifact:", job["ModelArtifacts"]["S3ModelArtifacts"])
    print("Run record:", STATE_PATH)
    print("Training seconds:", job.get("TrainingTimeInSeconds"))
    print("Billable seconds:", job.get("BillableTimeInSeconds"))
else:
    print("This experiment has not been submitted.")
"""),
    ]
    # Derive from the starter metadata, but never carry starter outputs forward.
    starter = json.loads((TRAINING / "1_train_model.ipynb").read_text())
    return dict(starter, cells=cells, metadata={
        "kernelspec": {"display_name": "conda_python3",
                       "language": "python", "name": "conda_python3"},
        "language_info": {"name": "python", "version": "3.10"},
    })


def adapted_config(original, key):
    spec = experiment(key)
    text = original.replace("num_classes: 90", "num_classes: 4")
    text = text.replace('fine_tune_checkpoint_type: "classification"',
                        'fine_tune_checkpoint_type: "detection"')
    import re
    text = re.sub(r'fine_tune_checkpoint: "[^"]+"',
                  'fine_tune_checkpoint: "checkpoint/ckpt-0"', text)
    text = re.sub(r"batch_size: (128|64)\b", f"batch_size: {spec['batch_size']}", text)
    text = text.replace("sync_replicas: true", "sync_replicas: false")
    text = text.replace("replicas_to_aggregate: 8", "replicas_to_aggregate: 1")
    text = re.sub(r"(num_steps|total_steps): (50000|25000)", r"\1: 2000", text)
    base_lr = "0.005" if key == "ssd_mobilenet" else "0.00125"
    warmup_lr = "0.001666667" if key == "ssd_mobilenet" else "0.000416667"
    text = re.sub(r"learning_rate_base: \.\d+", f"learning_rate_base: {base_lr}", text)
    text = re.sub(r"warmup_learning_rate: \.\d+", f"warmup_learning_rate: {warmup_lr}", text)
    text = re.sub(r"warmup_steps: (1000|2000)", "warmup_steps: 100", text)
    if "use_bfloat16:" in text:
        text = text.replace("use_bfloat16: true  # works only on TPUs", "use_bfloat16: false")
    else:
        text = text.replace("  unpad_groundtruth_tensors: false",
                            "  unpad_groundtruth_tensors: false\n  use_bfloat16: false")
    text = text.replace('"PATH_TO_BE_CONFIGURED/label_map.txt"',
                        '"/opt/ml/input/data/train/label_map.pbtxt"', 1)
    text = text.replace('"PATH_TO_BE_CONFIGURED/label_map.txt"',
                        '"/opt/ml/input/data/val/label_map.pbtxt"', 1)
    text = text.replace("PATH_TO_BE_CONFIGURED/train2017-?????-of-00256.tfrecord",
                        "/opt/ml/input/data/train/*.tfrecord")
    text = text.replace("PATH_TO_BE_CONFIGURED/val2017-?????-of-00032.tfrecord",
                        "/opt/ml/input/data/val/*.tfrecord")
    if key == "ssd_mobilenet":
        text = text.replace('  metrics_set: "coco_detection_metrics"',
                            '  metrics_set: "coco_detection_metrics"\n  batch_size: 1')
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-garden", type=Path)
    args = parser.parse_args()
    reference_dir = TRAINING / "model_zoo"
    reference_dir.mkdir(exist_ok=True)
    if args.model_garden:
        license_text = (args.model_garden / "LICENSE").read_bytes()
    else:
        license_url = (
            f"https://raw.githubusercontent.com/tensorflow/models/{MODEL_GARDEN_REVISION}/LICENSE"
        )
        with urllib.request.urlopen(license_url, timeout=30) as response:
            license_text = response.read()
    (reference_dir / "LICENSE.tensorflow").write_bytes(license_text)
    provenance = {}
    for key, values in EXPERIMENTS.items():
        relative = f"research/object_detection/configs/tf2/{values['model']}.config"
        url = f"https://raw.githubusercontent.com/tensorflow/models/{MODEL_GARDEN_REVISION}/{relative}"
        if args.model_garden:
            original = (args.model_garden / relative).read_bytes()
        else:
            with urllib.request.urlopen(url, timeout=30) as response:
                original = response.read()
        reference = reference_dir / (values["model"] + ".config")
        reference.write_bytes(original)
        adapted = adapted_config(original.decode(), key)
        target = TRAINING / "source_dir" / values["config"]
        target.write_text(adapted)
        difference = "".join(difflib.unified_diff(
            original.decode().splitlines(keepends=True),
            adapted.splitlines(keepends=True),
            fromfile=f"model_zoo/{reference.name}",
            tofile=f"source_dir/{target.name}",
        ))
        (reference_dir / f"{key}.diff").write_text(difference)
        provenance[key] = {"source_url": url, "revision": MODEL_GARDEN_REVISION,
                           "sha256": hashlib.sha256(original).hexdigest(),
                           "kind": "Model Zoo reference training config; not a project result"}
        output = TRAINING / f"1_train_model_{key}.ipynb"
        output.write_text(json.dumps(notebook(experiment(key)), indent=1) + "\n")
    (reference_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    deployment = ROOT / "2_run_inference/2_deploy_model.ipynb"
    starter_deployment = json.loads(subprocess.check_output(
        ["git", "-C", str(ROOT), "show", "ab8008a:2_run_inference/2_deploy_model.ipynb"]
    ))
    deployment.write_text(json.dumps(configure_deployment_notebook(starter_deployment), indent=1) + "\n")


if __name__ == "__main__":
    main()
