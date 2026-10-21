"""Helpers for the student-run notebooks. Importing this module performs no I/O."""

import contextlib
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import urllib.request


REGION = "us-east-1"
BUCKET = "obj-detection-proj"
MODEL_GARDEN_REVISION = "5f846b08bedd4866ee5da6f93a2c07d521cf2e02"
TRAIN_RATE = 0.736
EXPERIMENTS = {
    "ssd_mobilenet": {
        "model": "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8",
        "config": "pipeline_ssd_mobilenet.config",
        "job_name": "od-ssd-mobilenet-01",
        "batch_size": 8,
        "max_run": 7200,
        "expected_hours": [1.0, 2.0],
    },
    "faster_rcnn": {
        "model": "faster_rcnn_resnet50_v1_640x640_coco17_tpu-8",
        "config": "pipeline_faster_rcnn.config",
        "job_name": "od-faster-rcnn-01",
        "batch_size": 2,
        "max_run": 14400,
        "expected_hours": [1.5, 3.0],
    },
}
INPUTS = {
    "train": "s3://cd2688-object-detection-tf2/train/",
    "val": "s3://cd2688-object-detection-tf2/val/",
}
TERMINAL = {"Completed", "Failed", "Stopped"}


def experiment(key):
    spec = dict(EXPERIMENTS[key], key=key, steps=2000, num_classes=4)
    spec["output_path"] = f"s3://{BUCKET}/artifacts/{spec['job_name']}/"
    spec["tensorboard_path"] = f"s3://{BUCKET}/tensorboard/{spec['job_name']}/"
    return spec


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def check_disk(workspace, min_docker_gib=20, min_workspace_gib=3, min_tmp_gib=2):
    """Check actual filesystems; do not resize, prune, or relocate Docker."""
    docker_root = Path(subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True
    ).strip()).resolve()
    locations = [
        ("Docker data root", docker_root, min_docker_gib),
        ("Repository/checkpoints", Path(workspace).resolve(), min_workspace_gib),
        ("Temporary files", Path("/tmp"), min_tmp_gib),
    ]
    devices = {}
    for label, path, required in locations:
        # Fail closed if Docker's reported path cannot be inspected by this user.
        stat = path.stat()
        usage = shutil.disk_usage(path)
        print(f"{label}: {path}; free={usage.free / 2**30:.1f} GiB; "
              f"total={usage.total / 2**30:.1f} GiB")
        group = devices.setdefault(stat.st_dev, {"required": 0, "free": usage.free})
        group["required"] += required * 2**30
    for group in devices.values():
        if group["free"] < group["required"]:
            raise RuntimeError(
                "Insufficient disk headroom: need 20 GiB for Docker, 3 GiB for "
                "workspace/checkpoints and 2 GiB temporary space, summed when "
                "they share a filesystem. Stop; do not prune or move Docker "
                "automatically. The 30 GB notebook volume is not the root disk."
            )
    print("Disk heuristic passed; actual unbuilt image peak size is not guaranteed.")
    return str(docker_root)


def prepare_model_garden(training_dir):
    training_dir = Path(training_dir)
    target = training_dir / "docker/models"
    if not target.exists():
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(
            ["git", "-C", str(target), "remote", "add", "origin",
             "https://github.com/tensorflow/models.git"], check=True
        )
    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if head.returncode or head.stdout.strip() != MODEL_GARDEN_REVISION:
        if subprocess.check_output(
            ["git", "-C", str(target), "status", "--porcelain"], text=True
        ).strip():
            raise RuntimeError("Model Garden has local changes; not overwriting them.")
        subprocess.run(
            ["git", "-C", str(target), "fetch", "--depth=1",
             "origin", MODEL_GARDEN_REVISION], check=True
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", "--detach", MODEL_GARDEN_REVISION],
            check=True,
        )
    if subprocess.check_output(
        ["git", "-C", str(target), "status", "--porcelain"], text=True
    ).strip():
        raise RuntimeError("Model Garden must be clean for a reproducible image.")
    return target


def download_checkpoint(spec, training_dir):
    """Download on the notebook instance; extract only regular checkpoint files."""
    workspace = Path(training_dir) / ".phase2" / spec["key"]
    checkpoint = workspace / "checkpoint"
    archive = workspace / (spec["model"] + ".tar.gz")
    marker = workspace / "checkpoint-provenance.json"
    workspace.mkdir(parents=True, exist_ok=True)
    if marker.is_file():
        manifest = json.loads(marker.read_text())
        if all((checkpoint / name).is_file() and
               (checkpoint / name).stat().st_size == size
               for name, size in manifest["files"].items()):
            return checkpoint
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        url = ("https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/"
               f"20200711/{spec['model']}.tar.gz")
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as out:
            shutil.copyfileobj(source, out)
        temporary.replace(archive)
    checkpoint.mkdir(exist_ok=True)
    prefix = f"{spec['model']}/checkpoint/"
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            if not member.name.startswith(prefix) or not member.isfile():
                continue
            name = member.name[len(prefix):]
            if "/" in name or name in ("", ".", ".."):
                raise RuntimeError("Unexpected checkpoint archive path.")
            if name == "checkpoint" or name == "ckpt-0.index" or name.startswith("ckpt-0.data-"):
                with contextlib.closing(package.extractfile(member)) as source:
                    with (checkpoint / name).open("wb") as out:
                        shutil.copyfileobj(source, out)
    if not (checkpoint / "ckpt-0.index").is_file() or not list(checkpoint.glob("ckpt-0.data-*")):
        raise RuntimeError("Incomplete checkpoint; do not train.")
    write_json(marker, {
        "model": spec["model"],
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {p.name: p.stat().st_size for p in checkpoint.iterdir() if p.is_file()},
    })
    return checkpoint


def prepare_source(spec, training_dir):
    """Upload one model only, not all pretrained weights in the repository."""
    training_dir = Path(training_dir)
    checkpoint = training_dir / ".phase2" / spec["key"] / "checkpoint"
    if not (checkpoint / "ckpt-0.index").exists():
        raise RuntimeError("Run the checkpoint download cell first.")
    source = training_dir / ".phase2" / spec["key"] / "source"
    source.mkdir(parents=True, exist_ok=True)
    garden = training_dir / "docker/models/research/object_detection"
    for name in ("model_main_tf2.py", "exporter_main_v2.py"):
        shutil.copy2(garden / name, source / name)
    for name in ("run_training.sh", "validate_pipeline.py", spec["config"]):
        shutil.copy2(training_dir / "source_dir" / name, source / name)
    shutil.copytree(checkpoint, source / "checkpoint", dirs_exist_ok=True)
    return source


def check_bucket(s3_client):
    """Read-only; never creates a bucket or changes its access policy."""
    s3_client.head_bucket(Bucket=BUCKET)
    location = s3_client.get_bucket_location(Bucket=BUCKET)["LocationConstraint"]
    if location not in (None, REGION):
        raise RuntimeError(f"{BUCKET} must be in {REGION}, not {location}.")


def error_code(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code")


def describe_job(client, name):
    try:
        return client.describe_training_job(TrainingJobName=name)
    except Exception as error:
        message = getattr(error, "response", {}).get("Error", {}).get("Message", "")
        if error_code(error) in ("ValidationException", "ResourceNotFound") and re.search(
            r"not found|could not find|does not exist", message, re.IGNORECASE
        ):
            return None
        # Expired credentials, AccessDenied, network errors: NEVER treat as missing.
        raise


def active_jobs(client):
    jobs = []
    for status in ("InProgress", "Stopping"):
        token = None
        while True:
            args = {"StatusEquals": status, "MaxResults": 100}
            if token:
                args["NextToken"] = token
            page = client.list_training_jobs(**args)
            jobs.extend(page["TrainingJobSummaries"])
            token = page.get("NextToken")
            if not token:
                break
    return jobs


def config_hash(spec, source):
    return hashlib.sha256((Path(source) / spec["config"]).read_bytes()).hexdigest()


def validate_existing(job, spec, digest, image):
    hp = job["HyperParameters"]
    expected = {
        "num_train_steps": str(spec["steps"]),
        "batch_size": str(spec["batch_size"]),
        "config_sha256": digest,
        "model_garden_revision": MODEL_GARDEN_REVISION,
    }
    # Framework SDK versions may JSON-quote string hyperparameters.
    normalized = {}
    for key, value in hp.items():
        try:
            normalized[key] = str(json.loads(value))
        except (ValueError, TypeError):
            normalized[key] = str(value)
    if any(normalized.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Existing job configuration differs; not submitting or reusing.")
    if job["AlgorithmSpecification"]["TrainingImage"] != image:
        raise RuntimeError("Existing job used a different image.")
    resource = job["ResourceConfig"]
    if resource["InstanceCount"] != 1 or resource["InstanceType"] != "ml.g4dn.xlarge":
        raise RuntimeError("Existing job used unexpected training resources.")
    if job["OutputDataConfig"]["S3OutputPath"].rstrip("/") != spec["output_path"].rstrip("/"):
        raise RuntimeError("Existing job has an unexpected output prefix.")
    if job["StoppingCondition"]["MaxRuntimeInSeconds"] != spec["max_run"]:
        raise RuntimeError("Existing job has a different runtime limit.")


def job_record(job, spec, digest):
    record = {
        "job_name": spec["job_name"],
        "model": spec["model"],
        "config_sha256": digest,
        "batch_size": spec["batch_size"],
        "requested_steps": spec["steps"],
        "requested_image_exposures": spec["steps"] * spec["batch_size"],
        "num_class_slots": 4,
        "label_ids": [1, 2, 4],
        "status": job["TrainingJobStatus"],
        "model_artifact": job.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
        "training_seconds": job.get("TrainingTimeInSeconds"),
        "billable_seconds": job.get("BillableTimeInSeconds"),
        "hourly_rate_usd": TRAIN_RATE,
    }
    billed = record["billable_seconds"]
    record["training_compute_cost_usd"] = None if billed is None else billed * TRAIN_RATE / 3600
    return record


def launch_or_reuse(
    client, spec, source, image, state_path, submit, allow_new=False, preflight=None
):
    """One-shot launch; local lock plus AWS status checks, not a distributed lock.

    Use only one notebook controller. Other consoles/users can race AWS's list API.
    Ambiguous submission errors leave durable intent and require manual review.
    """
    if client.meta.region_name != REGION:
        raise RuntimeError("Only us-east-1 is allowed.")
    if spec != experiment(spec["key"]):
        raise RuntimeError("Unreviewed experiment settings; update code/budget first.")
    if not re.fullmatch(
        r"\d{12}\.dkr\.ecr\.us-east-1\.amazonaws\.com/od-tf2-object-detection:[a-z0-9-]+",
        image,
    ):
        raise RuntimeError("Use the reviewed generic us-east-1 ECR training image.")
    digest = config_hash(spec, source)
    state_path = Path(state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.exists() else {"jobs": {}}
        name = spec["job_name"]
        existing = describe_job(client, name)
        if existing is not None:
            validate_existing(existing, spec, digest, image)
            state["jobs"][name] = job_record(existing, spec, digest)
            write_json(state_path, state)
            if existing["TrainingJobStatus"] in ("Failed", "Stopped"):
                raise RuntimeError(f"{name} is {existing['TrainingJobStatus']}; no automatic retry.")
            print(f"Reusing {name}: {existing['TrainingJobStatus']}; no new job.")
            return existing
        if name in state["jobs"]:
            raise RuntimeError("Submission intent exists but AWS job is missing. Reconcile manually.")
        if not allow_new:
            print("Launch disabled. Review costs, then explicitly arm one new job.")
            return None
        # Check other local intents too, covering list API eventual consistency.
        for other in state["jobs"]:
            job = describe_job(client, other)
            if job is None or job["TrainingJobStatus"] not in TERMINAL:
                raise RuntimeError(f"Another recorded job is unresolved or active: {other}.")
        if active_jobs(client):
            raise RuntimeError("An account training job is active/stopping; no parallel launch.")
        if preflight is not None:
            preflight()  # Known budget/config failures must not create submission intent.
        state["jobs"][name] = {
            "job_name": name, "status": "SubmissionIntent", "config_sha256": digest,
        }
        write_json(state_path, state)  # Must happen BEFORE calling fit().
        submit(name)
        # Do not infer success or resubmit after a timeout or transient error.
        created = describe_job(client, name)
        if created is not None:
            validate_existing(created, spec, digest, image)
            state["jobs"][name] = job_record(created, spec, digest)
            write_json(state_path, state)
        print(f"Submitted {name}; re-run this cell to inspect it, never to retry it.")
        return created
