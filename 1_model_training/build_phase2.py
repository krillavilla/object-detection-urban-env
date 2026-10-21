"""Opt-in, student-run image build. Never import AWS SDKs or mutate on import."""

import hashlib
import os
from pathlib import Path
import subprocess

from phase2 import MODEL_GARDEN_REVISION, REGION, check_disk, error_code, write_json


REPOSITORY = "od-tf2-object-detection"


def image_tag(training_dir):
    digest = hashlib.sha256(MODEL_GARDEN_REVISION.encode())
    for name in ("Dockerfile.phase2", "constraints-phase2.txt", "smoke_phase2.py", ".dockerignore"):
        digest.update((Path(training_dir) / "docker" / name).read_bytes())
    return "tf213-" + digest.hexdigest()[:16]


def build_or_reuse(training_dir, aws_session):
    """Called only when the notebook explicitly arms its one-shot build switch."""
    if aws_session.region_name != REGION:
        raise RuntimeError("Only us-east-1 is allowed.")
    training_dir = Path(training_dir)
    target = training_dir / "docker/models"
    actual = subprocess.check_output(
        ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(target), "status", "--porcelain"], text=True
    ).strip()
    if actual != MODEL_GARDEN_REVISION or dirty:
        raise RuntimeError("Prepare the clean pinned Model Garden checkout first.")
    account = aws_session.client("sts").get_caller_identity()["Account"]
    ecr = aws_session.client("ecr")
    tag = image_tag(training_dir)
    image = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{REPOSITORY}:{tag}"
    exists = False
    try:
        ecr.describe_images(repositoryName=REPOSITORY, imageIds=[{"imageTag": tag}])
        exists = True
    except Exception as error:
        if error_code(error) not in ("RepositoryNotFoundException", "ImageNotFoundException"):
            raise  # Expired role credentials and permission errors stop the build.
    if not exists:
        check_disk(training_dir)
        subprocess.run(
            ["docker", "build", "-f", "Dockerfile.phase2", "-t", image, "."],
            cwd=training_dir / "docker", check=True,
        )
        # Build failures occur before ECR creation. Do not create on arbitrary errors.
        try:
            repository = ecr.describe_repositories(repositoryNames=[REPOSITORY])
            if repository["repositories"][0]["imageTagMutability"] != "IMMUTABLE":
                raise RuntimeError("ECR repository must use immutable image tags.")
        except Exception as error:
            if error_code(error) != "RepositoryNotFoundException":
                raise
            ecr.create_repository(
                repositoryName=REPOSITORY, imageTagMutability="IMMUTABLE",
                imageScanningConfiguration={"scanOnPush": False},
            )
        env = dict(os.environ, OD_ECR_REGISTRY=image.split("/")[0])
        # Keep ECR's temporary login password in a shell variable; never print it.
        # Role credentials remain in the normal provider chain, not notebook cells.
        login = (
            'ECR_PASSWORD=$(aws ecr get-login-password --region us-east-1) && '
            'printf "%s" "$ECR_PASSWORD" | docker login --username AWS '
            '--password-stdin "$OD_ECR_REGISTRY"; status=$?; '
            'unset ECR_PASSWORD; exit "$status"'
        )
        subprocess.run(["bash", "-c", login], env=env, check=True)
        subprocess.run(["docker", "push", image], check=True)
    else:
        repository = ecr.describe_repositories(repositoryNames=[REPOSITORY])
        if repository["repositories"][0]["imageTagMutability"] != "IMMUTABLE":
            raise RuntimeError("Existing ECR image tags must be immutable.")
        print("Reusing existing ECR image; no rebuild or push.")
    details = ecr.describe_images(
        repositoryName=REPOSITORY, imageIds=[{"imageTag": tag}]
    )["imageDetails"][0]
    write_json(training_dir / ".phase2/image.json", {
        "image_uri": image,
        "image_digest": details["imageDigest"],
        "model_garden_revision": MODEL_GARDEN_REVISION,
        "tag": tag,
    })
    return image
