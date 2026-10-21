"""Build the guarded Phase 6 deployment notebook from the starter notebook."""

import hashlib
import textwrap


def _source(text):
    return textwrap.dedent(text).strip().splitlines(keepends=True)


def _cell(kind, text):
    item = {
        "cell_type": kind,
        "metadata": {},
        "id": hashlib.sha256((kind + text.strip()).encode()).hexdigest()[:12],
        "source": _source(text),
    }
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def _replace_code(cells, marker, text):
    for cell in cells:
        if cell["cell_type"] == "code" and marker in "".join(cell["source"]):
            cell["source"] = _source(text)
            return
    raise RuntimeError(f"Deployment starter cell not found: {marker}")


def _replace_markdown(cells, marker, text):
    for cell in cells:
        if cell["cell_type"] == "markdown" and marker in "".join(cell["source"]):
            cell["source"] = _source(text)
            return
    raise RuntimeError(f"Deployment starter cell not found: {marker}")


def configure_deployment_notebook(document):
    """Replace unsafe starter deployment cells without changing category IDs."""
    cells = document["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _replace_code(cells, "%pip install", """
        %%capture
        %pip install "sagemaker==2.243.2"
    """)

    _replace_code(cells, "import os", """
        from pathlib import Path
        import glob
        import os

        import boto3
        from botocore.exceptions import ClientError
        import cv2
        import numpy as np
        import sagemaker
        from sagemaker.deserializers import JSONDeserializer
        from sagemaker.predictor import Predictor
        from sagemaker.serializers import JSONSerializer
        from sagemaker.tensorflow import TensorFlowModel
        %matplotlib inline
        import matplotlib.pyplot as plt
        import visualization_utils as viz_utils
    """)
    _replace_code(cells, "role = sagemaker.get_execution_role()", """
        REGION = "us-east-1"
        ENDPOINT_NAME = "od-faster-rcnn-endpoint"
        MODEL_NAME = "od-faster-rcnn-model"
        ENDPOINT_CONFIG_NAME = "od-faster-rcnn-endpoint-config"
        ENDPOINT_INSTANCE_TYPE = "ml.g4dn.xlarge"
        TENSORFLOW_SERVING_VERSION = "2.13"
        ALLOW_ENDPOINT = False  # TODO: Set True only after reviewing the one-hour cost limit.

        aws_session = boto3.Session(region_name=REGION)
        if boto3.Session().region_name not in (None, REGION):
            raise RuntimeError("Wrong region: stop and use us-east-1.")
        sm = aws_session.client("sagemaker")
        s3 = aws_session.client("s3")
        role = sagemaker.get_execution_role()
        sm_session = sagemaker.Session(boto_session=aws_session)
    """)
    _replace_code(cells, "model_artifact =", """
        # TODO: Do not change this reviewed Faster R-CNN artifact without repeating the checks below.
        TRAINING_JOB_NAME = "od-faster-rcnn-01"
        model_artifact = "s3://obj-detection-proj/artifacts/od-faster-rcnn-01/od-faster-rcnn-01/output/model.tar.gz"
    """)
    validation_source = """
        from urllib.parse import urlparse

        def validate_model_artifact():
            \"\"\"Read-only provenance and availability checks before endpoint creation.\"\"\"
            training_job = sm.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)
            expected_artifact = training_job["ModelArtifacts"]["S3ModelArtifacts"]
            if model_artifact != expected_artifact:
                raise RuntimeError(
                    "Configured model_artifact does not match "
                    f"{TRAINING_JOB_NAME}'s S3ModelArtifacts: {expected_artifact}"
                )
            location = urlparse(model_artifact)
            if location.scheme != "s3" or not location.netloc or not location.path.lstrip("/"):
                raise RuntimeError(f"Invalid S3 model artifact URI: {model_artifact}")
            s3.head_object(Bucket=location.netloc, Key=location.path.lstrip("/"))
            print("Training artifact provenance and S3 object checks passed. No resources created.")
    """
    if any("def validate_model_artifact" in "".join(cell.get("source", [])) for cell in cells):
        _replace_code(cells, "def validate_model_artifact", validation_source)
    else:
        artifact_index = next(
            index for index, cell in enumerate(cells)
            if cell["cell_type"] == "code" and "model_artifact =" in "".join(cell["source"])
        )
        cells.insert(artifact_index + 1, _cell("code", validation_source))
    cost_guard = """
        ## COST GUARD — one endpoint, under one hour
        This notebook can create exactly one `ml.g4dn.xlarge` endpoint in `us-east-1` at
        **$0.736/hour**. Keep the run below **one hour** (endpoint estimate: **under
        $0.736**, plus Notebook Instance time and storage). It refuses to reuse an endpoint,
        endpoint configuration, or model with this notebook's fixed generic names.

        TensorFlow Serving **2.13** is selected because `sagemaker==2.243.2` supports its
        `2.13.0` inference image in `us-east-1`, and it matches this SavedModel's TensorFlow
        2.13 export. The deployment switch defaults off and resets after every attempt.
    """
    if any("COST GUARD" in "".join(cell.get("source", [])) for cell in cells):
        _replace_markdown(cells, "COST GUARD", cost_guard)
    else:
        deployment_index = next(
            index for index, cell in enumerate(cells)
            if cell["cell_type"] == "code" and "model.deploy(" in "".join(cell["source"])
        )
        cells.insert(deployment_index, _cell("markdown", cost_guard))
    _replace_code(cells, "model = TensorFlowModel(", """
        def _is_missing(error):
            return error.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound")

        def _assert_absent(describe, resource_name, argument_name):
            try:
                describe(**{argument_name: resource_name})
            except ClientError as error:
                if _is_missing(error):
                    return
                raise
            raise RuntimeError(f"Refusing to create: {resource_name} already exists.")

        def endpoint_preflight():
            validate_model_artifact()
            if aws_session.region_name != REGION:
                raise RuntimeError("Only us-east-1 is allowed.")
            _assert_absent(sm.describe_endpoint, ENDPOINT_NAME, "EndpointName")
            _assert_absent(sm.describe_endpoint_config, ENDPOINT_CONFIG_NAME, "EndpointConfigName")
            _assert_absent(sm.describe_model, MODEL_NAME, "ModelName")

        def cleanup_notebook_resources():
            \"\"\"Delete only this notebook's fixed endpoint, config, and model names.\"\"\"
            try:
                sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
            except ClientError as error:
                if not _is_missing(error):
                    raise
            else:
                sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
                sm.get_waiter("endpoint_deleted").wait(EndpointName=ENDPOINT_NAME)
            for describe, delete, name, argument in (
                (sm.describe_endpoint_config, sm.delete_endpoint_config,
                 ENDPOINT_CONFIG_NAME, "EndpointConfigName"),
                (sm.describe_model, sm.delete_model, MODEL_NAME, "ModelName"),
            ):
                try:
                    describe(**{argument: name})
                except ClientError as error:
                    if not _is_missing(error):
                        raise
                else:
                    delete(**{argument: name})
            print("Notebook deployment resources are deleted or were already absent.")
    """)
    deploy_index = next(
        index for index, cell in enumerate(cells)
        if cell["cell_type"] == "code" and "def endpoint_preflight" in "".join(cell["source"])
    )
    if not any("Standalone crash-recovery cleanup" in "".join(c.get("source", [])) for c in cells):
        cells.insert(deploy_index + 1, _cell("code", """
            # Standalone crash-recovery cleanup: run after a kernel interruption.
            cleanup_notebook_resources()
        """))
    _replace_code(cells, "frames_path = sorted", """
        frames_path = sorted(
            glob.glob("../data/test_video/*.png"),
            key=lambda path: int(os.path.basename(path).split(".")[0].split("_")[1]),
        )
        if not frames_path:
            raise RuntimeError("No test-video frames found.")
    """)
    _replace_code(cells, "def load_image", """
        def load_image(path):
            \"\"\"Read an image as an RGB uint8 array.\"\"\"
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read frame: {path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """)
    _replace_code(cells, "def image_file_to_tensor", """
        def annotate_frames(predictor):
            images = []
            for index, path in enumerate(frames_path):
                if index % 10 == 0:
                    print(f"Processed {index}/{len(frames_path)} images.")
                image = load_image(path)
                detections = predictor.predict({"instances": [image.tolist()]})["predictions"][0]
                annotated = viz_utils.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.array(detections["detection_boxes"]),
                    [int(value) for value in detections["detection_classes"]],
                    detections["detection_scores"],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=100,
                    min_score_thresh=0.6,
                    agnostic_mode=False,
                )
                images.append(annotated)
            return images

        def write_and_verify_video(images):
            output_video = Path("output.mp4")
            height, width = images[0].shape[:2]
            writer = cv2.VideoWriter(
                str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("Could not open output.mp4 for writing.")
            try:
                for image in images:
                    writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            for index in (0, len(images) // 2, len(images) - 1):
                cv2.imwrite(str(Path(f"annotated_frame_{index:03d}.png")),
                            cv2.cvtColor(images[index], cv2.COLOR_RGB2BGR))
            capture = cv2.VideoCapture(str(output_video))
            try:
                frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                capture.release()
            if (frame_count, decoded_width, decoded_height) != (len(images), width, height):
                raise RuntimeError(
                    "Video verification failed: "
                    f"got {frame_count} frames at {decoded_width}x{decoded_height}; "
                    f"expected {len(images)} at {width}x{height}."
                )
            print(f"Verified {output_video}: {frame_count} frames, {width}x{height}.")
    """)
    _replace_code(cells, "plt.imshow(images[0])", """
        def deploy_infer_and_cleanup():
            if not ALLOW_ENDPOINT:
                print("Endpoint deployment disabled. Review the cost guard, then explicitly arm it.")
                return None
            endpoint_preflight()
            try:
                model = TensorFlowModel(
                    name=MODEL_NAME,
                    model_data=model_artifact,
                    role=role,
                    framework_version=TENSORFLOW_SERVING_VERSION,
                    sagemaker_session=sm_session,
                )
                model.create(instance_type=ENDPOINT_INSTANCE_TYPE)
                sm.create_endpoint_config(
                    EndpointConfigName=ENDPOINT_CONFIG_NAME,
                    ProductionVariants=[{
                        "VariantName": "AllTraffic",
                        "ModelName": MODEL_NAME,
                        "InitialInstanceCount": 1,
                        "InstanceType": ENDPOINT_INSTANCE_TYPE,
                    }],
                )
                sm.create_endpoint(
                    EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_CONFIG_NAME
                )
                sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT_NAME)
                predictor = Predictor(
                    ENDPOINT_NAME, serializer=JSONSerializer(), deserializer=JSONDeserializer(),
                    sagemaker_session=sm_session,
                )
                images = annotate_frames(predictor)
                write_and_verify_video(images)
                return images
            finally:
                cleanup_notebook_resources()

        try:
            images = deploy_infer_and_cleanup()
        finally:
            ALLOW_ENDPOINT = False
    """)
    _replace_code(cells, "frame_width = images[0].shape[0]", """
        if images:
            plt.imshow(images[0])
            plt.axis("off")
    """)
    return document
