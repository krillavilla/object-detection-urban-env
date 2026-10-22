import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "1_model_training"
sys.path.insert(0, str(TRAINING))
import phase2

IMAGE = "000000000000.dkr.ecr.us-east-1.amazonaws.com/od-tf2-object-detection:tf213-test"


class AwsError(Exception):
    def __init__(self, code, message="Denied"):
        self.response = {"Error": {"Code": code, "Message": message}}


class FakeClient:
    meta = SimpleNamespace(region_name="us-east-1")

    def __init__(self):
        self.jobs = {}
        self.extra_active = []

    def describe_training_job(self, TrainingJobName):
        if TrainingJobName not in self.jobs:
            raise AwsError("ValidationException", "Requested resource not found")
        return copy.deepcopy(self.jobs[TrainingJobName])

    def list_training_jobs(self, **kwargs):
        return {"TrainingJobSummaries": [
            x for x in self.extra_active
            if x["TrainingJobStatus"] == kwargs["StatusEquals"]
        ]}


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.spec = phase2.experiment("ssd_mobilenet")
        self.source = self.folder / "source"
        self.source.mkdir()
        (self.source / self.spec["config"]).write_text("reviewed test configuration")
        self.state = self.folder / "run_records.json"
        self.client = FakeClient()
        self.submit = Mock(side_effect=self.create)

    def job(self, status="InProgress"):
        return {
            "TrainingJobStatus": status,
            "HyperParameters": {
                "num_train_steps": "2000", "batch_size": "8",
                "config_sha256": phase2.config_hash(self.spec, self.source),
                "model_garden_revision": phase2.MODEL_GARDEN_REVISION,
            },
            "AlgorithmSpecification": {"TrainingImage": IMAGE},
            "ResourceConfig": {"InstanceCount": 1, "InstanceType": "ml.g4dn.xlarge"},
            "OutputDataConfig": {"S3OutputPath": self.spec["output_path"]},
            "StoppingCondition": {"MaxRuntimeInSeconds": self.spec["max_run"]},
        }

    def create(self, name):
        record = json.loads(self.state.read_text())["jobs"][name]
        self.assertEqual(record["status"], "SubmissionIntent")
        self.client.jobs[name] = self.job()

    def launch(self, **kwargs):
        return phase2.launch_or_reuse(
            self.client, self.spec, self.source, IMAGE, self.state, self.submit, **kwargs
        )

    def test_default_does_not_launch(self):
        self.assertIsNone(self.launch())
        self.submit.assert_not_called()

    def test_first_launch_records_intent_before_submit_and_never_duplicates(self):
        self.launch(allow_new=True)
        self.launch(allow_new=True)
        self.submit.assert_called_once()

    def test_completed_job_without_local_state_never_relaunches(self):
        self.client.jobs[self.spec["job_name"]] = self.job("Completed")
        self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_failed_and_stopped_never_retry(self):
        for status in ("Failed", "Stopped"):
            self.client.jobs[self.spec["job_name"]] = self.job(status)
            with self.assertRaises(RuntimeError):
                self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_active_and_stopping_account_jobs_block(self):
        for status in ("InProgress", "Stopping"):
            self.client.extra_active = [{"TrainingJobName": "od-other-01",
                                         "TrainingJobStatus": status}]
            with self.assertRaises(RuntimeError):
                self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_paginated_active_jobs(self):
        self.client.list_training_jobs = Mock(side_effect=[
            {"TrainingJobSummaries": [], "NextToken": "next"},
            {"TrainingJobSummaries": [{"TrainingJobName": "od-other-01"}]},
            {"TrainingJobSummaries": []},
        ])
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_auth_and_network_errors_never_mean_missing(self):
        for error in (AwsError("ExpiredTokenException"), AwsError("AccessDenied"),
                      TimeoutError("Network timeout")):
            self.client.describe_training_job = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_ambiguous_submission_blocks_retry(self):
        self.submit.side_effect = TimeoutError("Ambiguous response")
        with self.assertRaises(TimeoutError):
            self.launch(allow_new=True)
        with self.assertRaisesRegex(RuntimeError, "intent"):
            self.launch(allow_new=True)
        self.submit.assert_called_once()

    def test_other_unresolved_intent_blocks(self):
        phase2.write_json(self.state, {"jobs": {"od-other-01": {"status": "SubmissionIntent"}}})
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_preflight_failure_does_not_record_intent(self):
        preflight = Mock(side_effect=RuntimeError("Insufficient budget"))
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True, preflight=preflight)
        self.assertFalse(self.state.exists())
        self.submit.assert_not_called()

    def test_config_mismatch_blocks_reuse(self):
        self.client.jobs[self.spec["job_name"]] = self.job()
        (self.source / self.spec["config"]).write_text("changed")
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_region_and_settings_changes_block(self):
        self.client.meta = SimpleNamespace(region_name="us-west-2")
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True)
        self.client.meta = SimpleNamespace(region_name="us-east-1")
        self.spec["max_run"] *= 2
        with self.assertRaises(RuntimeError):
            self.launch(allow_new=True)
        self.submit.assert_not_called()

    def test_sdk_json_quoted_hyperparameters(self):
        job = self.job("Completed")
        job["HyperParameters"] = {k: json.dumps(v) for k, v in job["HyperParameters"].items()}
        self.client.jobs[self.spec["job_name"]] = job
        self.launch()
        self.submit.assert_not_called()

    def test_actual_cost_requires_billable_seconds(self):
        digest = phase2.config_hash(self.spec, self.source)
        self.assertIsNone(phase2.job_record(self.job(), self.spec, digest)["training_compute_cost_usd"])
        job = self.job("Completed")
        job["BillableTimeInSeconds"] = 3600
        record = phase2.job_record(job, self.spec, digest)
        self.assertEqual(record["training_compute_cost_usd"], .736)
        self.assertEqual(record["requested_image_exposures"], 16000)


class AssetTests(unittest.TestCase):
    def test_original_training_notebook_and_config_unchanged(self):
        relative = "1_train_model.ipynb"
        original = json.loads(subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "ab8008a:1_model_training/" + relative]
        ))
        current = json.loads((TRAINING / relative).read_text())
        self.assertEqual(len(current["cells"]), len(original["cells"]))
        for index, (actual, expected) in enumerate(zip(current["cells"], original["cells"])):
            if index == 19:
                self.assertEqual(actual["cell_type"], "markdown")
                self.assertNotIn("# your write-up goes here.", "".join(actual["source"]))
            else:
                self.assertEqual(actual, expected, f"Unexpected change in notebook cell {index}.")
        relative = "source_dir/pipeline.config"
        expected = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", "ab8008a:1_model_training/" + relative]
        )
        self.assertEqual((TRAINING / relative).read_bytes(), expected)

    def test_deployment_notebook_has_guarded_lifecycle_and_video_output(self):
        notebook = json.loads((ROOT / "2_run_inference/2_deploy_model.ipynb").read_text())
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn(
            's3://obj-detection-proj/artifacts/od-faster-rcnn-01/'
            'od-faster-rcnn-01/output/model.tar.gz',
            source,
        )
        self.assertIn('sm.describe_training_job(TrainingJobName=TRAINING_JOB_NAME)', source)
        self.assertIn('s3.head_object(', source)
        self.assertIn('REGION = "us-east-1"', source)
        self.assertIn('ENDPOINT_NAME = "od-faster-rcnn-endpoint"', source)
        self.assertIn('ENDPOINT_INSTANCE_TYPE = "ml.g4dn.xlarge"', source)
        self.assertIn('TENSORFLOW_SERVING_VERSION = "2.13"', source)
        self.assertIn('%pip install "sagemaker==2.243.2"', source)
        self.assertIn("ALLOW_ENDPOINT = False", source)
        self.assertIn("def cleanup_notebook_resources():", source)
        self.assertIn('sm.delete_endpoint(EndpointName=ENDPOINT_NAME)', source)
        self.assertIn("sm.delete_endpoint_config", source)
        self.assertIn("sm.delete_model", source)
        self.assertIn("finally:\n        cleanup_notebook_resources()", source)
        self.assertIn('cv2.VideoWriter_fourcc(*"mp4v")', source)
        self.assertIn('Path("output.mp4")', source)
        self.assertIn("CAP_PROP_FRAME_COUNT", source)
        self.assertIn("annotated_frame_", source)
        self.assertIn("4: {'id': 4, 'name': 'cyclist'}", source)

    def test_executed_notebooks_retain_python_and_launch_guards(self):
        for key in phase2.EXPERIMENTS:
            path = TRAINING / f"1_train_model_{key}.ipynb"
            nb = json.loads(path.read_text())
            for index, c in enumerate(nb["cells"]):
                source = "".join(c["source"])
                if c["cell_type"] != "code":
                    continue
                if "estimator.fit(" in source:
                    self.assertIn("COST GUARD", "".join(nb["cells"][index - 1]["source"]))
                    self.assertIn("ALLOW_NEW_JOB = False", source)
                    self.assertIn("max_run=SPEC", source)
                    self.assertIn('output_path=SPEC["output_path"]', source)
                    self.assertNotIn("max_retry_attempts", source)
                    self.assertNotIn("tensorboard_output_config", source)
            self.assertEqual(nb["metadata"]["kernelspec"]["name"], "conda_python3")

    def test_model_zoo_provenance_and_diffs(self):
        folder = TRAINING / "model_zoo"
        provenance = json.loads((folder / "provenance.json").read_text())
        for key, spec in phase2.EXPERIMENTS.items():
            reference = folder / (spec["model"] + ".config")
            self.assertEqual(hashlib.sha256(reference.read_bytes()).hexdigest(),
                             provenance[key]["sha256"])
            self.assertIn(phase2.MODEL_GARDEN_REVISION, provenance[key]["source_url"])
            self.assertIn("+    num_classes: 4", (folder / f"{key}.diff").read_text())

    def test_budget_preflight_counts_only_future_notebook_hours(self):
        nb = json.loads((TRAINING / "1_train_model_ssd_mobilenet.ipynb").read_text())
        text = next("".join(c["source"]) for c in nb["cells"]
                    if c["cell_type"] == "code" and
                    "def training_preflight():" in "".join(c["source"]))
        function = ast.parse(text).body[0]
        namespace = {
            "PREFLIGHT_CONFIG_SHA256": "same",
            "config_hash": lambda *_: "same",
            "SPEC": phase2.experiment("ssd_mobilenet"),
            "source_dir": Path("."),
            "REMAINING_PROJECT_BUDGET_USD": 13.59,
            "NOTEBOOK_INSTANCE_TYPE": "ml.g5.xlarge",
            "NOTEBOOK_ACTIVE_HOURS_REMAINING": 3,
            "check_bucket": Mock(), "s3": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<budget-test>", "exec"), namespace)
        namespace["training_preflight"]()
        namespace["NOTEBOOK_ACTIVE_HOURS_REMAINING"] = 4
        with self.assertRaises(RuntimeError):
            namespace["training_preflight"]()
        namespace["NOTEBOOK_ACTIVE_HOURS_REMAINING"] = None
        with self.assertRaises(RuntimeError):
            namespace["training_preflight"]()
    def test_source_bundle_contains_only_selected_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            training = Path(directory)
            api = training / "docker/models/research/object_detection"
            api.mkdir(parents=True)
            for name in ("model_main_tf2.py", "exporter_main_v2.py"):
                (api / name).write_text("# synthetic test source\n")
            common = training / "source_dir"
            common.mkdir()
            spec = phase2.experiment("ssd_mobilenet")
            for name in ("run_training.sh", "validate_pipeline.py", spec["config"]):
                (common / name).write_text("synthetic test file\n")
            for key in phase2.EXPERIMENTS:
                checkpoint = training / ".phase2" / key / "checkpoint"
                checkpoint.mkdir(parents=True)
                (checkpoint / "ckpt-0.index").write_text(key)
                (checkpoint / "ckpt-0.data-00000-of-00001").write_text(key)
            source = phase2.prepare_source(spec, training)
            self.assertEqual((source / "checkpoint/ckpt-0.index").read_text(), "ssd_mobilenet")
            self.assertFalse((source / "pipeline_faster_rcnn.config").exists())
            self.assertFalse((source / "checkpoint_faster_rcnn").exists())

    def test_constraint_syntax(self):
        from packaging.requirements import Requirement
        for line in (TRAINING / "docker/constraints-phase2.txt").read_text().splitlines():
            if line and not line.startswith("#"):
                Requirement(line)

    @unittest.skipUnless(os.environ.get("OD_PROTO_PATH"), "Set OD_PROTO_PATH for generated Model Garden protos.")
    def test_real_protobuf_configs(self):
        sys.path.insert(0, os.environ["OD_PROTO_PATH"])
        from google.protobuf import text_format
        from object_detection.protos import pipeline_pb2
        for key in phase2.EXPERIMENTS:
            spec = phase2.experiment(key)
            config = pipeline_pb2.TrainEvalPipelineConfig()
            text_format.Parse((TRAINING / "source_dir" / spec["config"]).read_text(), config)
            kind = config.model.WhichOneof("model")
            self.assertEqual(getattr(config.model, kind).num_classes, 4)
            train = config.train_config
            self.assertEqual(train.num_steps, 2000)
            self.assertEqual(train.batch_size, spec["batch_size"])
            self.assertFalse(train.use_bfloat16)
            self.assertFalse(train.sync_replicas)
            self.assertEqual(train.replicas_to_aggregate, 1)
            self.assertEqual(train.fine_tune_checkpoint, "checkpoint/ckpt-0")
            self.assertEqual(train.fine_tune_checkpoint_type, "detection")
            lr = train.optimizer.momentum_optimizer.learning_rate.cosine_decay_learning_rate
            self.assertEqual(lr.total_steps, 2000)
            self.assertEqual(lr.warmup_steps, 100)
            self.assertEqual(config.eval_config.batch_size, 1)
            self.assertEqual(config.train_input_reader.label_map_path,
                             "/opt/ml/input/data/train/label_map.pbtxt")
            self.assertEqual(config.eval_input_reader[0].label_map_path,
                             "/opt/ml/input/data/val/label_map.pbtxt")

    def test_fail_fast_training_script_stops_before_evaluation_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mock_python = directory / "python"
            mock_python.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$1" >> "$OD_TEST_LOG"\n'
                'case "$1" in validate_pipeline.py) exit 0;; *) exit 17;; esac\n'
            )
            mock_python.chmod(0o755)
            mock_mkdir = directory / "mkdir"
            mock_mkdir.write_text("#!/bin/sh\nexit 0\n")
            mock_mkdir.chmod(0o755)
            log = directory / "calls"
            env = dict(os.environ, PATH=f"{directory}:" + os.environ["PATH"],
                       SM_HP_MODEL_DIR=str(directory), SM_HP_PIPELINE_CONFIG_PATH="pipeline.config",
                       SM_HP_NUM_TRAIN_STEPS="2000", SM_NUM_GPUS="1", OD_TEST_LOG=str(log))
            result = subprocess.run(
                ["bash", str(TRAINING / "source_dir/run_training.sh")],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 17)
            self.assertEqual(log.read_text().splitlines(), ["validate_pipeline.py", "model_main_tf2.py"])
            self.assertNotIn("=== EXPORT", result.stdout)

    def test_shared_disk_requirements_are_added(self):
        with tempfile.TemporaryDirectory() as directory:
            for free_gib, passes in ((26, True), (24, False)):
                with patch("phase2.subprocess.check_output", return_value=directory), patch(
                    "phase2.shutil.disk_usage",
                    return_value=SimpleNamespace(free=free_gib * 2**30, total=30 * 2**30),
                ):
                    if passes:
                        phase2.check_disk(directory)
                    else:
                        with self.assertRaises(RuntimeError):
                            phase2.check_disk(directory)

    def test_checkpoint_extraction_and_reentry_do_not_redownload(self):
        spec = phase2.experiment("ssd_mobilenet")
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name in ("ckpt-0.index", "ckpt-0.data-00000-of-00001"):
                data = b"synthetic checkpoint test data"
                member = tarfile.TarInfo(f"{spec['model']}/checkpoint/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        with tempfile.TemporaryDirectory() as directory:
            with patch("phase2.urllib.request.urlopen", return_value=io.BytesIO(stream.getvalue())) as download:
                first = phase2.download_checkpoint(spec, directory)
                second = phase2.download_checkpoint(spec, directory)
                self.assertEqual(first, second)
                download.assert_called_once()
                self.assertTrue((first / "ckpt-0.index").is_file())
                self.assertIn(
                    "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/tf2/",
                    download.call_args.args[0],
                )


if __name__ == "__main__":
    unittest.main()
