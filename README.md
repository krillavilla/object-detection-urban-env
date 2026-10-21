# Object detection in an urban environment

## Public repository safety

This is a standalone coursework copy of the
[Udacity starter](https://github.com/udacity/cd2688-object-detection-in-urban-environment-project),
retaining its original history and license. Original training/deployment notebooks
and the EfficientNet pipeline configuration are unchanged during repository setup.

After cloning, enable the repository's credential checks (repeat in Studio):

```bash
git config --local core.hooksPath .githooks
python3 scripts/check_aws_secrets.py
```

The pre-commit hook checks all staged files, including saved notebook outputs.
The pre-push hook checks reachable Git history. Both reject likely AWS access-key
IDs and AWS secret/session credential field names without printing their values.
Hooks are local, can be bypassed, and pattern checks are not a guarantee: review
notebooks and diffs before committing. No server-side credential check is
configured. Never put AWS keys or tokens in this repository; Studio should use
its execution role.

The ignore rules exclude local credentials, environment files, TFRecords,
downloaded checkpoints, exported models, and video files. Existing starter images
and the example animation remain tracked. Keep generated videos and large model
artifacts outside Git and download them separately for backup.

Training and deployment launch paid AWS resources. Repository setup does not
launch any resources. **TODO:** Complete the read-only budget review before
running either notebook; use us-east-1, generic resource names, at most one
training job and one endpoint, and remove billable resources promptly.

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
