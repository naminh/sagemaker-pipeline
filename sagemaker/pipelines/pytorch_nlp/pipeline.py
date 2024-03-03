"""Pipeline Constructor module.

Module constructs a Sagemaker Pipeline with specified steps.

Typical usage example:

    sm_pipeline = PyTorchPipeline(
        role=role,
        lambda_role="",
        region="us-east-2",
        default_bucket="",
        project_name="pytorch-nlp"
    ).construct_pipeline()
    upsert_response = sm_pipeline.upsert(role_arn=role)
    execution = sm_pipeline.start(
        parameters=dict(
            ModelApprovalStatus="Approved",
            InferenceMethod="serverless",
            TrainData="s3://<>/tweet-sentiment-extraction/train.csv",
        ),
    )
"""

import os
import time

import boto3
import sagemaker

from dataclasses import dataclass
from typing import Tuple

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import (
    CacheConfig,
    ProcessingStep,
    TuningStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    ContinuousParameter,
    CategoricalParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.properties import PropertyFile

from sagemaker.pytorch.model import PyTorchModel
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.lambda_helper import Lambda


BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_sagemaker_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline_session(region, default_bucket):
    """Gets the sagemaker Pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )


def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


@dataclass
class PipelinesConfig:
    # TODO Later use this Config to init the Pipeline.
    pass


class PyTorchPipeline:
    """Sagemaker Pipeline constructor.

    Steps to involve:
        data_processing -> hyperparameter_tuner -> test_set_evaluation -> create_and_register_model -> deploy.

    Attributes:
        role: Sagemaker execution role.
        lambda_role: Lambda execution role.
        region: AWS region.
        default_bucket: Default S3 bucket for Sagemaker artefacts.
        project_name: Project name.
    """

    def __init__(
        self,
        role: str,
        lambda_role: str,
        region: str,
        default_bucket: str,
        project_name: str,
    ) -> None:
        self.region = region
        self.default_bucket = default_bucket
        self.project_name = project_name
        self.lambda_role = lambda_role

        if role is None:
            self.role = sagemaker.session.get_execution_role(self.sagemaker_session)
        else:
            self.role = role
        self.sagemaker_session = get_sagemaker_session(self.region, self.default_bucket)
        self.pipeline_session = get_pipeline_session(self.region, self.default_bucket)

        self.model_package_group_name = f"{self.project_name}-ModelPackageGroup"
        self.pipeline_name = f"{self.project_name}-Pipeline"
        self.model_prefix = f"{self.project_name}/model"
        self.model_path = f"s3://{self.default_bucket}/{self.model_prefix}"
        self.cache_config = CacheConfig(enable_caching=True, expire_after="30d")
        self.pipeline_parameters = {
            "processing_instance_count": ParameterInteger(
                name="ProcessingInstanceCount", default_value=1
            ),
            "instance_type": ParameterString(
                name="TrainingInstanceType", default_value="ml.m5.xlarge"
            ),
            "model_approval_status": ParameterString(
                name="ModelApprovalStatus",
                default_value="PendingManualApproval",  # "Approved"
            ),
            "train_data": ParameterString(
                name="TrainData",
                default_value=f"s3://{self.default_bucket}/tweet-sentiment-extraction/train.csv",
            ),
            "test_data": ParameterString(
                name="TestData",
                default_value=f"s3://{self.default_bucket}/tweet-sentiment-extraction/tune.csv",
            ),
            "inference_method": ParameterString(
                name="InferenceMethod",
                default_value="serverless",
            ),
            "pretrained_model_data_dir": ParameterString(
                name="PretrainedModelDataDir",
                default_value=f"s3://{self.default_bucket}/roberta_base/",
            ),
        }

    def get_data_processing_step(self) -> sagemaker.workflow.steps.ProcessingStep:
        """Create a data processing step.

        Load in csv data from S3 and process it for subsequent steps.

        Args:

        Returns:
            sagemaker.workflow.steps.ProcessingStep
        """

        sklearn_processor = SKLearnProcessor(
            framework_version="0.23-1",
            instance_type="ml.m5.large",
            instance_count=self.pipeline_parameters["processing_instance_count"],
            base_job_name=f"{self.project_name}-process",
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        processor_args = sklearn_processor.run(
            inputs=[
                ProcessingInput(
                    source=self.pipeline_parameters["train_data"],
                    destination="/opt/ml/processing/input",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train", source="/opt/ml/processing/train"
                ),
            ],
            code=os.path.join(BASE_DIR, "preprocessing.py"),
        )

        step_process = ProcessingStep(
            name=f"{self.project_name}-ProcesssingStep",
            step_args=processor_args,
            cache_config=self.cache_config,
        )

        return step_process

    def get_tuning_step(
        self, step_process: sagemaker.workflow.steps.ProcessingStep
    ) -> sagemaker.workflow.steps.TuningStep:
        """Construct Hyperparam Tuning Step.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep

        Args:
            step_process: The data processing step that prepares the training data.
        Returns:
            sagemaker.workflow.steps.TuningStep
        """
        # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html
        pt_estimator = PyTorch(
            entry_point="hp_tune.py",
            source_dir=BASE_DIR,
            role=self.role,
            py_version="py3",
            framework_version="1.8.1",
            instance_count=1,
            instance_type=self.pipeline_parameters["instance_type"],
            output_path=self.model_path,
        )

        pt_estimator.set_hyperparameters(
            epochs=1,
        )

        objective_metric_name = "average_test_jaccard"
        metric_definitions = [
            {
                "Name": "average_test_jaccard",
                "Regex": "Test Jaccard metric: ([0-9\\.]+)",
            }
        ]

        hyperparameter_ranges = {
            "learning_rate": ContinuousParameter(
                0.0001, 0.1, scaling_type="Logarithmic"
            ),
            "batch_size": CategoricalParameter([16, 32]),
        }

        # https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html
        hyper_tuner = HyperparameterTuner(
            pt_estimator,
            objective_metric_name,
            hyperparameter_ranges,
            metric_definitions=metric_definitions,
            max_jobs=3,
            max_parallel_jobs=3,
            strategy="Bayesian",
            objective_type="Minimize",
        )

        # https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep
        step_tuning = TuningStep(
            name=f"{self.project_name}-HPTuningStep",
            # step_args=hpo_args,
            tuner=hyper_tuner,
            inputs={
                "train": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "PRETRAINED_DATA_DIR": TrainingInput(
                    s3_data=self.pipeline_parameters["pretrained_model_data_dir"],
                ),
            },
            cache_config=self.cache_config,
        )

        return step_tuning

    def get_eval_step(
        self, step_tuning: sagemaker.workflow.steps.TuningStep
    ) -> sagemaker.workflow.steps.ProcessingStep:
        """Construct Processing Step for model evaluation on test data.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep

        Args:
            step_tuning: Hyperparam tuning step with saved models.
        Returns:
            sagemaker.workflow.steps.ProcessingStep
        """
        best_model_S3_artifact = step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=self.default_bucket, prefix=self.model_prefix
        )

        # Initialize the PyTorchProcessor
        # https://github.com/aws/sagemaker-python-sdk/blob/v2.117.0/src/sagemaker/pytorch/processing.py
        pytorch_processor = PyTorchProcessor(
            framework_version="1.8.1",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            base_job_name=f"{self.project_name}-eval",
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        # Run the processing job
        eval_args = pytorch_processor.run(
            code="evaluation.py",
            source_dir=BASE_DIR,
            inputs=[
                ProcessingInput(
                    input_name="ModelData",
                    source=best_model_S3_artifact,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    input_name="TestData",
                    source=self.pipeline_parameters["test_data"],
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation", source="/opt/ml/processing/evaluation"
                ),
            ],
        )

        evaluation_report = PropertyFile(
            name=f"{self.project_name}-TestPrediction",
            output_name="evaluation",
            path="pred.json",
        )

        step_eval = ProcessingStep(
            name=f"{self.project_name}-EvalStep",
            step_args=eval_args,
            property_files=[evaluation_report],
            cache_config=self.cache_config,
        )

        return step_eval

    def get_model_step(
        self,
        step_eval: sagemaker.workflow.steps.ProcessingStep,
        step_tuning: sagemaker.workflow.steps.TuningStep,
    ) -> Tuple[
        sagemaker.workflow.model_step.ModelStep, sagemaker.workflow.model_step.ModelStep
    ]:
        """Construct Model Step for model creation and registration.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.model_step.ModelStep

        Args:
            step_eval: Eval step with test metrics to be saved with model.
            step_tuning: Hyperparam tuning step with saved models.
        Returns:
            sagemaker.workflow.model_step.ModelStep, sagemaker.workflow.model_step.ModelStep
        """
        best_model_S3_artifact = step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=self.default_bucket, prefix=self.model_prefix
        )

        # https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/sagemaker.pytorch.html#sagemaker.pytorch.model.PyTorchModel
        pytorch_model = PyTorchModel(
            model_data=best_model_S3_artifact,
            role=self.role,
            framework_version="1.8.1",
            source_dir=BASE_DIR,
            py_version="py3",
            entry_point="inference.py",  # needs to be at the root of source_dir
            sagemaker_session=self.pipeline_session,
        )

        step_create_model = ModelStep(
            name=f"{self.project_name}-CreateModelStep",
            step_args=pytorch_model.create(instance_type="ml.t2.medium"),
        )

        ## NOTE: Due to PyTorch Processor bug this does not work. Running this duplicates the input_name for Eval Step.
        # model_metrics = ModelMetrics(
        #     model_statistics=MetricsSource(
        #         s3_uri=f'{step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]}/pred.json',
        #         content_type="application/json",
        #     )
        # )

        register_args = pytorch_model.register(
            content_types=["application/json", "text/csv"],
            response_types=["application/json", "text/csv"],
            inference_instances=["ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=self.model_package_group_name,
            approval_status=self.pipeline_parameters["model_approval_status"],
            # model_metrics=model_metrics,
        )
        step_register = ModelStep(
            name=f"{self.project_name}-RegisterModelStep", step_args=register_args
        )

        return step_create_model, step_register

    def get_deploy_step(
        self, step_create_model: sagemaker.workflow.model_step.ModelStep
    ) -> sagemaker.workflow.lambda_step.LambdaStep:
        """Construct Lambda Step to create a Lambda function that deploys the trained model.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.lambda_step.LambdaStep

        Args:
            step_create_model: Packaged trained model for inference.
        Returns:
            sagemaker.workflow.lambda_step.LambdaStep
        """
        current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        endpoint_config_name = (
            f"{self.project_name}-lambda-deploy-endpoint-config-{current_time}"
        )
        endpoint_name = f"{self.project_name}-lambda-deploy-endpoint-{current_time}"

        # By not using current_time in suffix, we will reuse Lambda function if it was created before.
        function_name = f"{self.project_name}-sagemaker-lambda-step-endpoint-deploy"

        # Lambda helper class can be used to create the Lambda function
        # https://sagemaker.readthedocs.io/en/stable/api/utility/lambda_helper.html?highlight=sagemaker%20helper%20lambda#sagemaker.lambda_helper.Lambda
        func = Lambda(
            function_name=function_name,
            execution_role_arn=self.lambda_role,
            script=os.path.join(BASE_DIR, "lambda_deploy.py"),
            handler="lambda_deploy.lambda_handler",
        )

        output_param_1 = LambdaOutput(
            output_name="statusCode", output_type=LambdaOutputTypeEnum.String
        )
        output_param_2 = LambdaOutput(
            output_name="body", output_type=LambdaOutputTypeEnum.String
        )
        output_param_3 = LambdaOutput(
            output_name="endpoint_name", output_type=LambdaOutputTypeEnum.String
        )
        output_param_4 = LambdaOutput(
            output_name="endpoint_config_name", output_type=LambdaOutputTypeEnum.String
        )

        # https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.lambda_step.LambdaStep
        step_deploy_lambda = LambdaStep(
            name=f"{self.project_name}-LambdaStep",
            lambda_func=func,
            inputs={
                "model_name": step_create_model.properties.ModelName,
                "endpoint_config_name": endpoint_config_name,
                "endpoint_name": endpoint_name,
                "inference_method": self.pipeline_parameters["inference_method"],
            },
            outputs=[output_param_1, output_param_2, output_param_3, output_param_4],
        )

        return step_deploy_lambda

    def construct_pipeline(self) -> sagemaker.workflow.pipeline.Pipeline:
        """Method to construct the pipeline with required steps.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#pipeline

        Args:
        Returns:
            sagemaker.workflow.pipeline.Pipeline
        """
        step_process = self.get_data_processing_step()
        step_tuning = self.get_tuning_step(step_process=step_process)
        step_eval = self.get_eval_step(step_tuning=step_tuning)
        step_create_model, step_register = self.get_model_step(
            step_eval=step_eval, step_tuning=step_tuning
        )
        step_deploy_lambda = self.get_deploy_step(step_create_model=step_create_model)

        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                self.pipeline_parameters["processing_instance_count"],
                self.pipeline_parameters["instance_type"],
                self.pipeline_parameters["model_approval_status"],
                self.pipeline_parameters["train_data"],
                self.pipeline_parameters["test_data"],
                self.pipeline_parameters["pretrained_model_data_dir"],
                self.pipeline_parameters["inference_method"],
            ],
            steps=[
                step_process,
                step_tuning,
                step_eval,
                step_create_model,
                step_register,
                step_deploy_lambda,
            ],
        )

        return pipeline
