"""Sagemaker Pipeline Constructor.

Tabular example pipeline with XGBoost model training and deployment.

https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html
"""

import os
import time
import boto3
import sagemaker

from dataclasses import dataclass
from typing import Tuple

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TuningStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import ContinuousParameter, HyperparameterTuner
from sagemaker.workflow.properties import PropertyFile

from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.xgboost import XGBoostPredictor
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


class TabularPipeline:
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

    def __init__(self, role, lambda_role, region, default_bucket, project_name) -> None:
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
                name="TrainingInstanceType", default_value="ml.m5.large"
            ),
            "model_approval_status": ParameterString(
                name="ModelApprovalStatus", default_value="PendingManualApproval"
            ),
            "input_data": ParameterString(
                name="InputData",
                default_value="s3://<>/abalone.csv",
            ),
            "inference_method": ParameterString(
                name="InferenceMethod",
                default_value="serverless",
            ),
        }
        self.image_uri = sagemaker.image_uris.retrieve(
            framework="xgboost",
            region=self.region,
            version="1.0-1",
            py_version="py3",
            instance_type="ml.m5.large",
        )

    def get_data_processing_step(self) -> sagemaker.workflow.steps.ProcessingStep:
        """Construct Processing Step for data processing.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.ProcessingStep

        Args:

        Returns:
            sagemaker.workflow.steps.ProcessingStep
        """
        # https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor
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
                    source=self.pipeline_parameters["input_data"],
                    destination="/opt/ml/processing/input",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train", source="/opt/ml/processing/train"
                ),
                ProcessingOutput(
                    output_name="validation", source="/opt/ml/processing/validation"
                ),
                ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
            ],
            code=os.path.join(BASE_DIR, "preprocessing.py"),
        )

        # https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.ProcessingStep
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
        # https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator
        # Load XGBoost image preconstructed by Sagemaker
        xgb_train = Estimator(
            image_uri=self.image_uri,
            instance_type=self.pipeline_parameters["instance_type"],
            instance_count=1,
            output_path=self.model_path,
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )
        xgb_train.set_hyperparameters(
            eval_metric="rmse",
            objective="reg:squarederror",
            num_round=5,
            max_depth=5,
            eta=0.2,
            gamma=4,
            min_child_weight=6,
            subsample=0.7,
        )

        objective_metric_name = "validation:rmse"

        hyperparameter_ranges = {
            "alpha": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
            "lambda": ContinuousParameter(0.01, 10, scaling_type="Logarithmic"),
        }

        # https://sagemaker.readthedocs.io/en/stable/api/training/tuner.html
        tuner_log = HyperparameterTuner(
            xgb_train,
            objective_metric_name,
            hyperparameter_ranges,
            max_jobs=3,
            max_parallel_jobs=3,
            strategy="Bayesian",
            objective_type="Minimize",
        )

        hpo_args = tuner_log.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                        "train"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    content_type="text/csv",
                ),
            }
        )

        # https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep
        step_tuning = TuningStep(
            name=f"{self.project_name}-HPTuningStep",
            step_args=hpo_args,
            cache_config=self.cache_config,
        )

        return step_tuning

    def get_eval_step(
        self,
        step_process: sagemaker.workflow.steps.ProcessingStep,
        step_tuning: sagemaker.workflow.steps.TuningStep,
    ) -> sagemaker.workflow.steps.ProcessingStep:
        """Construct Processing Step for model evaluation on test data.

        https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.TuningStep

        Args:
            step_process: The data processing step that prepares the training data.
            step_tuning: Hyperparam tuning step with saved models.
        Returns:
            sagemaker.workflow.steps.ProcessingStep
        """
        best_model_S3_artifact = step_tuning.get_top_model_s3_uri(
            top_k=0, s3_bucket=self.default_bucket, prefix=self.model_prefix
        )

        # https://sagemaker.readthedocs.io/en/stable/api/training/processing.html#sagemaker.processing.ScriptProcessor
        script_eval = ScriptProcessor(
            image_uri=self.image_uri,
            command=["python3"],
            instance_type="ml.m5.large",
            instance_count=1,
            base_job_name=f"{self.project_name}-eval",
            role=self.role,
            sagemaker_session=self.pipeline_session,
        )

        eval_args = script_eval.run(
            inputs=[
                ProcessingInput(
                    source=best_model_S3_artifact,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs[
                        "test"
                    ].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation", source="/opt/ml/processing/evaluation"
                ),
            ],
            code=os.path.join(BASE_DIR, "evaluation.py"),
        )

        evaluation_report = PropertyFile(
            name=f"{self.project_name}-BestModelEvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        )
        # https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#sagemaker.workflow.steps.ProcessingStep
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
        # Use predefined XGBoostPredictor for inference.
        best_model = Model(
            image_uri=self.image_uri,
            model_data=best_model_S3_artifact,
            predictor_cls=XGBoostPredictor,
            sagemaker_session=self.pipeline_session,
            role=self.role,
        )

        step_create_model = ModelStep(
            name=f"{self.project_name}-CreateModelStep",
            step_args=best_model.create(instance_type="ml.t2.medium"),
        )

        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=f'{step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]}/evaluation.json',
                content_type="application/json",
            )
        )

        register_args = best_model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=self.model_package_group_name,
            approval_status=self.pipeline_parameters["model_approval_status"],
            model_metrics=model_metrics,
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
        step_eval = self.get_eval_step(
            step_process=step_process, step_tuning=step_tuning
        )
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
                self.pipeline_parameters["input_data"],
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
