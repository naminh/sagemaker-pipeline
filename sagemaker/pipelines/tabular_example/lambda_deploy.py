"""Lambda Function code.

Given input model name, deploy it as Sagemaker Serverless or Realtime Endpoint.
"""

import boto3
import json
import logging


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

sm_client = boto3.client("sagemaker")


def lambda_handler(event, context):
    """Lambda to deploy to Sagemaker Endpoint.

    Creates an endpoint config and endpoint as per event input.

    Args:
      event:
        dict
            model_name: model name created by Sagemaker Training Job
            endpoint_config_name
            endpoint_name
            inference_method: takes either "serverless" or "realtime"
      context:
        Additional Lambda context.
    Returns:
        dict
    """

    # The name of the model created in the Pipeline CreateModelStep
    model_name = event["model_name"]

    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]
    inference_method = event["inference_method"]

    if inference_method == "serverless":
        logger.info(
            f"Deploying as Serverless Model with \
            inference_method: {inference_method}, \
            endpoint_config_name: {endpoint_config_name}, \
            inference_method: {inference_method},"
        )

        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "ServerlessConfig": {"MemorySizeInMB": 4096, "MaxConcurrency": 1},
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
    elif inference_method == "realtime":
        logger.info(
            f"Deploying as Real-time Model with \
            inference_method: {inference_method}, \
            endpoint_config_name: {endpoint_config_name}, \
            inference_method: {inference_method},"
        )
        create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "InstanceType": "ml.m5.large",
                    "InitialVariantWeight": 1,
                    "InitialInstanceCount": 1,
                    "ModelName": model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
        "endpoint_name": endpoint_name,
        "endpoint_config_name": endpoint_config_name,
    }
