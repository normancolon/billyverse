import os
from typing import Dict, List, Optional
import logging
import boto3
import sagemaker
from google.cloud import aiplatform
from google.cloud import storage
import tensorflow as tf
from infrastructure.config import config

logger = logging.getLogger("billieverse.infrastructure.deployment")

class AWSDeployer:
    """AWS model deployment manager"""
    
    def __init__(
        self,
        region: str = config.aws.region,
        bucket: str = config.aws.s3_bucket,
        instance_type: str = config.aws.sagemaker_instance
    ):
        # Initialize AWS clients
        self.s3 = boto3.client('s3', region_name=region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.sagemaker = sagemaker.Session()
        
        self.bucket = bucket
        self.instance_type = instance_type
        
        # Create bucket if not exists
        try:
            self.s3.head_bucket(Bucket=bucket)
        except:
            self.s3.create_bucket(Bucket=bucket)
    
    def upload_model(
        self,
        model_path: str,
        model_name: str,
        version: str = config.model.model_version
    ) -> str:
        """Upload model to S3"""
        try:
            # Save model artifacts
            s3_path = f"models/{model_name}/{version}"
            self.sagemaker.upload_data(
                path=model_path,
                bucket=self.bucket,
                key_prefix=s3_path
            )
            
            model_uri = f"s3://{self.bucket}/{s3_path}"
            logger.info(f"Uploaded model to {model_uri}")
            
            return model_uri
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise
    
    def deploy_endpoint(
        self,
        model_uri: str,
        model_name: str,
        instance_count: int = 1
    ) -> str:
        """Deploy model to SageMaker endpoint"""
        try:
            # Create model
            model = sagemaker.Model(
                model_data=model_uri,
                role=self._get_role(),
                framework_version="2.8.0",
                py_version="py38",
                entry_point="inference.py",
                sagemaker_session=self.sagemaker
            )
            
            # Deploy endpoint
            predictor = model.deploy(
                instance_type=self.instance_type,
                initial_instance_count=instance_count,
                endpoint_name=model_name
            )
            
            endpoint_name = predictor.endpoint_name
            logger.info(f"Deployed model to endpoint {endpoint_name}")
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying endpoint: {str(e)}")
            raise
    
    def _get_role(self) -> str:
        """Get or create IAM role for SageMaker"""
        try:
            iam = boto3.client('iam')
            role_name = "SageMakerExecutionRole"
            
            try:
                role = iam.get_role(RoleName=role_name)
                return role['Role']['Arn']
            except:
                # Create role
                trust_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {
                                "Service": "sagemaker.amazonaws.com"
                            },
                            "Action": "sts:AssumeRole"
                        }
                    ]
                }
                
                role = iam.create_role(
                    RoleName=role_name,
                    AssumeRolePolicyDocument=json.dumps(trust_policy)
                )
                
                # Attach policies
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                )
                
                return role['Role']['Arn']
                
        except Exception as e:
            logger.error(f"Error getting IAM role: {str(e)}")
            raise

class GCPDeployer:
    """GCP model deployment manager"""
    
    def __init__(
        self,
        project_id: str = config.gcp.project_id,
        region: str = config.gcp.region,
        bucket: str = config.gcp.gcs_bucket
    ):
        # Initialize GCP clients
        aiplatform.init(
            project=project_id,
            location=region
        )
        self.storage_client = storage.Client()
        
        self.project_id = project_id
        self.region = region
        self.bucket = bucket
        
        # Create bucket if not exists
        try:
            self.storage_client.get_bucket(bucket)
        except:
            self.storage_client.create_bucket(bucket)
    
    def upload_model(
        self,
        model_path: str,
        model_name: str,
        version: str = config.model.model_version
    ) -> str:
        """Upload model to GCS"""
        try:
            # Save model artifacts
            gcs_path = f"gs://{self.bucket}/models/{model_name}/{version}"
            bucket = self.storage_client.bucket(self.bucket)
            
            # Upload model files
            for root, _, files in os.walk(model_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, model_path)
                    blob_path = f"models/{model_name}/{version}/{relative_path}"
                    
                    blob = bucket.blob(blob_path)
                    blob.upload_from_filename(local_path)
            
            logger.info(f"Uploaded model to {gcs_path}")
            return gcs_path
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise
    
    def deploy_endpoint(
        self,
        model_uri: str,
        model_name: str,
        machine_type: str = config.gcp.machine_type,
        min_replicas: int = 1
    ) -> str:
        """Deploy model to AI Platform endpoint"""
        try:
            # Create model
            model = aiplatform.Model.upload(
                display_name=model_name,
                artifact_uri=model_uri,
                serving_container_image_uri=(
                    "us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-8"
                )
            )
            
            # Deploy endpoint
            endpoint = model.deploy(
                machine_type=machine_type,
                min_replica_count=min_replicas,
                accelerator_type=config.gcp.gpu_type,
                accelerator_count=config.gcp.gpu_count
            )
            
            endpoint_name = endpoint.resource_name
            logger.info(f"Deployed model to endpoint {endpoint_name}")
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying endpoint: {str(e)}")
            raise

class ModelDeployer:
    """Model deployment manager"""
    
    def __init__(self):
        self.aws = AWSDeployer()
        self.gcp = GCPDeployer()
        
        # Track deployments
        self.endpoints: Dict[str, Dict] = {}
    
    def deploy(
        self,
        model_path: str,
        model_name: str,
        platform: str = "aws",
        version: str = config.model.model_version,
        **kwargs
    ) -> str:
        """Deploy model to specified platform"""
        try:
            deployer = self.aws if platform == "aws" else self.gcp
            
            # Upload model
            model_uri = deployer.upload_model(
                model_path,
                model_name,
                version
            )
            
            # Deploy endpoint
            endpoint_name = deployer.deploy_endpoint(
                model_uri,
                model_name,
                **kwargs
            )
            
            # Track deployment
            self.endpoints[model_name] = {
                'platform': platform,
                'endpoint': endpoint_name,
                'version': version,
                'uri': model_uri
            }
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise

# Create global deployer instance
model_deployer = ModelDeployer() 