from typing import Dict, List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path

class RedisConfig(BaseSettings):
    """Redis configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    ssl: bool = Field(default=False)
    queue_names: List[str] = Field(
        default=[
            "market_data",
            "signals",
            "orders",
            "executions",
            "risk_updates"
        ]
    )

class KafkaConfig(BaseSettings):
    """Kafka configuration"""
    bootstrap_servers: List[str] = Field(default=["localhost:9092"])
    topics: Dict[str, Dict] = Field(
        default={
            "market_data": {
                "num_partitions": 3,
                "replication_factor": 2
            },
            "signals": {
                "num_partitions": 2,
                "replication_factor": 2
            },
            "orders": {
                "num_partitions": 2,
                "replication_factor": 2
            }
        }
    )
    consumer_group: str = Field(default="trading_group")
    auto_offset_reset: str = Field(default="latest")

class AWSConfig(BaseSettings):
    """AWS configuration"""
    region: str = Field(default="us-east-1")
    sagemaker_instance: str = Field(default="ml.g4dn.xlarge")
    s3_bucket: str = Field(default="ai-trading-models")
    ecs_cluster: str = Field(default="trading-cluster")
    ecr_repository: str = Field(default="trading-models")
    lambda_memory: int = Field(default=1024)
    lambda_timeout: int = Field(default=300)

class GCPConfig(BaseSettings):
    """GCP configuration"""
    project_id: str = Field(default="ai-trading")
    region: str = Field(default="us-central1")
    zone: str = Field(default="us-central1-a")
    machine_type: str = Field(default="n1-standard-4")
    gpu_type: str = Field(default="nvidia-tesla-t4")
    gpu_count: int = Field(default=1)
    gcs_bucket: str = Field(default="ai-trading-models")

class ModelConfig(BaseSettings):
    """Model deployment configuration"""
    batch_size: int = Field(default=32)
    num_workers: int = Field(default=4)
    update_interval: int = Field(default=300)  # 5 minutes
    max_queue_size: int = Field(default=1000)
    model_version: str = Field(default="v1")
    checkpoint_dir: str = Field(default="checkpoints")
    tensorboard_dir: str = Field(default="logs")

class DatabaseConfig(BaseSettings):
    """Database configuration"""
    postgres_url: str = Field(default="postgresql://user:pass@localhost:5432/trading")
    mongo_url: str = Field(default="mongodb://localhost:27017")
    mongo_db: str = Field(default="trading")
    redis_url: str = Field(default="redis://localhost:6379")

class InfrastructureConfig(BaseSettings):
    """Main infrastructure configuration"""
    env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Component configs
    redis: RedisConfig = Field(default_factory=RedisConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Paths
    base_path: Path = Field(default=Path(__file__).parent.parent)
    data_path: Path = Field(default=Path(__file__).parent.parent / "data")
    model_path: Path = Field(default=Path(__file__).parent.parent / "models")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    def setup(self):
        """Setup infrastructure configuration"""
        # Create necessary directories
        self.data_path.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)
        
        # Set environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
        
        return self

# Create global config instance
config = InfrastructureConfig().setup() 