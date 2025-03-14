from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, DirectoryPath, HttpUrl
from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from functools import lru_cache

class APIKeys(BaseSettings):
    """API Keys Configuration"""
    ALPHAVANTAGE_API_KEY: str = ""
    BINANCE_API_KEY: str = ""
    BINANCE_SECRET_KEY: str = ""
    COINBASE_API_KEY: str = ""
    COINBASE_SECRET_KEY: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class DatabaseSettings(BaseSettings):
    """Database Configuration"""
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "billieverse"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    @property
    def DATABASE_URL(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_HOST,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )
    
    class Config:
        env_file = ".env"

class ModelSettings(BaseSettings):
    """Model Configuration"""
    MODEL_BASE_PATH: DirectoryPath = Path("models")
    MARKET_MODEL_VERSION: str = "v1"
    RISK_MODEL_VERSION: str = "v1"
    STRATEGY_MODEL_VERSION: str = "v1"
    MODEL_CHECKPOINT_DIR: DirectoryPath = Path("models/checkpoints")
    MODEL_ARCHIVE_DIR: DirectoryPath = Path("models/archive")
    
    # Model hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    MAX_EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 10
    
    class Config:
        env_file = ".env"

class LoggingSettings(BaseSettings):
    """Logging Configuration"""
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[Path] = None
    LOG_RETENTION_DAYS: int = 30
    ENABLE_CONSOLE_LOGGING: bool = True
    ENABLE_FILE_LOGGING: bool = True
    
    class Config:
        env_file = ".env"

class TradingSettings(BaseSettings):
    """Trading Configuration"""
    MAX_POSITION_SIZE: float = 100000.0
    RISK_LIMIT_PERCENTAGE: float = 2.0
    STOP_LOSS_PERCENTAGE: float = 1.0
    TAKE_PROFIT_PERCENTAGE: float = 2.0
    MAX_LEVERAGE: float = 3.0
    MIN_ORDER_SIZE: float = 10.0
    MAX_SLIPPAGE_PERCENTAGE: float = 0.1
    TRADING_PAIRS: List[str] = ["BTC/USD", "ETH/USD"]
    
    class Config:
        env_file = ".env"

class WebSocketSettings(BaseSettings):
    """WebSocket Configuration"""
    WS_HOST: str = "0.0.0.0"
    WS_PORT: int = 8001
    PING_INTERVAL: int = 20
    PING_TIMEOUT: int = 10
    CLOSE_TIMEOUT: int = 10
    MAX_MESSAGE_SIZE: int = 1024 * 1024  # 1MB
    
    class Config:
        env_file = ".env"

class Settings(BaseSettings):
    """Main Settings Class"""
    # Application settings
    APP_NAME: str = "BillieVerse AI Trading"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security settings
    SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Component configurations
    api_keys: APIKeys = APIKeys()
    database: DatabaseSettings = DatabaseSettings()
    model: ModelSettings = ModelSettings()
    logging: LoggingSettings = LoggingSettings()
    trading: TradingSettings = TradingSettings()
    websocket: WebSocketSettings = WebSocketSettings()
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Create settings instance
settings = get_settings() 