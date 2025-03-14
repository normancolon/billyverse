from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger("billieverse.models")

class BaseModel(ABC):
    """Base class for all models in BillieVerse"""
    
    def __init__(self, model_name: str, version: str):
        self.model_name = model_name
        self.version = version
        self.model: Optional[Any] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metadata: Dict[str, Any] = {
            "name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "last_trained": None,
            "last_updated": None,
        }
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Preprocess input data before model inference"""
        pass
    
    @abstractmethod
    def postprocess(self, predictions: Any) -> Any:
        """Postprocess model predictions"""
        pass
    
    @abstractmethod
    def train(self, train_data: Any, validation_data: Any) -> Dict[str, float]:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> Any:
        """Make predictions using the model"""
        pass
    
    def save(self, path: str) -> None:
        """Save model and metadata"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state
        if isinstance(self.model, torch.nn.Module):
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'metadata': self.metadata
            }, path)
        else:
            raise NotImplementedError("Saving not implemented for this model type")
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model and metadata"""
        if not isinstance(self.model, torch.nn.Module):
            raise NotImplementedError("Loading not implemented for this model type")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metadata = checkpoint['metadata']
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        logger.info(f"Model loaded from {path}")
    
    def to_device(self, device: Optional[torch.device] = None) -> None:
        """Move model to specified device"""
        if device is not None:
            self.device = device
        
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata"""
        return self.metadata
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update model metadata"""
        self.metadata[key] = value
        self.metadata['last_updated'] = datetime.now().isoformat()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.model_name}, version={self.version})" 