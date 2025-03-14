import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from core.config import settings

logger = logging.getLogger("billieverse.models.sentiment")

class BaseSentimentAnalyzer(nn.Module):
    """Base class for sentiment analysis models"""
    
    def __init__(
        self,
        model_name: str = "finbert-base",
        max_length: int = 512,
        batch_size: int = 16
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            raise
        
        # Training metrics
        self.train_losses: list = []
        self.val_losses: list = []
        self.best_val_loss = float('inf')
        
        # Model metadata
        self.metadata = {
            "model_type": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "last_trained": None,
            "model_name": model_name,
            "max_length": max_length,
            "batch_size": batch_size
        }
    
    def preprocess_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Preprocess text data for model input"""
        try:
            # Tokenize texts
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            return encoded
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save model state and metadata"""
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'metadata': self.metadata,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_loss': self.best_val_loss
            }, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, path: str) -> None:
        """Load model state and metadata"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.metadata = checkpoint['metadata']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.best_val_loss = checkpoint['best_val_loss']
            self.logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise 