from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from core.config import settings

logger = logging.getLogger("billieverse.models.prediction")

class BasePricePredictor(nn.Module, ABC):
    """Base class for price prediction models"""
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        sequence_length: int,
        batch_size: int
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        
        # Training metrics
        self.train_losses: list = []
        self.val_losses: list = []
        self.best_val_loss = float('inf')
        
        # Model metadata
        self.metadata = {
            "model_type": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "last_trained": None,
            "input_size": input_size,
            "output_size": output_size,
            "sequence_length": sequence_length,
            "batch_size": batch_size
        }
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[list] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training/prediction"""
        try:
            if feature_cols is None:
                # Use all numeric columns except the target
                feature_cols = [
                    col for col in df.select_dtypes(include=[np.number]).columns
                    if col != target_col
                ]
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(df) - self.sequence_length):
                # Get sequence of features
                seq = df[feature_cols].iloc[i:i + self.sequence_length].values
                # Get target (next price)
                target = df[target_col].iloc[i + self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
            
            # Convert to tensors
            X = torch.FloatTensor(sequences).to(self.device)
            y = torch.FloatTensor(targets).to(self.device)
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_step(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Single training step"""
        try:
            self.train()
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = self(X)
            loss = criterion(y_pred, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            raise
    
    def validate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module
    ) -> float:
        """Validate model"""
        try:
            self.eval()
            with torch.no_grad():
                y_pred = self(X)
                loss = criterion(y_pred, y)
            return loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        try:
            self.eval()
            with torch.no_grad():
                predictions = self(X)
            return predictions.cpu().numpy()
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
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