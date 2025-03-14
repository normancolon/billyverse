import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from models.prediction.base import BasePricePredictor
from core.config import settings

logger = logging.getLogger("billieverse.models.trainer")

class ModelTrainer:
    """Trainer class for price prediction models"""
    
    def __init__(
        self,
        model: BasePricePredictor,
        save_dir: Optional[str] = None
    ):
        self.model = model
        self.save_dir = Path(save_dir or settings.model.MODEL_CHECKPOINT_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Training metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.best_model_path: Optional[str] = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[list] = None,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare and split data for training"""
        try:
            # Prepare sequences
            X, y = self.model.prepare_data(df, target_col, feature_cols)
            
            # Split into train, validation, and test sets
            X_train, X_temp, y_train, y_temp = train_test_split(
                X.cpu().numpy(),
                y.cpu().numpy(),
                test_size=test_size + val_size,
                shuffle=False  # Keep time series order
            )
            
            # Split temp into validation and test
            val_ratio = val_size / (test_size + val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=1-val_ratio,
                shuffle=False
            )
            
            # Convert back to tensors
            X_train = torch.FloatTensor(X_train).to(self.model.device)
            X_val = torch.FloatTensor(X_val).to(self.model.device)
            X_test = torch.FloatTensor(X_test).to(self.model.device)
            y_train = torch.FloatTensor(y_train).to(self.model.device)
            y_val = torch.FloatTensor(y_val).to(self.model.device)
            y_test = torch.FloatTensor(y_test).to(self.model.device)
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        num_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> Dict[str, List[float]]:
        """Train the model"""
        try:
            # Configure optimizers
            optim_config = self.model.configure_optimizers(
                learning_rate=learning_rate,
                weight_decay=weight_decay
            )
            optimizer = optim_config['optimizer']
            scheduler = optim_config['scheduler']
            
            # Loss function
            criterion = nn.MSELoss()
            
            # Training loop
            patience_counter = 0
            
            for epoch in range(num_epochs):
                # Training
                train_loss = 0
                num_batches = 0
                
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i + batch_size]
                    batch_y = y_train[i:i + batch_size]
                    
                    loss = self.model.train_step(
                        batch_X,
                        batch_y,
                        optimizer,
                        criterion
                    )
                    train_loss += loss
                    num_batches += 1
                
                avg_train_loss = train_loss / num_batches
                self.train_losses.append(avg_train_loss)
                
                # Validation
                val_loss = self.model.validate(X_val, y_val, criterion)
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model('best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}"
                    )
            
            # Load best model
            self.load_model('best_model.pt')
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(
        self,
        X_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(X_test)
                
                # Convert to numpy for metrics calculation
                y_true = y_test.cpu().numpy()
                y_pred = y_pred.cpu().numpy()
                
                # Calculate metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                self.logger.info(f"Test Metrics: {metrics}")
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def save_model(self, filename: str) -> None:
        """Save model checkpoint"""
        try:
            path = self.save_dir / filename
            self.model.save_model(str(path))
            self.best_model_path = str(path)
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filename: str) -> None:
        """Load model checkpoint"""
        try:
            path = self.save_dir / filename
            self.model.load_model(str(path))
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise 