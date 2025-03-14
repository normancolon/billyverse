import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from models.anomaly.base import BaseAnomalyDetector, AnomalyMetrics

logger = logging.getLogger("billieverse.models.anomaly.autoencoder")

class Autoencoder(nn.Module):
    """Autoencoder neural network for anomaly detection"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space"""
        return self.decoder(z)

class AutoencoderDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector"""
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        dropout: float = 0.2,
        contamination: float = 0.1,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_state: Optional[int] = None
    ):
        super().__init__(
            contamination=contamination,
            random_state=random_state
        )
        
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.device = device
        self.model = None
        
        if random_state is not None:
            torch.manual_seed(random_state)
    
    def _create_dataloader(
        self,
        X: np.ndarray
    ) -> DataLoader:
        """Create PyTorch DataLoader"""
        try:
            X_tensor = torch.FloatTensor(X).to(self.device)
            dataset = TensorDataset(X_tensor, X_tensor)  # (input, target)
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
        except Exception as e:
            self.logger.error(f"Error creating DataLoader: {str(e)}")
            raise
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> 'AutoencoderDetector':
        """Fit the autoencoder"""
        try:
            # Preprocess data
            X = self.preprocess_data(X)
            
            # Initialize model
            self.model = Autoencoder(
                input_dim=X.shape[1],
                hidden_dims=self.hidden_dims,
                latent_dim=self.latent_dim,
                dropout=self.dropout
            ).to(self.device)
            
            # Create DataLoader
            train_loader = self._create_dataloader(X)
            
            # Initialize optimizer and loss
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate
            )
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_x, _ in train_loader:
                    # Forward pass
                    x_hat, _ = self.model(batch_x)
                    loss = criterion(x_hat, batch_x)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / len(train_loader)
                    self.logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting autoencoder: {str(e)}")
            raise
    
    def score_samples(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute reconstruction error as anomaly score"""
        try:
            # Preprocess data
            X = self.preprocess_data(X)
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Compute reconstruction error
            self.model.eval()
            with torch.no_grad():
                X_hat, _ = self.model(X_tensor)
                reconstruction_error = torch.mean(
                    torch.pow(X_tensor - X_hat, 2),
                    dim=1
                ).cpu().numpy()
            
            return reconstruction_error
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> AnomalyMetrics:
        """Predict anomalies based on reconstruction error"""
        try:
            # Compute anomaly scores
            scores = self.score_samples(X)
            
            # Calculate threshold
            threshold = self.calculate_threshold(scores)
            
            # Identify anomalies
            labels = (scores > threshold).astype(int)
            num_anomalies = np.sum(labels)
            
            return AnomalyMetrics(
                anomaly_scores=scores,
                anomaly_labels=labels,
                threshold=threshold,
                num_anomalies=num_anomalies,
                contamination=self.contamination
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting anomalies: {str(e)}")
            raise
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            state = {
                'model_state': self.model.state_dict(),
                'hidden_dims': self.hidden_dims,
                'latent_dim': self.latent_dim,
                'dropout': self.dropout,
                'contamination': self.contamination,
                'scaler': self.scaler
            }
            torch.save(state, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'AutoencoderDetector':
        """Load model from disk"""
        try:
            state = torch.load(path)
            detector = cls(
                hidden_dims=state['hidden_dims'],
                latent_dim=state['latent_dim'],
                dropout=state['dropout'],
                contamination=state['contamination']
            )
            detector.model.load_state_dict(state['model_state'])
            detector.scaler = state['scaler']
            logger.info(f"Model loaded from {path}")
            return detector
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 