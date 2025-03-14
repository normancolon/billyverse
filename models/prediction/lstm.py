import torch
import torch.nn as nn
from typing import Dict, Any
from models.prediction.base import BasePricePredictor

class LSTMPredictor(BasePricePredictor):
    """LSTM-based price prediction model"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        sequence_length: int = 60,
        batch_size: int = 32,
        dropout: float = 0.2
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            batch_size=batch_size
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size // 2)
        
        # Update metadata
        self.metadata.update({
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        try:
            # LSTM layers
            lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size)
            
            # Self-attention
            # Reshape for attention: (seq_len, batch, hidden_size)
            lstm_out = lstm_out.transpose(0, 1)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            attn_out = attn_out.transpose(0, 1)  # Back to (batch, seq_len, hidden_size)
            
            # Residual connection and layer normalization
            lstm_out = self.layer_norm1(lstm_out + attn_out)
            
            # Take the last sequence output
            last_out = lstm_out[:, -1, :]  # Shape: (batch, hidden_size)
            
            # Fully connected layers with residual connections
            fc1_out = self.fc1(last_out)
            fc1_out = self.layer_norm2(fc1_out)
            fc1_out = self.relu(fc1_out)
            fc1_out = self.dropout(fc1_out)
            
            # Final output layer
            out = self.fc2(fc1_out)
            
            return out
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def configure_optimizers(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> Dict[str, Any]:
        """Configure optimizers and learning rate scheduler"""
        try:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring optimizers: {str(e)}")
            raise 