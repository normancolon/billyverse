import torch
import torch.nn as nn
import math
from typing import Dict, Any
from models.prediction.base import BasePricePredictor

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerPredictor(BasePricePredictor):
    """Transformer-based price prediction model"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        output_size: int = 1,
        sequence_length: int = 60,
        batch_size: int = 32,
        dropout: float = 0.1
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            sequence_length=sequence_length,
            batch_size=batch_size
        )
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dim_feedforward, output_size)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(dim_feedforward)
        
        # Update metadata
        self.metadata.update({
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout
        })
    
    def generate_mask(self, size: int) -> torch.Tensor:
        """Generate attention mask for transformer"""
        mask = torch.triu(
            torch.ones(size, size),
            diagonal=1
        ).bool()
        return mask.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        try:
            # Input projection and positional encoding
            x = self.input_projection(x)
            x = self.pos_encoder(x)
            
            # Create attention mask
            mask = self.generate_mask(x.size(1))
            
            # Transformer encoder
            x = self.transformer_encoder(x, mask=mask)
            
            # Take the last sequence output
            x = x[:, -1, :]  # Shape: (batch, d_model)
            
            # Fully connected layers
            x = self.layer_norm1(x)
            x = self.fc1(x)
            x = self.layer_norm2(x)
            x = self.relu(x)
            x = self.dropout(x)
            out = self.fc2(x)
            
            return out
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def configure_optimizers(
        self,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        warmup_steps: int = 4000
    ) -> Dict[str, Any]:
        """Configure optimizers with learning rate scheduler"""
        try:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98)
            )
            
            def lr_lambda(step):
                # Transformer learning rate scheduler from "Attention is All You Need"
                step = max(1, step)
                return min(
                    step ** (-0.5),
                    step * (warmup_steps ** (-1.5))
                )
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda
            )
            
            return {
                'optimizer': optimizer,
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring optimizers: {str(e)}")
            raise 