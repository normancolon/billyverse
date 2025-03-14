import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from models.sentiment.base import BaseSentimentAnalyzer
from core.config import settings

logger = logging.getLogger("billieverse.models.sentiment.trainer")

class SentimentTrainer:
    """Trainer for sentiment analysis models"""
    
    def __init__(
        self,
        model: BaseSentimentAnalyzer,
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
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[List[str], List[str], List[str], List[int], List[int], List[int]]:
        """Prepare and split data for training"""
        try:
            # First split: train + val, test
            train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
                texts,
                labels,
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            
            # Second split: train, val
            val_ratio = val_size / (1 - test_size)
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_val_texts,
                train_val_labels,
                test_size=val_ratio,
                random_state=random_state,
                stratify=train_val_labels
            )
            
            return (
                train_texts, val_texts, test_texts,
                train_labels, val_labels, test_labels
            )
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def train_epoch(
        self,
        train_texts: List[str],
        train_labels: List[int],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """Train for one epoch"""
        try:
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(train_texts), self.model.batch_size):
                batch_texts = train_texts[i:i + self.model.batch_size]
                batch_labels = train_labels[i:i + self.model.batch_size]
                
                # Prepare batch
                encoded = self.model.preprocess_text(batch_texts)
                labels = torch.tensor(batch_labels).to(self.model.device)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
                
                # Calculate loss
                loss = criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            return total_loss / num_batches
            
        except Exception as e:
            self.logger.error(f"Error in training epoch: {str(e)}")
            raise
    
    def validate(
        self,
        val_texts: List[str],
        val_labels: List[int],
        criterion: nn.Module
    ) -> Tuple[float, Dict[str, float]]:
        """Validate model"""
        try:
            self.model.eval()
            total_loss = 0
            num_batches = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for i in range(0, len(val_texts), self.model.batch_size):
                    batch_texts = val_texts[i:i + self.model.batch_size]
                    batch_labels = val_labels[i:i + self.model.batch_size]
                    
                    # Prepare batch
                    encoded = self.model.preprocess_text(batch_texts)
                    labels = torch.tensor(batch_labels).to(self.model.device)
                    
                    # Forward pass
                    logits = self.model(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask']
                    )
                    
                    # Calculate loss
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Get predictions
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Calculate metrics
            metrics = classification_report(
                all_labels,
                all_preds,
                output_dict=True
            )
            
            return total_loss / num_batches, metrics
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise
    
    def train(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: List[str],
        val_labels: List[int],
        num_epochs: int = 5,
        patience: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ) -> Dict[str, Any]:
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
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            patience_counter = 0
            best_val_metrics = None
            
            for epoch in range(num_epochs):
                # Training
                train_loss = self.train_epoch(
                    train_texts,
                    train_labels,
                    optimizer,
                    criterion
                )
                self.train_losses.append(train_loss)
                
                # Validation
                val_loss, val_metrics = self.validate(
                    val_texts,
                    val_labels,
                    criterion
                )
                self.val_losses.append(val_loss)
                
                # Learning rate scheduling
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_val_metrics = val_metrics
                    patience_counter = 0
                    # Save best model
                    self.save_model('best_model.pt')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 1 == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Val F1: {val_metrics['weighted avg']['f1-score']:.4f}"
                    )
            
            # Load best model
            self.load_model('best_model.pt')
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_val_metrics': best_val_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
    
    def evaluate(
        self,
        test_texts: List[str],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """Evaluate model on test set"""
        try:
            # Get loss and metrics
            test_loss, test_metrics = self.validate(
                test_texts,
                test_labels,
                nn.CrossEntropyLoss()
            )
            
            # Get confusion matrix
            self.model.eval()
            with torch.no_grad():
                encoded = self.model.preprocess_text(test_texts)
                logits = self.model(
                    input_ids=encoded['input_ids'],
                    attention_mask=encoded['attention_mask']
                )
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            conf_matrix = confusion_matrix(test_labels, preds)
            
            return {
                'test_loss': test_loss,
                'metrics': test_metrics,
                'confusion_matrix': conf_matrix.tolist()
            }
            
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