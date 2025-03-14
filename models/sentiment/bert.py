import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from transformers import AutoModel
from models.sentiment.base import BaseSentimentAnalyzer

class BERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """BERT-based sentiment analysis model"""
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        num_classes: int = 3,  # Negative, Neutral, Positive
        max_length: int = 512,
        batch_size: int = 16,
        dropout: float = 0.1
    ):
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            batch_size=batch_size
        )
        
        # Load BERT model
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {str(e)}")
            raise
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # Update metadata
        self.metadata.update({
            "num_classes": num_classes,
            "dropout": dropout
        })
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass through the network"""
        try:
            # Get BERT outputs
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Get [CLS] token representation
            pooled_output = outputs.pooler_output
            
            # Apply layer normalization and dropout
            pooled_output = self.layer_norm(pooled_output)
            pooled_output = self.dropout(pooled_output)
            
            # Classification
            logits = self.classifier(pooled_output)
            
            return logits
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def predict_sentiment(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts"""
        try:
            self.eval()
            results = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                encoded = self.preprocess_text(batch_texts)
                
                with torch.no_grad():
                    logits = self(
                        input_ids=encoded['input_ids'],
                        attention_mask=encoded['attention_mask']
                    )
                    probs = torch.softmax(logits, dim=1)
                
                # Convert to list of dictionaries
                batch_results = [
                    {
                        'negative': float(p[0]),
                        'neutral': float(p[1]),
                        'positive': float(p[2])
                    }
                    for p in probs.cpu().numpy()
                ]
                results.extend(batch_results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in sentiment prediction: {str(e)}")
            raise
    
    def configure_optimizers(
        self,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000
    ) -> Dict[str, Any]:
        """Configure optimizers with learning rate scheduler"""
        try:
            # Separate BERT and classifier parameters
            bert_params = list(self.bert.parameters())
            classifier_params = list(self.classifier.parameters())
            
            # Different learning rates for BERT and classifier
            optimizer = torch.optim.AdamW([
                {'params': bert_params, 'lr': learning_rate},
                {'params': classifier_params, 'lr': learning_rate * 10}
            ], weight_decay=weight_decay)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                return max(
                    0.0,
                    float(warmup_steps) / float(max(1, step))
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