import asyncio
import os
import pandas as pd
import torch
from typing import List, Dict, Any
from datetime import datetime
import logging
from pathlib import Path

from data.scraping.collector import MarketDataCollector
from models.sentiment.bert import BERTSentimentAnalyzer
from models.sentiment.trainer import SentimentTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.sentiment")

def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables"""
    return {
        'twitter_api_key': os.getenv('TWITTER_API_KEY'),
        'twitter_api_secret': os.getenv('TWITTER_API_SECRET'),
        'reddit_client_id': os.getenv('REDDIT_CLIENT_ID'),
        'reddit_client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'news_api_key': os.getenv('NEWS_API_KEY')
    }

def preprocess_data(data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """Preprocess collected data into a DataFrame"""
    records = []
    
    # Process news articles
    for article in data['news']:
        records.append({
            'text': f"{article['title']} {article['description']}",
            'source': 'news',
            'timestamp': article['published_at'],
            'url': article['url']
        })
    
    # Process tweets
    for tweet in data['tweets']:
        records.append({
            'text': tweet['text'],
            'source': 'twitter',
            'timestamp': tweet['created_at'],
            'url': f"https://twitter.com/{tweet['user']}"
        })
    
    # Process Reddit posts
    for post in data['reddit']:
        text = f"{post['title']}"
        if post['text']:
            text += f" {post['text']}"
        records.append({
            'text': text,
            'source': 'reddit',
            'timestamp': post['created_at'],
            'url': post['url']
        })
    
    return pd.DataFrame(records)

def apply_basic_sentiment(text: str) -> int:
    """Apply basic rule-based sentiment for initial labeling"""
    text = text.lower()
    
    # Positive keywords
    positive = ['bullish', 'surge', 'gain', 'rise', 'growth', 'profit', 'success',
               'positive', 'upward', 'strong', 'higher', 'rally', 'recover']
               
    # Negative keywords
    negative = ['bearish', 'crash', 'drop', 'fall', 'loss', 'decline', 'negative',
               'downward', 'weak', 'lower', 'plunge', 'risk']
    
    pos_count = sum(1 for word in positive if word in text)
    neg_count = sum(1 for word in negative if word in text)
    
    if pos_count > neg_count:
        return 2  # Positive
    elif neg_count > pos_count:
        return 0  # Negative
    return 1  # Neutral

async def main():
    try:
        # Load API keys
        api_keys = load_api_keys()
        
        # Initialize data collector
        collector = MarketDataCollector(**api_keys)
        
        # Define search parameters
        keywords = [
            "bitcoin", "crypto", "cryptocurrency", "BTC",
            "ethereum", "ETH", "blockchain"
        ]
        subreddits = [
            "Bitcoin", "CryptoCurrency", "CryptoMarkets",
            "ethereum", "BitcoinMarkets"
        ]
        
        logger.info("Collecting market sentiment data...")
        async with collector:
            data = await collector.collect_all_sources(
                keywords=keywords,
                subreddits=subreddits,
                days=7,  # Last 7 days of data
                max_tweets=1000,
                reddit_limit=1000
            )
        
        # Preprocess collected data
        logger.info("Preprocessing collected data...")
        df = preprocess_data(data)
        
        # Apply basic sentiment labels
        df['label'] = df['text'].apply(apply_basic_sentiment)
        
        # Initialize BERT model
        logger.info("Initializing BERT model...")
        model = BERTSentimentAnalyzer(
            model_name="ProsusAI/finbert",  # Pre-trained financial BERT
            num_classes=3,  # Negative, Neutral, Positive
            max_length=512,
            batch_size=16
        )
        
        # Initialize trainer
        trainer = SentimentTrainer(model)
        
        # Prepare data for training
        logger.info("Preparing training data...")
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
            trainer.prepare_data(texts, labels)
        
        # Train model
        logger.info("Training model...")
        history = trainer.train(
            train_texts=train_texts,
            train_labels=train_labels,
            val_texts=val_texts,
            val_labels=val_labels,
            num_epochs=5,
            patience=2,
            learning_rate=2e-5,
            weight_decay=0.01
        )
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = trainer.evaluate(test_texts, test_labels)
        
        logger.info("Test Metrics:")
        logger.info(f"Loss: {metrics['test_loss']:.4f}")
        logger.info(f"F1 Score: {metrics['metrics']['weighted avg']['f1-score']:.4f}")
        logger.info(f"Accuracy: {metrics['metrics']['accuracy']:.4f}")
        
        # Save training results
        results_dir = Path("results/sentiment")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        pd.DataFrame(metrics['metrics']).to_csv(
            results_dir / "test_metrics.csv"
        )
        
        # Save confusion matrix
        pd.DataFrame(
            metrics['confusion_matrix'],
            columns=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive'],
            index=['Actual Negative', 'Actual Neutral', 'Actual Positive']
        ).to_csv(results_dir / "confusion_matrix.csv")
        
        # Example predictions
        logger.info("\nExample Predictions:")
        example_texts = [
            "Bitcoin price surges to new all-time high as institutional adoption grows",
            "Cryptocurrency market faces severe correction amid regulatory concerns",
            "Trading volume remains stable as investors await market direction"
        ]
        
        predictions = model.predict_sentiment(example_texts)
        
        for text, pred in zip(example_texts, predictions):
            logger.info(f"\nText: {text}")
            logger.info(f"Sentiment: {pred}")
            logger.info(f"Predicted class: {max(pred.items(), key=lambda x: x[1])[0]}")

if __name__ == "__main__":
    asyncio.run(main()) 