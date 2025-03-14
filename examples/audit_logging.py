import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from web3 import Web3
import matplotlib.pyplot as plt
import seaborn as sns

from models.audit.logging import AIAuditLogger, ModelDecision
from infrastructure.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.audit")

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def create_sample_model(input_dim: int = 10) -> tf.keras.Model:
    """Create a sample trading model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    setattr(model, 'version', '1.0.0')
    
    return model

def generate_sample_data(n_samples: int = 1000, n_features: int = 10):
    """Generate sample trading data"""
    # Generate feature names
    feature_names = [
        f"feature_{i+1}" for i in range(n_features)
    ]
    
    # Generate random features
    X = np.random.random((n_samples, n_features))
    
    # Generate target (simple rule: if mean of features > 0.5, then 1)
    y = (np.mean(X, axis=1) > 0.5).astype(float)
    
    return X, y, feature_names

def plot_feature_importance(
    importance_dict: dict,
    title: str,
    save_path: str
):
    """Plot feature importance"""
    plt.figure(figsize=(10, 6))
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    sns.barplot(x=importance, y=features)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_model_metrics(
    metrics: dict,
    save_path: str
):
    """Plot model performance metrics"""
    plt.figure(figsize=(12, 6))
    
    # Plot confidence distribution
    plt.subplot(1, 2, 1)
    plt.hist(
        [d.confidence for d in metrics['decisions']],
        bins=20,
        alpha=0.7
    )
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    # Plot prediction distribution
    plt.subplot(1, 2, 2)
    plt.hist(
        [d.prediction for d in metrics['decisions']],
        bins=20,
        alpha=0.7
    )
    plt.title('Prediction Distribution')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

async def run_audit_example():
    """Run AI audit logging example"""
    try:
        logger.info("Starting AI audit logging example...")
        
        # Connect to Ethereum network
        web3 = Web3(Web3.HTTPProvider(os.getenv('ETH_NODE_URL')))
        private_key = os.getenv('PRIVATE_KEY')
        contract_address = os.getenv('AUDIT_CONTRACT_ADDRESS')
        
        if not web3.is_connected():
            raise Exception("Failed to connect to Ethereum network")
        
        # Initialize audit logger
        audit_logger = AIAuditLogger(
            web3,
            contract_address,
            private_key,
            str(results_dir / "audit_logs.json")
        )
        
        # Create sample model and data
        logger.info("Creating sample model and data...")
        model = create_sample_model()
        X, y, feature_names = generate_sample_data()
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        logger.info("Training model...")
        model.fit(X_train, y_train, epochs=5, verbose=0)
        
        # Initialize explainers
        logger.info("Initializing explainers...")
        audit_logger.initialize_explainers(
            model,
            "trading_model_v1",
            X_train[:100],  # Background data for explainers
            feature_names
        )
        
        # Generate and log decisions
        logger.info("Generating and logging model decisions...")
        decisions = []
        
        for i in range(len(X_test)):
            # Get model decision with explanations
            decision = audit_logger.explain_decision(
                model,
                "trading_model_v1",
                X_test[i:i+1],
                feature_names
            )
            
            # Add some metadata
            decision.metadata.update({
                'actual_value': float(y_test[i]),
                'timestamp': (
                    datetime.now() - timedelta(minutes=i)
                ).isoformat()
            })
            
            # Log decision
            await audit_logger.log_decision(decision)
            decisions.append(decision)
        
        # Verify log integrity
        logger.info("Verifying log integrity...")
        is_valid = audit_logger.verify_log_integrity()
        logger.info(f"Log integrity check: {'PASSED' if is_valid else 'FAILED'}")
        
        # Analyze model performance
        logger.info("Analyzing model performance...")
        performance = audit_logger.analyze_model_performance(
            "trading_model_v1"
        )
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # 1. Feature Importance
        plot_feature_importance(
            performance['feature_importance'],
            'Average Feature Importance',
            str(results_dir / "feature_importance.png")
        )
        
        # 2. Model Metrics
        plot_model_metrics(
            {'decisions': decisions},
            str(results_dir / "model_metrics.png")
        )
        
        # Print summary
        logger.info("\nAudit Summary:")
        logger.info(f"Total decisions logged: {performance['total_decisions']}")
        logger.info(f"Average confidence: {performance['avg_confidence']:.4f}")
        logger.info(f"Average prediction: {performance['avg_prediction']:.4f}")
        
        logger.info("\nTop 5 most important features:")
        sorted_features = sorted(
            performance['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        logger.info("\nAI audit logging example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in audit example: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run example
    asyncio.run(run_audit_example()) 