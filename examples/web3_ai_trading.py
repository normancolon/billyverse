import os
import asyncio
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from web3 import Web3
from datetime import datetime, timedelta
from dotenv import load_dotenv

from models.web3.ai_contracts import Web3AIDeployer
from models.quantum.trading import QuantumTrading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.web3_ai")

# Create results directory
results_dir = Path("results/web3_ai")
results_dir.mkdir(exist_ok=True, parents=True)

def create_sample_model() -> tf.keras.Model:
    """Create a sample trading model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sample_data(n_samples: int = 1000) -> tuple:
    """Generate sample training data"""
    X = np.random.normal(0, 1, (n_samples, 10))
    y = (np.mean(X, axis=1) > 0).astype(float)
    
    return X, y

async def run_web3_ai_example():
    """Run Web3 AI trading example"""
    try:
        logger.info("Starting Web3 AI trading example...")
        
        # Load environment variables
        eth_node_url = os.getenv('ETH_NODE_URL')
        private_key = os.getenv('PRIVATE_KEY')
        chain_id = int(os.getenv('CHAIN_ID', '1'))  # Default to mainnet
        
        if not all([eth_node_url, private_key]):
            raise ValueError(
                "Missing required environment variables. "
                "Please set ETH_NODE_URL and PRIVATE_KEY"
            )
        
        # Connect to Ethereum network
        web3 = Web3(Web3.HTTPProvider(eth_node_url))
        if not web3.is_connected():
            raise Exception("Failed to connect to Ethereum network")
        
        # Initialize Web3 AI deployer
        deployer = Web3AIDeployer(
            web3,
            private_key,
            chain_id,
            gas_price_gwei=50
        )
        
        # Create and train model
        logger.info("Creating and training model...")
        model = create_sample_model()
        X_train, y_train = generate_sample_data()
        model.fit(X_train, y_train, epochs=5, verbose=0)
        
        # Deploy model to blockchain
        logger.info("Deploying model to blockchain...")
        model_deployment = await deployer.deploy_model(
            model,
            "trading_model_v1"
        )
        
        logger.info(f"Model deployed at: {model_deployment.contract_address}")
        logger.info(f"Model hash: {model_deployment.model_hash}")
        
        # Deploy trading agent
        logger.info("Deploying trading agent...")
        agent_deployment = await deployer.deploy_trading_agent(
            "trading_model_v1",
            min_confidence=0.8,
            max_trade_amount=0.1  # Max 0.1 ETH per trade
        )
        
        logger.info(f"Trading agent deployed at: {agent_deployment.contract_address}")
        
        # Generate some predictions and record them
        logger.info("Recording predictions on blockchain...")
        X_test = np.random.normal(0, 1, (5, 10))
        predictions = model.predict(X_test)
        
        for i, (x, pred) in enumerate(zip(X_test, predictions)):
            confidence = float(abs(pred - 0.5) * 2)  # Scale to [0, 1]
            
            await deployer.record_prediction(
                "trading_model_v1",
                x,
                pred,
                confidence
            )
            
            logger.info(f"Prediction {i+1} recorded with confidence: {confidence:.4f}")
        
        # Execute some sample trades
        logger.info("\nExecuting sample trades...")
        
        # Sample token addresses (replace with actual addresses in production)
        tokens = [
            "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
            "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",  # AAVE
            "0x514910771AF9Ca656af840dff83E8264EcF986CA"   # LINK
        ]
        
        for i, (x, pred) in enumerate(zip(X_test[:3], predictions[:3])):
            token = tokens[i]
            is_buy = float(pred) > 0.5
            amount = 0.01  # 0.01 ETH
            price = 1000.0  # Example price in USD
            
            receipt = await deployer.execute_trade(
                "trading_model_v1",
                token,
                is_buy,
                amount,
                price,
                x,
                pred
            )
            
            logger.info(
                f"Trade {i+1} executed: "
                f"{'Bought' if is_buy else 'Sold'} {amount} ETH worth of tokens "
                f"at ${price}"
            )
            logger.info(f"Transaction hash: {receipt['transactionHash'].hex()}")
        
        # Save deployment information
        deployment_info = {
            'model_contract': model_deployment.contract_address,
            'agent_contract': agent_deployment.contract_address,
            'model_hash': model_deployment.model_hash,
            'chain_id': chain_id,
            'timestamp': datetime.now().isoformat()
        }
        
        deployment_file = results_dir / "deployment_info.json"
        import json
        with open(deployment_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info(f"\nDeployment information saved to {deployment_file}")
        logger.info("\nWeb3 AI trading example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in Web3 AI example: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run example
    asyncio.run(run_web3_ai_example()) 