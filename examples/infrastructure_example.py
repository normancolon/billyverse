import os
import logging
import asyncio
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from pathlib import Path

from infrastructure.config import config
from infrastructure.queues import queue_manager
from infrastructure.streaming import stream_manager
from infrastructure.deployment import model_deployer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.infrastructure")

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

async def process_market_data(message):
    """Process market data from Kafka stream"""
    try:
        # Parse message
        data = json.loads(message.value)
        symbol = data['symbol']
        price = float(data['price'])
        timestamp = datetime.fromtimestamp(data['timestamp'])
        
        # Add to queue for processing
        await queue_manager.get_queue('market_data').push({
            'symbol': symbol,
            'price': price,
            'timestamp': timestamp.isoformat()
        })
        
        logger.info(f"Processed market data for {symbol}")
        
    except Exception as e:
        logger.error(f"Error processing market data: {str(e)}")

async def process_signals(message):
    """Process trading signals from queue"""
    try:
        # Parse message
        data = json.loads(message.value)
        
        # Log signal
        logger.info(f"Received signal: {data}")
        
        # Add to execution queue
        await queue_manager.get_queue('execution').push(data)
        
    except Exception as e:
        logger.error(f"Error processing signal: {str(e)}")

def create_dummy_model():
    """Create a simple model for testing deployment"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    model_dir = results_dir / "test_model"
    model.save(model_dir)
    
    return str(model_dir)

async def run_infrastructure_example():
    """Run example demonstrating infrastructure components"""
    try:
        logger.info("Starting infrastructure example...")
        
        # 1. Start Kafka streams
        logger.info("Setting up Kafka streams...")
        await stream_manager.start_producers(['market_data', 'signals'])
        await stream_manager.start_consumers({
            'market_data': process_market_data,
            'signals': process_signals
        })
        
        # 2. Initialize queues
        logger.info("Setting up Redis queues...")
        market_queue = queue_manager.get_queue('market_data')
        signal_queue = queue_manager.get_queue('signals')
        execution_queue = queue_manager.get_queue('execution')
        
        # 3. Deploy model
        logger.info("Deploying test model...")
        model_path = create_dummy_model()
        
        # Deploy to AWS
        aws_endpoint = model_deployer.deploy(
            model_path=model_path,
            model_name="test_model",
            platform="aws",
            instance_count=1
        )
        logger.info(f"Deployed model to AWS endpoint: {aws_endpoint}")
        
        # Deploy to GCP
        gcp_endpoint = model_deployer.deploy(
            model_path=model_path,
            model_name="test_model",
            platform="gcp",
            min_replicas=1
        )
        logger.info(f"Deployed model to GCP endpoint: {gcp_endpoint}")
        
        # 4. Simulate market data
        logger.info("Simulating market data stream...")
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        start_price = {s: 100.0 for s in symbols}
        
        for _ in range(10):
            for symbol in symbols:
                # Generate random price movement
                price = start_price[symbol] * (1 + np.random.normal(0, 0.01))
                start_price[symbol] = price
                
                # Create market data message
                message = {
                    'symbol': symbol,
                    'price': price,
                    'timestamp': datetime.now().timestamp()
                }
                
                # Publish to Kafka
                await stream_manager.get_stream('market_data').publish(
                    json.dumps(message)
                )
                
                await asyncio.sleep(0.1)
        
        # 5. Process queues
        logger.info("Processing queues...")
        async def print_queue_messages(queue_name):
            queue = queue_manager.get_queue(queue_name)
            while True:
                message = await queue.pop()
                if message:
                    logger.info(f"{queue_name} message: {message}")
                await asyncio.sleep(0.1)
        
        # Process queues for 5 seconds
        tasks = [
            print_queue_messages('market_data'),
            print_queue_messages('signals'),
            print_queue_messages('execution')
        ]
        
        await asyncio.gather(
            *tasks,
            return_exceptions=True
        )
        
        logger.info("Infrastructure example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in infrastructure example: {str(e)}")
        raise
    finally:
        # Cleanup
        await stream_manager.close_all()
        await queue_manager.clear_all()

if __name__ == "__main__":
    asyncio.run(run_infrastructure_example()) 