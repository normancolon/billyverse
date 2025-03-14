import os
import asyncio
import logging
from pathlib import Path
from web3 import Web3
from datetime import datetime, timedelta
import json
import numpy as np
import tensorflow as tf

from models.defi.contracts import UniswapV2Router, UniswapV2Pair, ERC20Token
from models.defi.yield_farming import YieldFarmingStrategy
from models.defi.arbitrage import ArbitrageStrategy
from infrastructure.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.defi")

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Contract addresses (Ethereum Mainnet)
ADDRESSES = {
    'routers': {
        'uniswap': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',
        'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'
    },
    'tokens': {
        'weth': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
        'usdc': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
        'dai': '0x6B175474E89094C44Da98b954EedeAC495271d0F',
        'usdt': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
    },
    'pools': {
        'uniswap_eth_usdc': '0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc',
        'uniswap_eth_usdt': '0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852',
        'sushi_eth_usdc': '0x397FF1542f962076d0BFE58eA045FfA2d347ACa0',
        'sushi_eth_usdt': '0x06da0fd433C1A5d7a4faa01111c044910A184553'
    }
}

def create_yield_model():
    """Create and train a simple yield prediction model"""
    # Create dummy training data
    n_samples = 1000
    features = np.random.random((n_samples, 4))  # TVL, ratio, reserve0, reserve1
    yields = np.random.random(n_samples) * 0.2  # 0-20% yields
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train model
    model.fit(features, yields, epochs=10, verbose=0)
    
    # Save model
    model_path = results_dir / "yield_model"
    model.save(model_path)
    
    return str(model_path)

def create_arbitrage_model():
    """Create and train a simple arbitrage validation model"""
    # Create dummy training data
    n_samples = 1000
    features = np.random.random((n_samples, 3))  # profit, path_length, amount
    labels = (features[:, 0] > 0.002) & (features[:, 1] < 3)  # Profitable short paths
    
    # Create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train model
    model.fit(features, labels, epochs=10, verbose=0)
    
    # Save model
    model_path = results_dir / "arbitrage_model"
    model.save(model_path)
    
    return str(model_path)

async def run_defi_example():
    """Run DeFi trading example"""
    try:
        logger.info("Starting DeFi trading example...")
        
        # Connect to Ethereum network
        web3 = Web3(Web3.HTTPProvider(os.getenv('ETH_NODE_URL')))
        private_key = os.getenv('PRIVATE_KEY')
        
        if not web3.is_connected():
            raise Exception("Failed to connect to Ethereum network")
        
        # Create and train models
        yield_model_path = create_yield_model()
        arbitrage_model_path = create_arbitrage_model()
        
        # Initialize yield farming strategy
        yield_farmer = YieldFarmingStrategy(
            web3,
            ADDRESSES['routers']['uniswap'],
            private_key,
            yield_model_path
        )
        
        # Initialize arbitrage strategy
        arbitrage_trader = ArbitrageStrategy(
            web3,
            ADDRESSES['routers'],
            private_key,
            arbitrage_model_path
        )
        
        # Define pools to monitor
        pools = [
            {
                'address': ADDRESSES['pools']['uniswap_eth_usdc'],
                'tokens': [ADDRESSES['tokens']['weth'], ADDRESSES['tokens']['usdc']],
                'exchange': 'uniswap'
            },
            {
                'address': ADDRESSES['pools']['uniswap_eth_usdt'],
                'tokens': [ADDRESSES['tokens']['weth'], ADDRESSES['tokens']['usdt']],
                'exchange': 'uniswap'
            },
            {
                'address': ADDRESSES['pools']['sushi_eth_usdc'],
                'tokens': [ADDRESSES['tokens']['weth'], ADDRESSES['tokens']['usdc']],
                'exchange': 'sushiswap'
            },
            {
                'address': ADDRESSES['pools']['sushi_eth_usdt'],
                'tokens': [ADDRESSES['tokens']['weth'], ADDRESSES['tokens']['usdt']],
                'exchange': 'sushiswap'
            }
        ]
        
        # 1. Optimize and execute yield farming strategy
        logger.info("Optimizing yield farming allocation...")
        allocations = await yield_farmer.optimize_allocation(
            pools,
            total_amount=1.0,  # 1 ETH
            min_yield=0.05  # 5% minimum yield
        )
        
        if allocations:
            logger.info("Executing yield farming allocations...")
            tx_hashes = await yield_farmer.execute_allocation(allocations)
            logger.info(f"Yield farming transactions: {tx_hashes}")
        
        # 2. Monitor and execute arbitrage opportunities
        logger.info("Starting arbitrage monitoring...")
        
        # Monitor for 5 minutes
        monitoring_task = asyncio.create_task(
            arbitrage_trader.monitor_opportunities(
                pools,
                start_token=ADDRESSES['tokens']['weth'],
                amount=0.1,  # 0.1 ETH per trade
                min_profit=0.001,  # 0.1% minimum profit
                update_interval=10
            )
        )
        
        try:
            await asyncio.wait_for(monitoring_task, timeout=300)  # 5 minutes
        except asyncio.TimeoutError:
            logger.info("Completed 5 minutes of arbitrage monitoring")
        
        logger.info("DeFi trading example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in DeFi example: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run example
    asyncio.run(run_defi_example()) 