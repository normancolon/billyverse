from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
from web3 import Web3

from models.defi.contracts import UniswapV2Router, UniswapV2Pair, ERC20Token
from infrastructure.config import config

logger = logging.getLogger("billieverse.defi.yield_farming")

class YieldFarmingStrategy:
    """AI-driven yield farming strategy"""
    
    def __init__(
        self,
        web3: Web3,
        router_address: str,
        private_key: str,
        model_path: Optional[str] = None
    ):
        self.web3 = web3
        self.router = UniswapV2Router(web3, router_address, private_key)
        
        # Load AI model if provided
        self.model = None
        self.scaler = StandardScaler()
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
    
    def _create_model(
        self,
        input_dim: int,
        lstm_units: int = 64
    ) -> tf.keras.Model:
        """Create LSTM model for yield prediction"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                lstm_units,
                input_shape=(None, input_dim),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(
                lstm_units // 2,
                return_sequences=False
            ),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    async def _get_pool_features(
        self,
        pool_address: str,
        token_addresses: List[str]
    ) -> Dict:
        """Get pool features for prediction"""
        # Get pool data
        pool = UniswapV2Pair(self.web3, pool_address)
        reserve0, reserve1, _ = await pool.get_reserves()
        
        # Get token data
        tokens = [ERC20Token(self.web3, addr) for addr in token_addresses]
        decimals = [await token.decimals() for token in tokens]
        
        # Calculate features
        tvl = (reserve0 / 10**decimals[0]) * (reserve1 / 10**decimals[1])
        ratio = reserve0 / reserve1
        
        return {
            'tvl': tvl,
            'ratio': ratio,
            'reserve0': reserve0 / 10**decimals[0],
            'reserve1': reserve1 / 10**decimals[1]
        }
    
    async def train(
        self,
        pool_addresses: List[str],
        token_addresses: List[List[str]],
        historical_yields: List[float],
        epochs: int = 100,
        validation_split: float = 0.2
    ):
        """Train yield prediction model"""
        try:
            # Gather training data
            features = []
            for pool_addr, token_addrs in zip(pool_addresses, token_addresses):
                pool_features = await self._get_pool_features(
                    pool_addr,
                    token_addrs
                )
                features.append(list(pool_features.values()))
            
            # Prepare data
            X = np.array(features)
            y = np.array(historical_yields)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Create and train model
            if not self.model:
                self.model = self._create_model(X.shape[1])
            
            # Train
            history = self.model.fit(
                X_scaled,
                y,
                epochs=epochs,
                validation_split=validation_split,
                verbose=1
            )
            
            logger.info(
                f"Model trained. Final loss: {history.history['loss'][-1]:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    async def predict_yield(
        self,
        pool_address: str,
        token_addresses: List[str]
    ) -> float:
        """Predict yield for pool"""
        try:
            if not self.model:
                raise ValueError("Model not trained")
            
            # Get features
            features = await self._get_pool_features(
                pool_address,
                token_addresses
            )
            
            # Scale and predict
            X = np.array([list(features.values())])
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0][0]
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting yield: {str(e)}")
            raise
    
    async def optimize_allocation(
        self,
        pools: List[Dict],
        total_amount: float,
        min_yield: float = 0.0
    ) -> List[Dict]:
        """Optimize allocation across pools"""
        try:
            allocations = []
            remaining = total_amount
            
            # Get predicted yields
            pool_yields = []
            for pool in pools:
                predicted_yield = await self.predict_yield(
                    pool['address'],
                    pool['tokens']
                )
                pool_yields.append({
                    'pool': pool,
                    'yield': predicted_yield
                })
            
            # Sort by yield
            pool_yields.sort(key=lambda x: x['yield'], reverse=True)
            
            # Allocate based on yields
            for pool_yield in pool_yields:
                if pool_yield['yield'] < min_yield:
                    break
                    
                # Calculate allocation
                weight = pool_yield['yield'] / sum(
                    py['yield'] for py in pool_yields
                )
                amount = total_amount * weight
                
                if amount > 0:
                    allocations.append({
                        'pool': pool_yield['pool'],
                        'amount': amount,
                        'predicted_yield': pool_yield['yield']
                    })
                    remaining -= amount
            
            logger.info(
                f"Optimized allocation across {len(allocations)} pools"
            )
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing allocation: {str(e)}")
            raise
    
    async def execute_allocation(
        self,
        allocations: List[Dict]
    ) -> List[str]:
        """Execute optimized allocation"""
        try:
            tx_hashes = []
            
            for allocation in allocations:
                pool = allocation['pool']
                amount = allocation['amount']
                
                # Add liquidity
                deadline = int(
                    (datetime.now() + timedelta(minutes=10)).timestamp()
                )
                
                # Calculate minimum amounts
                amounts_out = await self.router.get_amounts_out(
                    int(amount * 1e18),  # Convert to wei
                    [self.web3.eth.contract.address] + pool['tokens']
                )
                
                min_amount = int(amounts_out[-1] * 0.99)  # 1% slippage
                
                # Execute swap and add liquidity
                tx_hash = await self.router.swap_exact_eth_for_tokens(
                    int(amount * 1e18),
                    min_amount,
                    [self.web3.eth.contract.address] + pool['tokens'],
                    self.router.account.address,
                    deadline
                )
                
                tx_hashes.append(tx_hash)
                logger.info(
                    f"Added liquidity to pool {pool['address']}: {amount} ETH"
                )
            
            return tx_hashes
            
        except Exception as e:
            logger.error(f"Error executing allocation: {str(e)}")
            raise 