from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import logging
from web3 import Web3
import networkx as nx
from dataclasses import dataclass

from models.defi.contracts import UniswapV2Router, UniswapV2Pair, ERC20Token
from infrastructure.config import config

logger = logging.getLogger("billieverse.defi.arbitrage")

@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity"""
    path: List[str]  # List of token addresses in the path
    exchanges: List[str]  # List of exchange addresses
    profit: float  # Expected profit in ETH
    input_amount: float  # Required input amount in ETH
    min_output: float  # Minimum output amount with slippage
    deadline: int  # Transaction deadline

class ArbitrageStrategy:
    """AI-driven arbitrage trading strategy"""
    
    def __init__(
        self,
        web3: Web3,
        router_addresses: Dict[str, str],
        private_key: str,
        model_path: Optional[str] = None
    ):
        self.web3 = web3
        
        # Initialize routers for each exchange
        self.routers = {
            name: UniswapV2Router(web3, addr, private_key)
            for name, addr in router_addresses.items()
        }
        
        # Load AI model if provided
        self.model = None
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        
        # Initialize graph for path finding
        self.graph = nx.DiGraph()
    
    def _create_model(
        self,
        input_dim: int,
        lstm_units: int = 64
    ) -> tf.keras.Model:
        """Create LSTM model for opportunity detection"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                lstm_units,
                input_shape=(None, input_dim),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(lstm_units // 2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def _get_reserves(
        self,
        pool_address: str
    ) -> Tuple[int, int, int]:
        """Get pool reserves"""
        pool = UniswapV2Pair(self.web3, pool_address)
        return await pool.get_reserves()
    
    async def _calculate_output_amount(
        self,
        amount_in: int,
        path: List[str],
        router: UniswapV2Router
    ) -> int:
        """Calculate output amount for a path"""
        try:
            amounts = await router.get_amounts_out(amount_in, path)
            return amounts[-1]
        except Exception as e:
            logger.error(f"Error calculating output: {str(e)}")
            return 0
    
    async def update_graph(
        self,
        pools: List[Dict]
    ):
        """Update arbitrage graph with current pool states"""
        try:
            # Clear existing edges
            self.graph.clear_edges()
            
            # Add edges for each pool
            for pool in pools:
                reserve0, reserve1, _ = await self._get_reserves(
                    pool['address']
                )
                
                # Add bidirectional edges
                self.graph.add_edge(
                    pool['tokens'][0],
                    pool['tokens'][1],
                    weight=reserve0/reserve1,
                    exchange=pool['exchange'],
                    pool=pool['address']
                )
                self.graph.add_edge(
                    pool['tokens'][1],
                    pool['tokens'][0],
                    weight=reserve1/reserve0,
                    exchange=pool['exchange'],
                    pool=pool['address']
                )
            
            logger.info(f"Updated graph with {len(pools)} pools")
            
        except Exception as e:
            logger.error(f"Error updating graph: {str(e)}")
            raise
    
    def _find_arbitrage_paths(
        self,
        start_token: str,
        min_profit: float = 1.001,  # 0.1% minimum profit
        max_hops: int = 4
    ) -> List[List[str]]:
        """Find potential arbitrage paths"""
        paths = []
        
        def dfs(current: str, path: List[str], product: float):
            if len(path) > 1 and current == start_token:
                if product > min_profit:
                    paths.append(path)
                return
                
            if len(path) >= max_hops:
                return
                
            for _, next_token, data in self.graph.edges(current, data=True):
                if next_token not in path or next_token == start_token:
                    dfs(
                        next_token,
                        path + [next_token],
                        product * data['weight']
                    )
        
        dfs(start_token, [start_token], 1.0)
        return paths
    
    async def find_opportunities(
        self,
        start_token: str,
        amount: float,
        min_profit: float = 0.001  # 0.1% minimum profit
    ) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities"""
        try:
            opportunities = []
            paths = self._find_arbitrage_paths(
                start_token,
                min_profit + 1.0
            )
            
            amount_wei = int(amount * 1e18)  # Convert to wei
            
            for path in paths:
                # Get exchanges for each hop
                exchanges = [
                    self.graph[path[i]][path[i+1]]['exchange']
                    for i in range(len(path)-1)
                ]
                
                # Calculate actual output
                current_amount = amount_wei
                routers_used = []
                
                for i in range(len(path)-1):
                    router = self.routers[exchanges[i]]
                    routers_used.append(router)
                    
                    output = await self._calculate_output_amount(
                        current_amount,
                        [path[i], path[i+1]],
                        router
                    )
                    
                    if output == 0:
                        break
                    current_amount = output
                
                # Calculate profit
                profit = (current_amount - amount_wei) / 1e18
                
                if profit > min_profit:
                    # Create opportunity
                    deadline = int(
                        (datetime.now() + timedelta(minutes=10)).timestamp()
                    )
                    
                    opportunities.append(
                        ArbitrageOpportunity(
                            path=path,
                            exchanges=exchanges,
                            profit=profit,
                            input_amount=amount,
                            min_output=int(amount_wei * (1 + min_profit)),
                            deadline=deadline
                        )
                    )
            
            # Sort by profit
            opportunities.sort(key=lambda x: x.profit, reverse=True)
            
            logger.info(
                f"Found {len(opportunities)} profitable arbitrage opportunities"
            )
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding opportunities: {str(e)}")
            raise
    
    async def execute_arbitrage(
        self,
        opportunity: ArbitrageOpportunity
    ) -> str:
        """Execute arbitrage trade"""
        try:
            # Get first router
            router = self.routers[opportunity.exchanges[0]]
            
            # Execute initial swap
            tx_hash = await router.swap_exact_eth_for_tokens(
                int(opportunity.input_amount * 1e18),
                opportunity.min_output,
                opportunity.path,
                router.account.address,
                opportunity.deadline
            )
            
            logger.info(
                f"Executed arbitrage trade: {tx_hash.hex()}"
            )
            return tx_hash
            
        except Exception as e:
            logger.error(f"Error executing arbitrage: {str(e)}")
            raise
    
    async def monitor_opportunities(
        self,
        pools: List[Dict],
        start_token: str,
        amount: float,
        min_profit: float = 0.001,
        update_interval: int = 10
    ):
        """Monitor for arbitrage opportunities"""
        try:
            while True:
                # Update graph
                await self.update_graph(pools)
                
                # Find opportunities
                opportunities = await self.find_opportunities(
                    start_token,
                    amount,
                    min_profit
                )
                
                # Execute best opportunity
                if opportunities:
                    best = opportunities[0]
                    if self.model:
                        # Use AI to validate opportunity
                        features = [
                            best.profit,
                            len(best.path),
                            best.input_amount
                        ]
                        
                        confidence = self.model.predict(
                            np.array([features])
                        )[0][0]
                        
                        if confidence > 0.8:  # 80% confidence threshold
                            await self.execute_arbitrage(best)
                    else:
                        # Execute without AI validation
                        await self.execute_arbitrage(best)
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except Exception as e:
            logger.error(f"Error monitoring opportunities: {str(e)}")
            raise 