import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import ccxt.async_support as ccxt
from typing import Dict, List

from models.hft.rl_strategy import RLStrategy
from models.hft.market_maker import MarketMaker, MarketMakingParams
from models.hft.arbitrage import ArbitrageStrategy, ArbitrageParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.hft")

class MarketDataSimulator:
    """Simulates market data for testing"""
    
    def __init__(
        self,
        symbols: List[str],
        venues: List[Dict],
        base_price: float = 50000,
        volatility: float = 0.0001,
        tick_interval: float = 0.001  # 1ms
    ):
        self.symbols = symbols
        self.venues = venues
        self.base_price = base_price
        self.volatility = volatility
        self.tick_interval = tick_interval
        
        # Initialize prices
        self.prices = {
            venue['id']: {
                symbol: base_price
                for symbol in symbols
            }
            for venue in venues
        }
    
    async def generate_market_data(self) -> Dict:
        """Generate simulated market data"""
        try:
            # Add random walk to prices
            for venue_id in self.prices:
                for symbol in self.symbols:
                    # Generate price movement
                    price_change = np.random.normal(
                        0,
                        self.volatility * self.prices[venue_id][symbol]
                    )
                    self.prices[venue_id][symbol] *= (1 + price_change)
                    
                    # Generate market data
                    price = self.prices[venue_id][symbol]
                    spread = price * 0.0002  # 2bps spread
                    
                    yield {
                        'venue_id': venue_id,
                        'symbol': symbol,
                        'bid': price - spread/2,
                        'ask': price + spread/2,
                        'bid_size': np.random.uniform(0.1, 1.0),
                        'ask_size': np.random.uniform(0.1, 1.0),
                        'last_price': price,
                        'volume': np.random.uniform(1.0, 10.0),
                        'timestamp': datetime.now()
                    }
            
            await asyncio.sleep(self.tick_interval)
            
        except Exception as e:
            logger.error(f"Error generating market data: {str(e)}")
            return {}

async def run_rl_strategy_example():
    """Example using RL-based HFT strategy"""
    try:
        # Initialize strategy
        strategy = RLStrategy(
            symbols=['BTC/USD'],
            position_limits={'BTC/USD': 1.0},
            risk_limits={'max_drawdown': 0.02},
            state_dim=30,
            action_dim=3
        )
        
        # Create market data simulator
        simulator = MarketDataSimulator(
            symbols=['BTC/USD'],
            venues=[{'id': 'exchange1'}],
            base_price=50000,
            volatility=0.0001
        )
        
        # Run simulation
        logger.info("Starting RL strategy simulation...")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < timedelta(minutes=5):
            async for market_data in simulator.generate_market_data():
                strategy.handle_market_data(market_data)
                
                # Log metrics
                if strategy.metrics.total_trades > 0:
                    logger.info(
                        f"RL Strategy Metrics - "
                        f"Trades: {strategy.metrics.total_trades}, "
                        f"PnL: ${strategy.metrics.total_pnl:.2f}, "
                        f"Sharpe: {strategy.metrics.sharpe_ratio:.2f}, "
                        f"Latency: {strategy.metrics.avg_latency:.0f}μs"
                    )
        
        logger.info("RL strategy simulation completed")
        
    except Exception as e:
        logger.error(f"Error in RL strategy example: {str(e)}")

async def run_market_maker_example():
    """Example using market making strategy"""
    try:
        # Initialize strategy
        params = MarketMakingParams(
            spread_multiplier=1.2,
            inventory_target=0.0,
            inventory_limit=1.0,
            min_spread=0.0001,
            max_spread=0.01,
            quote_size=0.1
        )
        
        strategy = MarketMaker(
            symbols=['BTC/USD'],
            position_limits={'BTC/USD': 1.0},
            risk_limits={'max_drawdown': 0.02},
            params=params
        )
        
        # Create market data simulator
        simulator = MarketDataSimulator(
            symbols=['BTC/USD'],
            venues=[{'id': 'exchange1'}],
            base_price=50000,
            volatility=0.0001
        )
        
        # Run simulation
        logger.info("Starting market making simulation...")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < timedelta(minutes=5):
            async for market_data in simulator.generate_market_data():
                strategy.handle_market_data(market_data)
                
                # Log metrics
                if strategy.metrics.total_trades > 0:
                    logger.info(
                        f"Market Making Metrics - "
                        f"Trades: {strategy.metrics.total_trades}, "
                        f"PnL: ${strategy.metrics.total_pnl:.2f}, "
                        f"Fill Rate: {strategy.metrics.fill_rate:.2%}, "
                        f"Latency: {strategy.metrics.avg_latency:.0f}μs"
                    )
        
        logger.info("Market making simulation completed")
        
    except Exception as e:
        logger.error(f"Error in market making example: {str(e)}")

async def run_arbitrage_example():
    """Example using arbitrage strategy"""
    try:
        # Define venues
        venues = [
            {'id': 'binance', 'name': 'Binance', 'trading_fee': 0.001},
            {'id': 'coinbase', 'name': 'Coinbase', 'trading_fee': 0.0015},
            {'id': 'kraken', 'name': 'Kraken', 'trading_fee': 0.002}
        ]
        
        # Initialize strategy
        params = ArbitrageParams(
            min_spread=0.0001,
            min_profit=0.0002,
            max_position=1.0,
            trade_size=0.1,
            execution_timeout=100,
            slippage_tolerance=0.0001
        )
        
        strategy = ArbitrageStrategy(
            symbols=['BTC/USD'],
            position_limits={'BTC/USD': 1.0},
            risk_limits={'max_drawdown': 0.02},
            venues=venues,
            params=params
        )
        
        # Create market data simulator
        simulator = MarketDataSimulator(
            symbols=['BTC/USD'],
            venues=venues,
            base_price=50000,
            volatility=0.0001
        )
        
        # Run simulation
        logger.info("Starting arbitrage simulation...")
        start_time = datetime.now()
        
        while (datetime.now() - start_time) < timedelta(minutes=5):
            async for market_data in simulator.generate_market_data():
                strategy.handle_market_data(market_data)
                
                # Log metrics
                if strategy.metrics.total_trades > 0:
                    logger.info(
                        f"Arbitrage Metrics - "
                        f"Trades: {strategy.metrics.total_trades}, "
                        f"PnL: ${strategy.metrics.total_pnl:.2f}, "
                        f"Win Rate: {strategy.metrics.win_rate:.2%}, "
                        f"Latency: {strategy.metrics.avg_latency:.0f}μs"
                    )
        
        logger.info("Arbitrage simulation completed")
        
    except Exception as e:
        logger.error(f"Error in arbitrage example: {str(e)}")

def plot_results(strategies: List[Dict]):
    """Plot performance comparison of strategies"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn')
        
        # Create results directory
        results_dir = Path("results/hft")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot PnL comparison
        plt.figure(figsize=(12, 6))
        for strategy in strategies:
            plt.plot(
                strategy['pnl_history'],
                label=f"{strategy['name']} (Sharpe: {strategy['metrics'].sharpe_ratio:.2f})"
            )
        plt.title("Strategy PnL Comparison")
        plt.xlabel("Trade")
        plt.ylabel("PnL ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "pnl_comparison.png")
        plt.close()
        
        # Plot latency distribution
        plt.figure(figsize=(12, 6))
        for strategy in strategies:
            sns.histplot(
                strategy['latency_history'],
                label=strategy['name'],
                alpha=0.5,
                bins=50
            )
        plt.title("Strategy Latency Distribution")
        plt.xlabel("Latency (μs)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "latency_distribution.png")
        plt.close()
        
        # Save summary statistics
        with open(results_dir / "performance_summary.txt", "w") as f:
            f.write("HFT Strategy Performance Summary\n")
            f.write("=" * 30 + "\n\n")
            
            for strategy in strategies:
                metrics = strategy['metrics']
                f.write(f"{strategy['name']}:\n")
                f.write(f"- Total Trades: {metrics.total_trades}\n")
                f.write(f"- Total PnL: ${metrics.total_pnl:.2f}\n")
                f.write(f"- Sharpe Ratio: {metrics.sharpe_ratio:.2f}\n")
                f.write(f"- Win Rate: {metrics.win_rate:.2%}\n")
                f.write(f"- Avg Latency: {metrics.avg_latency:.0f}μs\n")
                f.write(f"- Max Drawdown: {metrics.max_drawdown:.2%}\n\n")
        
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")

async def main():
    """Run all HFT strategy examples"""
    try:
        # Run strategies
        await asyncio.gather(
            run_rl_strategy_example(),
            run_market_maker_example(),
            run_arbitrage_example()
        )
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    # Run event loop
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main()) 