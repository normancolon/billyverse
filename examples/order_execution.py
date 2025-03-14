import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import yfinance as yf

from models.execution.base import BaseOrderExecutor
from models.execution.rl_executor import RLOrderExecutor
from models.execution.smart_router import SmartOrderRouter
from models.execution.backtester import ExecutionBacktester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.execution")

def fetch_market_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date, interval='1m')
        
        # Calculate additional metrics
        data['volume'] = data['Volume']
        data['price'] = data['Close']
        data['volatility'] = data['Close'].pct_change().rolling(window=10).std()
        data['spread'] = (data['High'] - data['Low']) / data['Close']
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise

def run_rl_executor_example():
    """Example using RL-based order executor"""
    try:
        # Initialize executor
        executor = RLOrderExecutor(
            state_dim=10,
            action_dim=10,
            hidden_dim=64,
            memory_size=10000,
            batch_size=32,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001
        )
        
        # Execute some orders
        for i in range(5):
            order = executor.execute_order(
                symbol='BTC-USD',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=1.0,
                order_type='market'
            )
            
            if order:
                logger.info(
                    f"Order {order.order_id} executed: "
                    f"Filled {order.filled_quantity:.2f} @ {order.price:.2f}"
                )
                
                metrics = executor.execution_metrics.get(order.order_id)
                if metrics:
                    logger.info(
                        f"Execution metrics - "
                        f"Shortfall: {metrics.implementation_shortfall:.4f}, "
                        f"Impact: {metrics.market_impact:.4f}, "
                        f"Fill Rate: {metrics.fill_rate:.2%}"
                    )
        
    except Exception as e:
        logger.error(f"Error in RL executor example: {str(e)}")
        raise

def run_smart_router_example():
    """Example using smart order router"""
    try:
        # Define venues
        venues = [
            {'id': 'binance', 'name': 'Binance'},
            {'id': 'coinbase', 'name': 'Coinbase'},
            {'id': 'kraken', 'name': 'Kraken'}
        ]
        
        # Initialize router
        router = SmartOrderRouter(
            venues=venues,
            max_venues=3,
            update_interval=60,
            executor_threads=4
        )
        
        # Execute some orders
        for i in range(5):
            order = router.execute_order(
                symbol='BTC-USD',
                side='buy' if i % 2 == 0 else 'sell',
                quantity=1.0,
                order_type='market'
            )
            
            if order:
                logger.info(
                    f"Order {order.order_id} executed across venues: "
                    f"{order.metadata['venues']}"
                )
                
                for detail in order.execution_details:
                    logger.info(
                        f"Execution on {detail['venue_id']}: "
                        f"{detail['quantity']:.2f} @ {detail['price']:.2f}"
                    )
        
    except Exception as e:
        logger.error(f"Error in smart router example: {str(e)}")
        raise

def run_backtest_example():
    """Example using backtester"""
    try:
        # Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        market_data = fetch_market_data(
            symbol='BTC-USD',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        # Define venues for smart router
        venues = [
            {'id': 'binance', 'name': 'Binance'},
            {'id': 'coinbase', 'name': 'Coinbase'},
            {'id': 'kraken', 'name': 'Kraken'}
        ]
        
        # Initialize backtester with smart router
        backtester = ExecutionBacktester(
            market_data=market_data,
            executor_class=SmartOrderRouter,
            executor_params={'venues': venues},
            simulation_params={
                'latency': 50,  # ms
                'spread': 0.001,  # 10 bps
                'market_impact': 0.1,
                'fill_rate': 0.95,
                'transaction_cost': 0.001
            }
        )
        
        # Run backtest
        results = backtester.run_backtest()
        
        # Plot and save results
        save_dir = Path("results/execution_backtest")
        backtester.plot_results(save_dir)
        
        logger.info(
            f"Backtest completed - "
            f"Total Orders: {results.total_orders}, "
            f"Avg Fill Rate: {results.avg_fill_rate:.2%}, "
            f"Avg Slippage: {results.avg_slippage:.4f}"
        )
        
    except Exception as e:
        logger.error(f"Error in backtest example: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Run examples
        logger.info("Running RL Executor example...")
        run_rl_executor_example()
        
        logger.info("\nRunning Smart Router example...")
        run_smart_router_example()
        
        logger.info("\nRunning Backtest example...")
        run_backtest_example()
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise 