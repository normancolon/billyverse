import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from models.portfolio.base import PortfolioOptimizer, OptimizationConstraints
from models.portfolio.sector_rotation import SectorRotationStrategy
from models.portfolio.risk import RiskManager, HedgeParameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.portfolio")

def fetch_market_data(
    symbols: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        # Download data
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Calculate returns
        returns = data['Adj Close'].pct_change().dropna()
        
        return returns
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None

def run_mpt_example():
    """Example using basic MPT optimization"""
    try:
        # Define portfolio universe
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'JPM', 'BAC', 'GS', 'MS', 'WFC',
            'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY',
            'XOM', 'CVX', 'COP', 'EOG', 'SLB'
        ]
        
        # Fetch data
        returns = fetch_market_data(
            symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None:
            return
        
        # Set optimization constraints
        constraints = OptimizationConstraints(
            min_weight=0.02,
            max_weight=0.15,
            min_assets=8,
            max_sector_weight=0.4
        )
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            returns=returns,
            constraints=constraints
        )
        
        # Run optimization
        weights, metrics = optimizer.optimize_portfolio(risk_aversion=2.0)
        
        if weights and metrics:
            logger.info("MPT Optimization Results:")
            logger.info(f"Expected Return: {metrics.returns:.2%}")
            logger.info(f"Volatility: {metrics.volatility:.2%}")
            logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"Max Drawdown: {metrics.max_drawdown:.2%}")
            
            # Plot weights
            plt.figure(figsize=(12, 6))
            plt.bar(weights.keys(), weights.values())
            plt.title("Portfolio Weights (MPT)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("results/mpt_weights.png")
            plt.close()
        
    except Exception as e:
        logger.error(f"Error in MPT example: {str(e)}")

def run_sector_rotation_example():
    """Example using sector rotation strategy"""
    try:
        # Define sector ETFs
        sector_etfs = {
            'XLK': 'Technology',
            'XLF': 'Financials',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLB': 'Materials',
            'XLU': 'Utilities',
            'XLRE': 'Real Estate'
        }
        
        # Define stocks in each sector
        stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
            'Financials': ['JPM', 'BAC', 'GS', 'MS', 'WFC'],
            'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'BMY'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        
        # Create sector mapping
        sector_mapping = {
            stock: sector
            for sector, stock_list in stocks.items()
            for stock in stock_list
        }
        
        # Fetch stock data
        all_stocks = [s for sl in stocks.values() for s in sl]
        returns = fetch_market_data(
            all_stocks,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None:
            return
        
        # Fetch sector ETF data for hedging
        hedge_universe = fetch_market_data(
            list(sector_etfs.keys()),
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # Initialize strategy
        strategy = SectorRotationStrategy(
            returns=returns,
            sector_mapping=sector_mapping,
            model_params={
                'lookback_periods': 12,
                'prediction_horizon': 3
            }
        )
        
        # Train model
        train_scores = strategy.train_model(validation_split=0.2)
        logger.info("Sector Model Training Scores:")
        logger.info(f"RF Score: {train_scores.get('rf_score', 0):.3f}")
        logger.info(f"NN Score: {train_scores.get('nn_score', 0):.3f}")
        
        # Run optimization
        weights, metrics = strategy.optimize_portfolio(risk_aversion=2.0)
        
        if weights and metrics:
            logger.info("\nSector Rotation Results:")
            logger.info(f"Expected Return: {metrics.returns:.2%}")
            logger.info(f"Volatility: {metrics.volatility:.2%}")
            logger.info(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            
            # Plot sector weights
            sector_weights = strategy.sector_weights
            plt.figure(figsize=(12, 6))
            plt.bar(sector_weights.keys(), sector_weights.values())
            plt.title("Sector Allocations")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("results/sector_weights.png")
            plt.close()
        
    except Exception as e:
        logger.error(f"Error in sector rotation example: {str(e)}")

def run_risk_management_example():
    """Example using risk management and hedging"""
    try:
        # Define portfolio
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
            'JPM', 'BAC', 'GS', 'MS', 'WFC'
        ]
        
        # Fetch data
        returns = fetch_market_data(
            symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None:
            return
        
        # Fetch benchmark data
        benchmark = fetch_market_data(
            ['SPY'],
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if benchmark is None:
            return
        
        # Fetch hedge universe data
        hedge_symbols = ['SH', 'PSQ', 'DOG', 'VIXY', 'GLD', 'TLT']
        hedge_universe = fetch_market_data(
            hedge_symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # Initialize risk manager
        risk_manager = RiskManager(
            returns=returns,
            benchmark_returns=benchmark['SPY'],
            hedge_universe=hedge_universe,
            hedge_params=HedgeParameters(
                hedge_ratio=0.3,
                min_correlation=0.6
            )
        )
        
        # Create sample portfolio
        weights = {
            symbol: 1/len(symbols)
            for symbol in symbols
        }
        
        # Get risk metrics
        metrics = risk_manager.calculate_risk_metrics(weights)
        if metrics:
            logger.info("\nInitial Risk Metrics:")
            logger.info(f"VaR (95%): {metrics.var_95:.2%}")
            logger.info(f"Beta: {metrics.beta:.2f}")
            logger.info(f"Correlation: {metrics.correlation:.2f}")
            
            # Check risk limits
            violations = risk_manager.check_risk_limits(weights)
            logger.info("\nRisk Limit Violations:")
            for limit, violated in violations.items():
                logger.info(f"{limit}: {'Violated' if violated else 'OK'}")
        
        # Update hedges
        hedges = risk_manager.update_hedges(weights)
        if hedges:
            logger.info("\nSelected Hedges:")
            for instrument, ratio in hedges.items():
                logger.info(f"{instrument}: {ratio:.2%}")
            
            # Get updated risk metrics with hedges
            hedged_metrics = risk_manager.calculate_risk_metrics(
                weights,
                include_hedges=True
            )
            
            if hedged_metrics:
                logger.info("\nHedged Risk Metrics:")
                logger.info(f"VaR (95%): {hedged_metrics.var_95:.2%}")
                logger.info(f"Beta: {hedged_metrics.beta:.2f}")
                logger.info(f"Correlation: {hedged_metrics.correlation:.2f}")
        
        # Generate risk report
        report = risk_manager.get_risk_report(weights)
        if report:
            # Plot risk metrics
            plt.figure(figsize=(12, 6))
            metrics = ['var_95', 'beta', 'correlation', 'tracking_error']
            values = [report[m] for m in metrics]
            plt.bar(metrics, values)
            plt.title("Portfolio Risk Metrics")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("results/risk_metrics.png")
            plt.close()
        
    except Exception as e:
        logger.error(f"Error in risk management example: {str(e)}")

def main():
    """Run all portfolio optimization examples"""
    try:
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Run examples
        logger.info("Running MPT optimization example...")
        run_mpt_example()
        
        logger.info("\nRunning sector rotation example...")
        run_sector_rotation_example()
        
        logger.info("\nRunning risk management example...")
        run_risk_management_example()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 