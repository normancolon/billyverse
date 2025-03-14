import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from models.risk.base import BaseRiskAssessor
from models.risk.monte_carlo import MonteCarloRiskAssessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.risk")

def fetch_market_data(
    symbols: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            data[symbol] = hist['Close']
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise

def plot_risk_metrics(
    rolling_metrics: pd.DataFrame,
    save_dir: Path
) -> None:
    """Plot rolling risk metrics"""
    try:
        # Create plots directory
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Plot metrics
        metrics = ['var', 'es', 'sharpe', 'sortino', 'volatility']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        
        for ax, metric in zip(axes, metrics):
            ax.plot(rolling_metrics.index, rolling_metrics[metric])
            ax.set_title(f"Rolling {metric.upper()}")
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "rolling_metrics.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting risk metrics: {str(e)}")
        raise

def plot_monte_carlo(
    simulation_results: 'SimulationResults',
    save_dir: Path
) -> None:
    """Plot Monte Carlo simulation results"""
    try:
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot price paths
        plt.figure(figsize=(12, 6))
        plt.plot(simulation_results.simulated_paths.T, alpha=0.1, color='blue')
        plt.plot(simulation_results.mean_path, 'r--', label='Mean Path')
        plt.plot(simulation_results.worst_case_path, 'r-', label='Worst Case')
        plt.plot(simulation_results.best_case_path, 'g-', label='Best Case')
        plt.title("Monte Carlo Simulation Paths")
        plt.xlabel("Days")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / "monte_carlo_paths.png")
        plt.close()
        
        # Plot VaR and ES distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        sns.histplot(simulation_results.var_estimates, ax=ax1)
        ax1.axvline(
            np.mean(simulation_results.var_estimates),
            color='r',
            linestyle='--',
            label='Mean VaR'
        )
        ax1.set_title("VaR Distribution")
        ax1.legend()
        
        sns.histplot(simulation_results.es_estimates, ax=ax2)
        ax2.axvline(
            np.mean(simulation_results.es_estimates),
            color='r',
            linestyle='--',
            label='Mean ES'
        )
        ax2.set_title("Expected Shortfall Distribution")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "risk_distributions.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting Monte Carlo results: {str(e)}")
        raise

def main():
    try:
        # Set up directories
        results_dir = Path("results/risk_assessment")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define portfolio
        symbols = ['BTC-USD', 'ETH-USD']
        weights = np.array([0.6, 0.4])
        
        # Fetch market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        logger.info("Fetching market data...")
        df = fetch_market_data(symbols, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Calculate portfolio values
        portfolio_values = (df * weights).sum(axis=1).values
        
        # Initialize risk assessors
        base_assessor = BaseRiskAssessor(
            confidence_level=0.95,
            risk_free_rate=0.02
        )
        
        mc_assessor = MonteCarloRiskAssessor(
            n_simulations=10000,
            n_days=252,
            confidence_level=0.95,
            risk_free_rate=0.02
        )
        
        # Calculate basic risk metrics
        logger.info("Calculating basic risk metrics...")
        metrics, rolling_metrics = base_assessor.assess_risk(portfolio_values)
        
        # Run Monte Carlo simulation
        logger.info("Running Monte Carlo simulation...")
        mc_metrics, mc_rolling_metrics, simulation_results = mc_assessor.assess_risk(
            portfolio_values,
            use_t_dist=True
        )
        
        # Save results
        logger.info("Saving results...")
        
        # Save basic metrics
        with open(results_dir / "risk_metrics.txt", "w") as f:
            f.write("Basic Risk Metrics:\n")
            f.write(f"VaR (95%): {metrics.var:.4f}\n")
            f.write(f"Expected Shortfall: {metrics.es:.4f}\n")
            f.write(f"Sharpe Ratio: {metrics.sharpe:.4f}\n")
            f.write(f"Sortino Ratio: {metrics.sortino:.4f}\n")
            f.write(f"Maximum Drawdown: {metrics.max_drawdown:.4f}\n")
            f.write(f"Volatility (annualized): {metrics.volatility:.4f}\n")
            f.write(f"Skewness: {metrics.skewness:.4f}\n")
            f.write(f"Kurtosis: {metrics.kurtosis:.4f}\n")
            
            f.write("\nMonte Carlo Simulation Results:\n")
            for metric, (lower, upper) in simulation_results.confidence_intervals.items():
                f.write(f"{metric} 95% CI: ({lower:.4f}, {upper:.4f})\n")
        
        # Save rolling metrics
        if rolling_metrics is not None:
            rolling_metrics.to_csv(results_dir / "rolling_metrics.csv")
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_risk_metrics(rolling_metrics, results_dir)
        plot_monte_carlo(simulation_results, results_dir)
        
        logger.info("Risk assessment completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}")
        raise

if __name__ == "__main__":
    main() 