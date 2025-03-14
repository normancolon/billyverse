import os
import asyncio
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from dotenv import load_dotenv

from models.quantum.trading import QuantumTrading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.quantum")

# Create results directory
results_dir = Path("results/quantum")
results_dir.mkdir(exist_ok=True, parents=True)

def generate_sample_market_data(
    n_assets: int = 5,
    n_days: int = 252  # One trading year
) -> pd.DataFrame:
    """Generate sample market data"""
    # Generate random returns
    returns = np.random.normal(0.001, 0.02, (n_days, n_assets))
    
    # Create price series
    prices = np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrame
    dates = pd.date_range(
        end=datetime.now(),
        periods=n_days,
        freq='B'
    )
    
    assets = [f'ASSET_{i+1}' for i in range(n_assets)]
    
    return pd.DataFrame(
        prices,
        index=dates,
        columns=assets
    )

def plot_portfolio_weights(
    weights: np.ndarray,
    asset_names: list,
    save_path: str
):
    """Plot portfolio allocation"""
    plt.figure(figsize=(10, 6))
    plt.pie(
        weights,
        labels=asset_names,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title('Quantum-Optimized Portfolio Allocation')
    plt.axis('equal')
    plt.savefig(save_path)
    plt.close()

def plot_risk_metrics(
    risk_metrics: dict,
    save_path: str
):
    """Plot risk metrics"""
    plt.figure(figsize=(10, 6))
    
    metrics = {
        'VaR (95%)': risk_metrics['var_95'],
        'CVaR (95%)': risk_metrics['cvar_95'],
        'Max Loss': risk_metrics['max_loss'],
        'Volatility': risk_metrics['volatility']
    }
    
    plt.bar(metrics.keys(), metrics.values())
    plt.title('Portfolio Risk Metrics')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_quantum_predictions(
    predictions: list,
    save_path: str
):
    """Plot quantum ML predictions"""
    plt.figure(figsize=(12, 6))
    
    # Plot predictions
    plt.subplot(1, 2, 1)
    values = [p.prediction for p in predictions]
    plt.hist(values, bins=20)
    plt.title('Prediction Distribution')
    plt.xlabel('Predicted Value')
    plt.ylabel('Count')
    
    # Plot confidence
    plt.subplot(1, 2, 2)
    confidence = [p.confidence for p in predictions]
    plt.hist(confidence, bins=20)
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

async def run_quantum_example():
    """Run quantum trading example"""
    try:
        logger.info("Starting quantum trading example...")
        
        # Initialize quantum trading system
        quantum_trader = QuantumTrading(
            backend='qiskit',  # Using Qiskit simulator
            n_qubits=10
        )
        
        # Generate sample market data
        logger.info("Generating sample market data...")
        market_data = generate_sample_market_data()
        
        # Calculate returns and covariance
        returns = market_data.pct_change().dropna().mean()
        covariance = market_data.pct_change().dropna().cov()
        
        # Optimize portfolio
        logger.info("Optimizing portfolio using quantum computing...")
        portfolio = quantum_trader.optimize_portfolio(
            returns.values,
            covariance.values,
            risk_tolerance=0.5,
            n_shots=1000
        )
        
        # Plot portfolio allocation
        plot_portfolio_weights(
            portfolio.weights,
            market_data.columns,
            str(results_dir / "portfolio_allocation.png")
        )
        
        logger.info("\nPortfolio Metrics:")
        logger.info(f"Expected Return: {portfolio.expected_return:.4f}")
        logger.info(f"Risk: {portfolio.risk:.4f}")
        logger.info(f"Sharpe Ratio: {portfolio.sharpe_ratio:.4f}")
        
        # Assess risk
        logger.info("\nAssessing portfolio risk...")
        risk_metrics = quantum_trader.assess_risk(
            portfolio,
            market_data.pct_change().dropna(),
            n_scenarios=1000
        )
        
        # Plot risk metrics
        plot_risk_metrics(
            risk_metrics,
            str(results_dir / "risk_metrics.png")
        )
        
        logger.info("\nRisk Metrics:")
        logger.info(f"VaR (95%): {risk_metrics['var_95']:.4f}")
        logger.info(f"CVaR (95%): {risk_metrics['cvar_95']:.4f}")
        logger.info(f"Max Loss: {risk_metrics['max_loss']:.4f}")
        logger.info(f"Volatility: {risk_metrics['volatility']:.4f}")
        
        # Prepare data for quantum ML
        logger.info("\nTraining quantum ML models...")
        X = market_data.pct_change().dropna().values[:-1]
        y = (market_data.pct_change().dropna().mean(axis=1).values[1:] > 0).astype(float)
        
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train quantum ML models
        quantum_trader.train_quantum_ml(
            X_train,
            y_train,
            n_layers=2,
            n_shots=1000
        )
        
        # Make predictions
        logger.info("\nMaking quantum predictions...")
        predictions = []
        for features in X_test:
            prediction = quantum_trader.predict(
                features,
                method='vqc'
            )
            predictions.append(prediction)
        
        # Plot predictions
        plot_quantum_predictions(
            predictions,
            str(results_dir / "quantum_predictions.png")
        )
        
        # Calculate prediction metrics
        y_pred = [p.prediction for p in predictions]
        accuracy = np.mean(
            (np.array(y_pred) > 0.5) == y_test
        )
        avg_confidence = np.mean([p.confidence for p in predictions])
        avg_circuit_depth = np.mean([p.circuit_depth for p in predictions])
        avg_execution_time = np.mean([p.execution_time for p in predictions])
        
        logger.info("\nQuantum ML Metrics:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Average Confidence: {avg_confidence:.4f}")
        logger.info(f"Average Circuit Depth: {avg_circuit_depth:.1f}")
        logger.info(f"Average Execution Time: {avg_execution_time:.4f}s")
        
        logger.info("\nQuantum trading example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in quantum example: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run example
    asyncio.run(run_quantum_example()) 