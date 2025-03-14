import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import torch

from models.anomaly.base import AnomalyMetrics
from models.anomaly.autoencoder import AutoencoderDetector
from models.anomaly.traditional import IsolationForestDetector, DBSCANDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.anomaly")

def fetch_market_data(
    symbols: list,
    start_date: str,
    end_date: str,
    interval: str = '1h'
) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            data[symbol] = hist
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        raise

def extract_features(data: dict) -> pd.DataFrame:
    """Extract features for anomaly detection"""
    try:
        features = []
        
        for symbol, df in data.items():
            # Calculate returns
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close']).diff()
            
            # Calculate volatility
            df['volatility'] = df['returns'].rolling(window=24).std()
            
            # Calculate volume features
            df['volume_change'] = df['Volume'].pct_change()
            df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=24).mean()
            
            # Calculate price features
            df['price_ma_ratio'] = df['Close'] / df['Close'].rolling(window=24).mean()
            df['high_low_ratio'] = df['High'] / df['Low']
            
            # Select features
            feature_cols = [
                'returns', 'log_returns', 'volatility',
                'volume_change', 'volume_ma_ratio',
                'price_ma_ratio', 'high_low_ratio'
            ]
            
            symbol_features = df[feature_cols].copy()
            symbol_features.columns = [f"{symbol}_{col}" for col in feature_cols]
            features.append(symbol_features)
        
        # Combine features
        features_df = pd.concat(features, axis=1)
        features_df = features_df.dropna()  # Remove rows with missing values
        
        return features_df
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def plot_anomalies(
    data: pd.DataFrame,
    metrics: AnomalyMetrics,
    symbol: str,
    detector_name: str,
    save_dir: Path
) -> None:
    """Plot detected anomalies"""
    try:
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot price with anomalies
        ax1.plot(data.index, data[f"{symbol}_Close"], label='Price')
        anomaly_dates = data.index[metrics.anomaly_labels == 1]
        anomaly_prices = data[f"{symbol}_Close"][metrics.anomaly_labels == 1]
        ax1.scatter(
            anomaly_dates,
            anomaly_prices,
            color='red',
            label='Anomaly',
            alpha=0.6
        )
        ax1.set_title(f"{detector_name}: Price Anomalies for {symbol}")
        ax1.legend()
        ax1.grid(True)
        
        # Plot anomaly scores
        ax2.plot(data.index, metrics.anomaly_scores, label='Anomaly Score')
        ax2.axhline(
            y=metrics.threshold,
            color='r',
            linestyle='--',
            label='Threshold'
        )
        ax2.set_title('Anomaly Scores')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"{symbol}_{detector_name.lower()}_anomalies.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting anomalies: {str(e)}")
        raise

def main():
    try:
        # Set up directories
        results_dir = Path("results/anomaly_detection")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define parameters
        symbols = ['BTC-USD', 'ETH-USD']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Last 30 days
        
        # Fetch market data
        logger.info("Fetching market data...")
        data = fetch_market_data(
            symbols,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='1h'
        )
        
        # Extract features
        logger.info("Extracting features...")
        features_df = extract_features(data)
        
        # Initialize detectors
        detectors = {
            'Autoencoder': AutoencoderDetector(
                hidden_dims=[32, 16],
                latent_dim=8,
                contamination=0.05
            ),
            'IsolationForest': IsolationForestDetector(
                n_estimators=100,
                contamination=0.05
            ),
            'DBSCAN': DBSCANDetector(
                eps=0.5,
                min_samples=5,
                contamination=0.05
            )
        }
        
        # Detect anomalies using each method
        for name, detector in detectors.items():
            logger.info(f"Running {name} detector...")
            
            # Fit and predict
            metrics = detector.fit_predict(features_df)
            
            # Save results
            with open(results_dir / f"{name.lower()}_results.txt", "w") as f:
                f.write(f"{name} Anomaly Detection Results:\n")
                f.write(f"Number of anomalies: {metrics.num_anomalies}\n")
                f.write(f"Contamination: {metrics.contamination:.3f}\n")
                f.write(f"Threshold: {metrics.threshold:.3f}\n")
            
            # Plot results for each symbol
            for symbol in symbols:
                plot_anomalies(
                    data[symbol],
                    metrics,
                    symbol,
                    name,
                    results_dir
                )
        
        logger.info("Anomaly detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        raise

if __name__ == "__main__":
    main() 