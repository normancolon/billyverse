import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("billieverse.models.anomaly")

@dataclass
class AnomalyMetrics:
    """Container for anomaly detection metrics"""
    anomaly_scores: np.ndarray  # Anomaly scores for each data point
    anomaly_labels: np.ndarray  # Binary labels (0: normal, 1: anomaly)
    threshold: float  # Threshold used for classification
    num_anomalies: int  # Number of detected anomalies
    contamination: float  # Proportion of anomalies in the data

class BaseAnomalyDetector(ABC):
    """Base class for anomaly detection"""
    
    def __init__(
        self,
        contamination: float = 0.1,  # Expected proportion of anomalies
        random_state: Optional[int] = None,
        n_jobs: int = -1  # Use all CPU cores
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.scaler = StandardScaler()
        self.logger = logger
        
        if not 0 < contamination < 1:
            raise ValueError("Contamination must be between 0 and 1")
    
    def preprocess_data(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        scale: bool = True
    ) -> np.ndarray:
        """Preprocess input data"""
        try:
            # Convert to numpy array if needed
            if isinstance(data, pd.DataFrame):
                data = data.values
            
            # Handle missing values
            if np.isnan(data).any():
                self.logger.warning("Missing values detected, filling with mean")
                data = np.nan_to_num(data, nan=np.nanmean(data))
            
            # Scale data if requested
            if scale:
                data = self.scaler.fit_transform(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def calculate_threshold(
        self,
        scores: np.ndarray,
        method: str = 'percentile'
    ) -> float:
        """Calculate anomaly threshold"""
        try:
            if method == 'percentile':
                return np.percentile(scores, (1 - self.contamination) * 100)
            elif method == 'std':
                return np.mean(scores) + 2 * np.std(scores)
            else:
                raise ValueError(f"Unsupported threshold method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error calculating threshold: {str(e)}")
            raise
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'BaseAnomalyDetector':
        """Fit the anomaly detector"""
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> AnomalyMetrics:
        """Predict anomalies"""
        pass
    
    def fit_predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> AnomalyMetrics:
        """Fit and predict in one step"""
        try:
            self.fit(X)
            return self.predict(X)
            
        except Exception as e:
            self.logger.error(f"Error in fit_predict: {str(e)}")
            raise
    
    @abstractmethod
    def score_samples(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute anomaly scores"""
        pass
    
    def save_model(self, path: str) -> None:
        """Save model to disk"""
        try:
            import joblib
            joblib.dump(self, path)
            self.logger.info(f"Model saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, path: str) -> 'BaseAnomalyDetector':
        """Load model from disk"""
        try:
            import joblib
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 