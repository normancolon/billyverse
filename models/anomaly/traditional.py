import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import logging
from models.anomaly.base import BaseAnomalyDetector, AnomalyMetrics

logger = logging.getLogger("billieverse.models.anomaly.traditional")

class IsolationForestDetector(BaseAnomalyDetector):
    """Isolation Forest anomaly detector"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = 'auto',
        contamination: float = 0.1,
        max_features: float = 1.0,
        bootstrap: bool = False,
        n_jobs: int = -1,
        random_state: Optional[int] = None
    ):
        super().__init__(
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state
        )
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> 'IsolationForestDetector':
        """Fit the Isolation Forest"""
        try:
            X = self.preprocess_data(X)
            self.model.fit(X)
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting Isolation Forest: {str(e)}")
            raise
    
    def score_samples(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute anomaly scores"""
        try:
            X = self.preprocess_data(X)
            # Convert decision function to positive anomaly scores
            return -self.model.score_samples(X)
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> AnomalyMetrics:
        """Predict anomalies"""
        try:
            X = self.preprocess_data(X)
            scores = self.score_samples(X)
            
            # Convert predictions from {1: inlier, -1: outlier} to {0: normal, 1: anomaly}
            labels = (self.model.predict(X) == -1).astype(int)
            num_anomalies = np.sum(labels)
            
            return AnomalyMetrics(
                anomaly_scores=scores,
                anomaly_labels=labels,
                threshold=self.calculate_threshold(scores),
                num_anomalies=num_anomalies,
                contamination=self.contamination
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting anomalies: {str(e)}")
            raise

class DBSCANDetector(BaseAnomalyDetector):
    """DBSCAN-based anomaly detector"""
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
        contamination: float = 0.1,
        n_jobs: int = -1
    ):
        super().__init__(
            contamination=contamination,
            n_jobs=n_jobs
        )
        
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            n_jobs=n_jobs
        )
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> 'DBSCANDetector':
        """Fit DBSCAN"""
        try:
            X = self.preprocess_data(X)
            self.model.fit(X)
            return self
            
        except Exception as e:
            self.logger.error(f"Error fitting DBSCAN: {str(e)}")
            raise
    
    def score_samples(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """Compute anomaly scores based on distance to nearest core point"""
        try:
            X = self.preprocess_data(X)
            
            # Find core samples
            core_samples_mask = np.zeros_like(self.model.labels_, dtype=bool)
            core_samples_mask[self.model.core_sample_indices_] = True
            
            # Get core samples
            core_samples = X[core_samples_mask]
            
            if len(core_samples) == 0:
                # If no core samples found, all points are anomalies
                return np.ones(len(X)) * np.inf
            
            # Compute minimum distance to any core point
            distances = np.min([
                np.linalg.norm(X - core, axis=1)
                for core in core_samples
            ], axis=0)
            
            return distances
            
        except Exception as e:
            self.logger.error(f"Error computing anomaly scores: {str(e)}")
            raise
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> AnomalyMetrics:
        """Predict anomalies"""
        try:
            X = self.preprocess_data(X)
            scores = self.score_samples(X)
            
            # Points with label -1 are outliers
            labels = (self.model.labels_ == -1).astype(int)
            num_anomalies = np.sum(labels)
            
            return AnomalyMetrics(
                anomaly_scores=scores,
                anomaly_labels=labels,
                threshold=self.eps,  # Use eps as threshold
                num_anomalies=num_anomalies,
                contamination=self.contamination
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting anomalies: {str(e)}")
            raise 