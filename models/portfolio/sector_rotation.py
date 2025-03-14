import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from models.portfolio.base import PortfolioOptimizer, OptimizationConstraints

logger = logging.getLogger("billieverse.models.portfolio.sector")

class SectorRotationModel:
    """ML model for predicting sector performance"""
    
    def __init__(
        self,
        lookback_periods: int = 12,
        prediction_horizon: int = 3,
        hidden_units: List[int] = [64, 32],
        dropout_rate: float = 0.2
    ):
        self.lookback_periods = lookback_periods
        self.prediction_horizon = prediction_horizon
        
        # Initialize models
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # Initialize neural network
        self.nn_model = self._build_nn_model(
            input_dim=lookback_periods,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate
        )
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.is_trained = False
    
    def _build_nn_model(
        self,
        input_dim: int,
        hidden_units: List[int],
        dropout_rate: float
    ) -> tf.keras.Model:
        """Build neural network model"""
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Hidden layers
        for units in hidden_units:
            model.add(tf.keras.layers.Dense(
                units,
                activation='relu'
            ))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(tf.keras.layers.Dense(
            self.prediction_horizon,
            activation='linear'
        ))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_features(
        self,
        returns: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        features = []
        targets = []
        
        for i in range(len(returns) - self.lookback_periods - self.prediction_horizon):
            # Get feature window
            feature_window = returns.iloc[i:i + self.lookback_periods]
            
            # Get target window
            target_window = returns.iloc[
                i + self.lookback_periods:
                i + self.lookback_periods + self.prediction_horizon
            ]
            
            features.append(feature_window.values.flatten())
            targets.append(target_window.mean().values)
        
        return np.array(features), np.array(targets)
    
    def train(
        self,
        sector_returns: pd.DataFrame,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train sector rotation models"""
        try:
            # Prepare data
            X, y = self._prepare_features(sector_returns)
            
            # Scale data
            X_scaled = self.feature_scaler.fit_transform(X)
            y_scaled = self.target_scaler.fit_transform(y)
            
            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y_scaled[:split_idx], y_scaled[split_idx:]
            
            # Train random forest
            self.rf_model.fit(X_train, y_train)
            rf_score = self.rf_model.score(X_val, y_val)
            
            # Train neural network
            history = self.nn_model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                verbose=0
            )
            
            nn_score = history.history['val_loss'][-1]
            
            self.is_trained = True
            
            return {
                'rf_score': rf_score,
                'nn_score': nn_score
            }
            
        except Exception as e:
            logger.error(f"Error training sector rotation model: {str(e)}")
            return {}
    
    def predict(
        self,
        recent_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Predict sector returns"""
        try:
            if not self.is_trained:
                logger.error("Model not trained")
                return None
            
            # Prepare features
            features = recent_returns.tail(self.lookback_periods).values.reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get predictions
            rf_pred = self.rf_model.predict(features_scaled)
            nn_pred = self.nn_model.predict(features_scaled, verbose=0)
            
            # Inverse transform predictions
            rf_pred = self.target_scaler.inverse_transform(rf_pred)
            nn_pred = self.target_scaler.inverse_transform(nn_pred)
            
            # Ensemble predictions (simple average)
            ensemble_pred = (rf_pred + nn_pred) / 2
            
            # Create prediction DataFrame
            predictions = pd.DataFrame(
                ensemble_pred,
                columns=[f'Period_{i+1}' for i in range(self.prediction_horizon)],
                index=recent_returns.columns
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting sector returns: {str(e)}")
            return None

class SectorRotationStrategy(PortfolioOptimizer):
    """AI-driven sector rotation strategy"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        sector_mapping: Dict[str, str],
        constraints: Optional[OptimizationConstraints] = None,
        benchmark_returns: Optional[pd.Series] = None,
        model_params: Optional[Dict] = None
    ):
        super().__init__(
            returns,
            constraints,
            benchmark_returns,
            sector_mapping
        )
        
        # Initialize sector rotation model
        self.sector_model = SectorRotationModel(
            **(model_params or {})
        )
        
        # Calculate sector returns
        self.sector_returns = self._calculate_sector_returns()
        
        # Initialize sector allocations
        self.sector_weights: Dict[str, float] = {}
        self.sector_weight_history: List[Dict[str, float]] = []
    
    def _calculate_sector_returns(self) -> pd.DataFrame:
        """Calculate historical sector returns"""
        sector_returns = {}
        
        for sector in set(self.sector_mapping.values()):
            # Get assets in sector
            sector_assets = [
                asset for asset, s in self.sector_mapping.items()
                if s == sector
            ]
            
            # Calculate equal-weighted sector returns
            if sector_assets:
                sector_returns[sector] = self.returns[sector_assets].mean(axis=1)
        
        return pd.DataFrame(sector_returns)
    
    def train_model(
        self,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """Train sector rotation model"""
        return self.sector_model.train(
            self.sector_returns,
            validation_split
        )
    
    def optimize_portfolio(
        self,
        risk_aversion: float = 2.0,
        method: str = 'SLSQP'
    ) -> Tuple[Dict[str, float], PortfolioMetrics]:
        """
        Optimize portfolio with sector rotation
        
        1. Predict sector returns
        2. Adjust asset expected returns based on sector predictions
        3. Optimize portfolio with adjusted returns
        """
        try:
            # Get sector predictions
            sector_predictions = self.sector_model.predict(self.sector_returns)
            if sector_predictions is None:
                return super().optimize_portfolio(risk_aversion, method)
            
            # Calculate sector scores (average predicted return)
            sector_scores = sector_predictions.mean(axis=1)
            
            # Adjust asset expected returns
            original_returns = self.mean_returns.copy()
            for asset in self.assets:
                sector = self.sector_mapping.get(asset)
                if sector in sector_scores:
                    # Blend original and sector-predicted returns
                    self.mean_returns[asset] = (
                        0.7 * original_returns[asset] +
                        0.3 * sector_scores[sector]
                    )
            
            # Run portfolio optimization
            weights, metrics = super().optimize_portfolio(risk_aversion, method)
            
            # Reset mean returns
            self.mean_returns = original_returns
            
            # Update sector weights
            if weights:
                self.sector_weights = self._calculate_sector_weights(weights)
                self.sector_weight_history.append(self.sector_weights)
            
            return weights, metrics
            
        except Exception as e:
            logger.error(f"Error in sector rotation optimization: {str(e)}")
            return None, None
    
    def _calculate_sector_weights(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate sector weights from asset weights"""
        sector_weights = {}
        
        for asset, weight in weights.items():
            sector = self.sector_mapping.get(asset)
            if sector:
                sector_weights[sector] = (
                    sector_weights.get(sector, 0) + weight
                )
        
        return sector_weights
    
    def get_sector_momentum(
        self,
        lookback_periods: int = 6
    ) -> Dict[str, float]:
        """Calculate sector momentum scores"""
        try:
            momentum_scores = {}
            
            # Get recent sector returns
            recent_returns = self.sector_returns.tail(lookback_periods)
            
            for sector in recent_returns.columns:
                # Calculate momentum factors
                returns = recent_returns[sector]
                
                # 1. Return momentum
                return_momentum = returns.mean()
                
                # 2. Trend strength
                trend = np.polyfit(range(len(returns)), returns, 1)[0]
                
                # 3. Volatility-adjusted momentum
                vol_adj_momentum = return_momentum / returns.std()
                
                # Combine factors
                momentum_scores[sector] = (
                    0.4 * return_momentum +
                    0.4 * trend +
                    0.2 * vol_adj_momentum
                )
            
            return momentum_scores
            
        except Exception as e:
            logger.error(f"Error calculating sector momentum: {str(e)}")
            return {} 