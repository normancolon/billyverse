import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger("billieverse.models.risk")

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var: float  # Value at Risk
    es: float   # Expected Shortfall
    sharpe: float  # Sharpe Ratio
    sortino: float  # Sortino Ratio
    max_drawdown: float  # Maximum Drawdown
    volatility: float  # Volatility
    skewness: float  # Return Distribution Skewness
    kurtosis: float  # Return Distribution Kurtosis

class BaseRiskAssessor:
    """Base class for risk assessment"""
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        risk_free_rate: float = 0.02,
        rolling_window: int = 252  # One trading year
    ):
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        self.risk_free_rate = risk_free_rate
        self.rolling_window = rolling_window
        self.logger = logger
        
        # Validation
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if time_horizon < 1:
            raise ValueError("Time horizon must be positive")
    
    def calculate_returns(
        self,
        prices: np.ndarray,
        log_returns: bool = True
    ) -> np.ndarray:
        """Calculate returns from price series"""
        try:
            if log_returns:
                returns = np.log(prices[1:] / prices[:-1])
            else:
                returns = (prices[1:] / prices[:-1]) - 1
            return returns
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {str(e)}")
            raise
    
    def calculate_var(
        self,
        returns: np.ndarray,
        method: str = 'historical'
    ) -> float:
        """Calculate Value at Risk"""
        try:
            if method == 'historical':
                return -np.percentile(returns, (1 - self.confidence_level) * 100)
            elif method == 'parametric':
                mean = np.mean(returns)
                std = np.std(returns)
                return -(mean + stats.norm.ppf(self.confidence_level) * std)
            else:
                raise ValueError(f"Unsupported VaR method: {method}")
                
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {str(e)}")
            raise
    
    def calculate_es(
        self,
        returns: np.ndarray,
        var: float
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        try:
            tail_losses = returns[returns <= -var]
            return -np.mean(tail_losses) if len(tail_losses) > 0 else -var
            
        except Exception as e:
            self.logger.error(f"Error calculating ES: {str(e)}")
            raise
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Calculate Sharpe Ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
            if annualize:
                return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
            return np.mean(excess_returns) / np.std(returns)
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            raise
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        annualize: bool = True
    ) -> float:
        """Calculate Sortino Ratio"""
        try:
            excess_returns = returns - self.risk_free_rate / 252
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
            
            if annualize:
                return np.sqrt(252) * np.mean(excess_returns) / downside_std
            return np.mean(excess_returns) / downside_std
            
        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            raise
    
    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate Maximum Drawdown"""
        try:
            peak = np.maximum.accumulate(prices)
            drawdown = (prices - peak) / peak
            return np.min(drawdown)
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            raise
    
    def calculate_rolling_metrics(
        self,
        prices: np.ndarray,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """Calculate rolling risk metrics"""
        try:
            if window is None:
                window = self.rolling_window
            
            returns = self.calculate_returns(prices)
            rolling_metrics = []
            
            for i in range(window, len(returns) + 1):
                window_returns = returns[i-window:i]
                var = self.calculate_var(window_returns)
                metrics = {
                    'var': var,
                    'es': self.calculate_es(window_returns, var),
                    'sharpe': self.calculate_sharpe_ratio(window_returns),
                    'sortino': self.calculate_sortino_ratio(window_returns),
                    'volatility': np.std(window_returns) * np.sqrt(252),
                    'skewness': stats.skew(window_returns),
                    'kurtosis': stats.kurtosis(window_returns)
                }
                rolling_metrics.append(metrics)
            
            return pd.DataFrame(rolling_metrics)
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {str(e)}")
            raise
    
    def assess_risk(
        self,
        prices: np.ndarray,
        calculate_rolling: bool = True
    ) -> Tuple[RiskMetrics, Optional[pd.DataFrame]]:
        """Comprehensive risk assessment"""
        try:
            returns = self.calculate_returns(prices)
            var = self.calculate_var(returns)
            
            metrics = RiskMetrics(
                var=var,
                es=self.calculate_es(returns, var),
                sharpe=self.calculate_sharpe_ratio(returns),
                sortino=self.calculate_sortino_ratio(returns),
                max_drawdown=self.calculate_max_drawdown(prices),
                volatility=np.std(returns) * np.sqrt(252),
                skewness=stats.skew(returns),
                kurtosis=stats.kurtosis(returns)
            )
            
            rolling_metrics = None
            if calculate_rolling:
                rolling_metrics = self.calculate_rolling_metrics(prices)
            
            return metrics, rolling_metrics
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            raise 