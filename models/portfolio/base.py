import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy.optimize import minimize

logger = logging.getLogger("billieverse.models.portfolio")

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics"""
    returns: float  # Expected portfolio return
    volatility: float  # Portfolio volatility
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Maximum historical drawdown
    var_95: float  # 95% Value at Risk
    beta: float  # Portfolio beta to benchmark
    tracking_error: float  # Tracking error to benchmark
    information_ratio: float  # Risk-adjusted excess return
    diversification_score: float  # Portfolio diversification metric
    turnover: float  # Portfolio turnover ratio

@dataclass
class OptimizationConstraints:
    """Container for portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    min_total: float = 0.95  # Minimum total allocation
    max_total: float = 1.05  # Maximum total allocation
    max_sector_weight: float = 0.3  # Maximum sector allocation
    min_assets: int = 10  # Minimum number of assets
    max_turnover: float = 0.2  # Maximum portfolio turnover
    risk_free_rate: float = 0.02  # Risk-free rate for Sharpe ratio

class PortfolioOptimizer:
    """Base class for portfolio optimization using MPT"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        constraints: Optional[OptimizationConstraints] = None,
        benchmark_returns: Optional[pd.Series] = None,
        sector_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            returns: DataFrame of asset returns
            constraints: Optimization constraints
            benchmark_returns: Benchmark returns for relative metrics
            sector_mapping: Mapping of assets to sectors
        """
        self.returns = returns
        self.constraints = constraints or OptimizationConstraints()
        self.benchmark_returns = benchmark_returns
        self.sector_mapping = sector_mapping or {}
        
        # Calculate core metrics
        self.mean_returns = returns.mean()
        self.cov_matrix = returns.cov()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)
        
        # Initialize tracking
        self.current_weights: Dict[str, float] = {}
        self.weight_history: List[Dict[str, float]] = []
        self.metrics_history: List[PortfolioMetrics] = []
    
    def _calculate_portfolio_metrics(
        self,
        weights: np.ndarray
    ) -> PortfolioMetrics:
        """Calculate portfolio performance metrics"""
        try:
            # Convert weights to array
            weights = np.array(weights)
            
            # Calculate basic metrics
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_vol = np.sqrt(
                weights.T @ self.cov_matrix @ weights
            )
            
            # Calculate Sharpe ratio
            sharpe = (
                (portfolio_return - self.constraints.risk_free_rate) /
                portfolio_vol
            )
            
            # Calculate historical portfolio returns
            historical_returns = self.returns @ weights
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + historical_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Calculate VaR
            var_95 = np.percentile(historical_returns, 5)
            
            # Calculate beta and tracking error if benchmark available
            if self.benchmark_returns is not None:
                covariance = np.cov(
                    historical_returns,
                    self.benchmark_returns
                )[0, 1]
                benchmark_var = np.var(self.benchmark_returns)
                beta = covariance / benchmark_var
                
                tracking_diff = historical_returns - self.benchmark_returns
                tracking_error = np.std(tracking_diff)
                information_ratio = np.mean(tracking_diff) / tracking_error
            else:
                beta = 0.0
                tracking_error = 0.0
                information_ratio = 0.0
            
            # Calculate diversification score
            sector_weights = {}
            for asset, weight in zip(self.assets, weights):
                sector = self.sector_mapping.get(asset, 'Unknown')
                sector_weights[sector] = (
                    sector_weights.get(sector, 0) + weight
                )
            
            herfindahl = sum(w * w for w in sector_weights.values())
            diversification_score = 1 - herfindahl
            
            # Calculate turnover if current weights exist
            if self.current_weights:
                current_w = np.array([
                    self.current_weights.get(asset, 0)
                    for asset in self.assets
                ])
                turnover = np.sum(np.abs(weights - current_w))
            else:
                turnover = 0.0
            
            return PortfolioMetrics(
                returns=portfolio_return,
                volatility=portfolio_vol,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                var_95=var_95,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                diversification_score=diversification_score,
                turnover=turnover
            )
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {str(e)}")
            return None
    
    def _objective_function(
        self,
        weights: np.ndarray,
        risk_aversion: float = 2.0
    ) -> float:
        """
        Objective function for portfolio optimization
        Maximizes utility = returns - risk_aversion * volatility
        """
        try:
            metrics = self._calculate_portfolio_metrics(weights)
            if metrics is None:
                return float('inf')
            
            utility = (
                metrics.returns -
                risk_aversion * metrics.volatility -
                0.5 * metrics.turnover  # Penalty for turnover
            )
            
            return -utility  # Minimize negative utility
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return float('inf')
    
    def _get_constraints(self) -> List[Dict]:
        """Get optimization constraints"""
        constraints = [
            # Sum of weights constraint
            {
                'type': 'ineq',
                'fun': lambda x: self.constraints.max_total - np.sum(x)
            },
            {
                'type': 'ineq',
                'fun': lambda x: np.sum(x) - self.constraints.min_total
            },
            # Minimum number of assets constraint
            {
                'type': 'ineq',
                'fun': lambda x: np.sum(x > 0.01) - self.constraints.min_assets
            }
        ]
        
        # Add sector constraints if mapping exists
        if self.sector_mapping:
            sectors = set(self.sector_mapping.values())
            for sector in sectors:
                sector_assets = [
                    i for i, asset in enumerate(self.assets)
                    if self.sector_mapping.get(asset) == sector
                ]
                if sector_assets:
                    constraints.append({
                        'type': 'ineq',
                        'fun': lambda x, idx=sector_assets: (
                            self.constraints.max_sector_weight -
                            np.sum(x[idx])
                        )
                    })
        
        return constraints
    
    def optimize_portfolio(
        self,
        risk_aversion: float = 2.0,
        method: str = 'SLSQP'
    ) -> Tuple[Dict[str, float], PortfolioMetrics]:
        """
        Optimize portfolio weights using MPT
        
        Args:
            risk_aversion: Risk aversion parameter
            method: Optimization method
        
        Returns:
            Tuple of (optimal weights, portfolio metrics)
        """
        try:
            # Initial guess: equal weights
            initial_weights = np.array([1.0 / self.n_assets] * self.n_assets)
            
            # Set bounds for individual weights
            bounds = [
                (self.constraints.min_weight, self.constraints.max_weight)
                for _ in range(self.n_assets)
            ]
            
            # Get constraints
            constraints = self._get_constraints()
            
            # Run optimization
            result = minimize(
                self._objective_function,
                initial_weights,
                args=(risk_aversion,),
                method=method,
                bounds=bounds,
                constraints=constraints
            )
            
            if not result.success:
                logger.warning(
                    f"Portfolio optimization failed: {result.message}"
                )
                return None, None
            
            # Get optimal weights and metrics
            optimal_weights = result.x
            metrics = self._calculate_portfolio_metrics(optimal_weights)
            
            # Update current weights and history
            self.current_weights = dict(zip(self.assets, optimal_weights))
            self.weight_history.append(self.current_weights)
            self.metrics_history.append(metrics)
            
            return self.current_weights, metrics
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {str(e)}")
            return None, None
    
    def get_rebalancing_trades(
        self,
        portfolio_value: float
    ) -> Dict[str, Dict[str, float]]:
        """
        Get trades needed to rebalance to optimal weights
        
        Args:
            portfolio_value: Current portfolio value
        
        Returns:
            Dict of trades {asset: {'amount': amount, 'direction': 1/-1}}
        """
        try:
            if not self.current_weights:
                return {}
            
            trades = {}
            for asset in self.assets:
                # Calculate target position
                target_position = (
                    self.current_weights.get(asset, 0) * portfolio_value
                )
                
                # Calculate current position
                current_position = (
                    self.weight_history[-2].get(asset, 0) * portfolio_value
                    if len(self.weight_history) > 1
                    else 0
                )
                
                # Calculate trade amount and direction
                trade_amount = abs(target_position - current_position)
                if trade_amount > 100:  # Minimum trade size
                    direction = 1 if target_position > current_position else -1
                    trades[asset] = {
                        'amount': trade_amount,
                        'direction': direction
                    }
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalancing trades: {str(e)}")
            return {} 