import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from scipy.stats import norm

logger = logging.getLogger("billieverse.models.portfolio.risk")

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float  # 95% Value at Risk
    cvar_95: float  # 95% Conditional VaR
    beta: float  # Portfolio beta
    correlation: float  # Correlation with benchmark
    tracking_error: float  # Tracking error
    active_risk: float  # Active risk vs benchmark
    concentration_risk: float  # Portfolio concentration
    tail_risk: float  # Tail risk measure
    liquidity_score: float  # Portfolio liquidity score
    stress_loss: float  # Stress test loss

@dataclass
class HedgeParameters:
    """Parameters for hedging strategies"""
    hedge_ratio: float = 1.0  # Target hedge ratio
    rebalance_threshold: float = 0.1  # Threshold for hedge rebalancing
    max_hedge_cost: float = 0.002  # Maximum acceptable hedge cost
    min_correlation: float = 0.7  # Minimum correlation for hedge instrument
    max_tracking_error: float = 0.02  # Maximum tracking error for hedge

class RiskManager:
    """Portfolio risk management and hedging"""
    
    def __init__(
        self,
        returns: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        hedge_universe: Optional[pd.DataFrame] = None,
        hedge_params: Optional[HedgeParameters] = None,
        risk_limits: Optional[Dict[str, float]] = None
    ):
        """
        Initialize risk manager
        
        Args:
            returns: Asset returns
            benchmark_returns: Benchmark returns
            hedge_universe: Returns of potential hedge instruments
            hedge_params: Hedging parameters
            risk_limits: Risk limits dictionary
        """
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        self.hedge_universe = hedge_universe
        self.hedge_params = hedge_params or HedgeParameters()
        
        # Set default risk limits if not provided
        self.risk_limits = risk_limits or {
            'max_var_95': 0.02,  # 2% daily VaR limit
            'max_leverage': 1.5,  # 150% gross exposure
            'max_concentration': 0.25,  # 25% max position
            'min_liquidity': 0.8,  # 80% portfolio liquidity
            'max_correlation': 0.7,  # 70% max correlation
            'max_drawdown': 0.15  # 15% max drawdown
        }
        
        # Initialize tracking
        self.current_hedges: Dict[str, float] = {}
        self.hedge_history: List[Dict[str, float]] = []
        self.risk_metrics_history: List[RiskMetrics] = []
        
        self.logger = logger
    
    def calculate_risk_metrics(
        self,
        weights: Dict[str, float],
        include_hedges: bool = True
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Convert weights to array
            assets = list(weights.keys())
            w = np.array([weights[asset] for asset in assets])
            
            # Include hedge positions if requested
            if include_hedges and self.current_hedges:
                assets.extend(self.current_hedges.keys())
                w = np.append(
                    w,
                    [self.current_hedges[h] for h in self.current_hedges]
                )
            
            # Get returns for calculation
            portfolio_returns = self.returns[assets] @ w
            
            # Calculate VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            # Calculate beta and correlation if benchmark available
            if self.benchmark_returns is not None:
                covariance = np.cov(
                    portfolio_returns,
                    self.benchmark_returns
                )[0, 1]
                benchmark_var = np.var(self.benchmark_returns)
                beta = covariance / benchmark_var
                correlation = np.corrcoef(
                    portfolio_returns,
                    self.benchmark_returns
                )[0, 1]
                
                # Calculate tracking error and active risk
                tracking_diff = portfolio_returns - self.benchmark_returns
                tracking_error = np.std(tracking_diff)
                active_risk = tracking_error * np.sqrt(252)  # Annualized
            else:
                beta = 0.0
                correlation = 0.0
                tracking_error = 0.0
                active_risk = 0.0
            
            # Calculate concentration risk
            concentration_risk = np.sum(w * w)  # Herfindahl index
            
            # Calculate tail risk (expected shortfall beyond 3 std)
            tail_threshold = portfolio_returns.mean() - 3 * portfolio_returns.std()
            tail_risk = portfolio_returns[portfolio_returns <= tail_threshold].mean()
            
            # Calculate liquidity score (simple version)
            liquidity_score = np.sum(w[w > 0.01]) / np.sum(w)
            
            # Calculate stress loss (simple historical stress test)
            worst_period_return = portfolio_returns.min()
            stress_loss = abs(worst_period_return)
            
            metrics = RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                correlation=correlation,
                tracking_error=tracking_error,
                active_risk=active_risk,
                concentration_risk=concentration_risk,
                tail_risk=tail_risk,
                liquidity_score=liquidity_score,
                stress_loss=stress_loss
            )
            
            self.risk_metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {str(e)}")
            return None
    
    def check_risk_limits(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check if portfolio violates risk limits"""
        try:
            violations = {}
            metrics = self.calculate_risk_metrics(weights)
            
            if metrics is None:
                return {'calculation_error': True}
            
            # Check VaR limit
            violations['var_violation'] = (
                abs(metrics.var_95) > self.risk_limits['max_var_95']
            )
            
            # Check leverage limit
            gross_exposure = sum(abs(w) for w in weights.values())
            violations['leverage_violation'] = (
                gross_exposure > self.risk_limits['max_leverage']
            )
            
            # Check concentration limit
            max_position = max(abs(w) for w in weights.values())
            violations['concentration_violation'] = (
                max_position > self.risk_limits['max_concentration']
            )
            
            # Check liquidity limit
            violations['liquidity_violation'] = (
                metrics.liquidity_score < self.risk_limits['min_liquidity']
            )
            
            # Check correlation limit
            violations['correlation_violation'] = (
                abs(metrics.correlation) > self.risk_limits['max_correlation']
            )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {str(e)}")
            return {'calculation_error': True}
    
    def optimize_hedge_ratio(
        self,
        portfolio_returns: pd.Series,
        hedge_returns: pd.Series
    ) -> float:
        """Optimize hedge ratio using regression"""
        try:
            # Calculate optimal hedge ratio using OLS
            covariance = np.cov(portfolio_returns, hedge_returns)[0, 1]
            hedge_variance = np.var(hedge_returns)
            
            # Calculate optimal hedge ratio
            hedge_ratio = -covariance / hedge_variance
            
            # Apply hedge ratio limits
            hedge_ratio = np.clip(
                hedge_ratio,
                0.0,
                self.hedge_params.hedge_ratio
            )
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error optimizing hedge ratio: {str(e)}")
            return 0.0
    
    def select_hedge_instruments(
        self,
        portfolio_returns: pd.Series
    ) -> Dict[str, float]:
        """Select optimal hedge instruments"""
        try:
            if self.hedge_universe is None:
                return {}
            
            hedge_scores = {}
            
            for hedge in self.hedge_universe.columns:
                hedge_returns = self.hedge_universe[hedge]
                
                # Calculate correlation
                correlation = np.corrcoef(
                    portfolio_returns,
                    hedge_returns
                )[0, 1]
                
                # Calculate tracking error
                tracking_error = np.std(
                    portfolio_returns - hedge_returns
                )
                
                # Calculate hedge score
                if (abs(correlation) > self.hedge_params.min_correlation and
                    tracking_error < self.hedge_params.max_tracking_error):
                    hedge_scores[hedge] = {
                        'correlation': abs(correlation),
                        'tracking_error': tracking_error,
                        'ratio': self.optimize_hedge_ratio(
                            portfolio_returns,
                            hedge_returns
                        )
                    }
            
            # Select top hedges
            selected_hedges = {}
            for hedge, scores in sorted(
                hedge_scores.items(),
                key=lambda x: x[1]['correlation'],
                reverse=True
            )[:3]:  # Select top 3 hedges
                selected_hedges[hedge] = scores['ratio']
            
            return selected_hedges
            
        except Exception as e:
            self.logger.error(f"Error selecting hedge instruments: {str(e)}")
            return {}
    
    def update_hedges(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Update portfolio hedges"""
        try:
            # Calculate portfolio returns
            portfolio_returns = self.returns[weights.keys()] @ np.array(
                list(weights.values())
            )
            
            # Select hedge instruments
            new_hedges = self.select_hedge_instruments(portfolio_returns)
            
            # Check if rebalancing needed
            if self.current_hedges:
                hedge_diff = {
                    h: abs(new_hedges.get(h, 0) - self.current_hedges.get(h, 0))
                    for h in set(new_hedges) | set(self.current_hedges)
                }
                
                if max(hedge_diff.values()) < self.hedge_params.rebalance_threshold:
                    return self.current_hedges
            
            # Update hedges
            self.current_hedges = new_hedges
            self.hedge_history.append(new_hedges)
            
            return new_hedges
            
        except Exception as e:
            self.logger.error(f"Error updating hedges: {str(e)}")
            return {}
    
    def calculate_hedge_costs(
        self,
        old_hedges: Dict[str, float],
        new_hedges: Dict[str, float]
    ) -> float:
        """Calculate costs of updating hedges"""
        try:
            total_cost = 0.0
            
            # Calculate trading costs for hedge changes
            for hedge in set(old_hedges) | set(new_hedges):
                old_position = old_hedges.get(hedge, 0)
                new_position = new_hedges.get(hedge, 0)
                
                # Simple transaction cost model
                trade_size = abs(new_position - old_position)
                trade_cost = trade_size * 0.0002  # 2bps per trade
                
                total_cost += trade_cost
            
            return total_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating hedge costs: {str(e)}")
            return float('inf')
    
    def get_risk_report(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Generate comprehensive risk report"""
        try:
            metrics = self.calculate_risk_metrics(weights)
            if metrics is None:
                return {}
            
            report = {
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'beta': metrics.beta,
                'correlation': metrics.correlation,
                'tracking_error': metrics.tracking_error,
                'active_risk': metrics.active_risk,
                'concentration_risk': metrics.concentration_risk,
                'tail_risk': metrics.tail_risk,
                'liquidity_score': metrics.liquidity_score,
                'stress_loss': metrics.stress_loss,
                'gross_exposure': sum(abs(w) for w in weights.values()),
                'net_exposure': sum(weights.values()),
                'hedge_ratio': sum(self.current_hedges.values()),
                'num_positions': sum(1 for w in weights.values() if abs(w) > 0.01)
            }
            
            # Add risk limit statuses
            violations = self.check_risk_limits(weights)
            report.update({
                f'{k}_status': 'Violation' if v else 'OK'
                for k, v in violations.items()
            })
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {str(e)}")
            return {} 