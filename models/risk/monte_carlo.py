import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime
import logging
from dataclasses import dataclass
from models.risk.base import BaseRiskAssessor, RiskMetrics

logger = logging.getLogger("billieverse.models.risk.monte_carlo")

@dataclass
class SimulationResults:
    """Container for Monte Carlo simulation results"""
    simulated_paths: np.ndarray  # Shape: (n_simulations, n_days)
    var_estimates: np.ndarray  # VaR for each simulation
    es_estimates: np.ndarray  # ES for each simulation
    confidence_intervals: Dict[str, Tuple[float, float]]  # CI for each metric
    worst_case_path: np.ndarray  # Worst performing path
    best_case_path: np.ndarray  # Best performing path
    mean_path: np.ndarray  # Mean path

class MonteCarloRiskAssessor(BaseRiskAssessor):
    """Monte Carlo simulation for risk assessment"""
    
    def __init__(
        self,
        n_simulations: int = 10000,
        n_days: int = 252,
        confidence_level: float = 0.95,
        time_horizon: int = 1,
        risk_free_rate: float = 0.02,
        random_seed: Optional[int] = None
    ):
        super().__init__(
            confidence_level=confidence_level,
            time_horizon=time_horizon,
            risk_free_rate=risk_free_rate
        )
        
        self.n_simulations = n_simulations
        self.n_days = n_days
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def estimate_parameters(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Estimate parameters for the simulation"""
        try:
            # Calculate mean and volatility
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            # Calculate higher moments
            skew = stats.skew(returns)
            kurt = stats.kurtosis(returns)
            
            return mu, sigma, skew, kurt
            
        except Exception as e:
            self.logger.error(f"Error estimating parameters: {str(e)}")
            raise
    
    def simulate_gbm(
        self,
        initial_price: float,
        mu: float,
        sigma: float
    ) -> np.ndarray:
        """Simulate Geometric Brownian Motion"""
        try:
            # Generate random normal variables
            Z = np.random.normal(
                size=(self.n_simulations, self.n_days)
            )
            
            # Calculate daily returns
            daily_returns = (
                mu - 0.5 * sigma**2
            ) + sigma * Z
            
            # Calculate price paths
            price_paths = np.zeros((self.n_simulations, self.n_days + 1))
            price_paths[:, 0] = initial_price
            
            for t in range(1, self.n_days + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    daily_returns[:, t-1]
                )
            
            return price_paths
            
        except Exception as e:
            self.logger.error(f"Error in GBM simulation: {str(e)}")
            raise
    
    def simulate_student_t(
        self,
        initial_price: float,
        mu: float,
        sigma: float,
        df: float = 5
    ) -> np.ndarray:
        """Simulate with Student's t-distribution for fat tails"""
        try:
            # Generate random t variables
            Z = stats.t.rvs(
                df=df,
                size=(self.n_simulations, self.n_days)
            )
            
            # Standardize to match target moments
            Z = (Z - np.mean(Z)) / np.std(Z)
            
            # Calculate daily returns
            daily_returns = mu + sigma * Z
            
            # Calculate price paths
            price_paths = np.zeros((self.n_simulations, self.n_days + 1))
            price_paths[:, 0] = initial_price
            
            for t in range(1, self.n_days + 1):
                price_paths[:, t] = price_paths[:, t-1] * np.exp(
                    daily_returns[:, t-1]
                )
            
            return price_paths
            
        except Exception as e:
            self.logger.error(f"Error in Student's t simulation: {str(e)}")
            raise
    
    def calculate_confidence_intervals(
        self,
        metric_values: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """Calculate confidence intervals for metrics"""
        try:
            if confidence_level is None:
                confidence_level = self.confidence_level
            
            lower = np.percentile(
                metric_values,
                (1 - confidence_level) * 100 / 2
            )
            upper = np.percentile(
                metric_values,
                (1 + confidence_level) * 100 / 2
            )
            
            return lower, upper
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence intervals: {str(e)}")
            raise
    
    def run_simulation(
        self,
        prices: np.ndarray,
        use_t_dist: bool = True,
        df: float = 5
    ) -> SimulationResults:
        """Run Monte Carlo simulation"""
        try:
            # Calculate historical returns
            returns = self.calculate_returns(prices)
            
            # Estimate parameters
            mu, sigma, skew, kurt = self.estimate_parameters(returns)
            
            # Run simulation
            initial_price = prices[-1]
            if use_t_dist:
                paths = self.simulate_student_t(
                    initial_price=initial_price,
                    mu=mu,
                    sigma=sigma,
                    df=df
                )
            else:
                paths = self.simulate_gbm(
                    initial_price=initial_price,
                    mu=mu,
                    sigma=sigma
                )
            
            # Calculate metrics for each path
            var_estimates = np.zeros(self.n_simulations)
            es_estimates = np.zeros(self.n_simulations)
            
            for i in range(self.n_simulations):
                path_returns = self.calculate_returns(paths[i])
                var_estimates[i] = self.calculate_var(path_returns)
                es_estimates[i] = self.calculate_es(path_returns, var_estimates[i])
            
            # Calculate confidence intervals
            confidence_intervals = {
                'var': self.calculate_confidence_intervals(var_estimates),
                'es': self.calculate_confidence_intervals(es_estimates),
                'final_price': self.calculate_confidence_intervals(paths[:, -1])
            }
            
            # Get summary paths
            worst_case_idx = np.argmin(paths[:, -1])
            best_case_idx = np.argmax(paths[:, -1])
            mean_path = np.mean(paths, axis=0)
            
            return SimulationResults(
                simulated_paths=paths,
                var_estimates=var_estimates,
                es_estimates=es_estimates,
                confidence_intervals=confidence_intervals,
                worst_case_path=paths[worst_case_idx],
                best_case_path=paths[best_case_idx],
                mean_path=mean_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise
    
    def assess_risk(
        self,
        prices: np.ndarray,
        calculate_rolling: bool = True,
        use_t_dist: bool = True
    ) -> Tuple[RiskMetrics, Optional[pd.DataFrame], SimulationResults]:
        """Comprehensive risk assessment with Monte Carlo simulation"""
        try:
            # Get base metrics
            metrics, rolling_metrics = super().assess_risk(
                prices,
                calculate_rolling
            )
            
            # Run Monte Carlo simulation
            simulation_results = self.run_simulation(
                prices,
                use_t_dist=use_t_dist
            )
            
            return metrics, rolling_metrics, simulation_results
            
        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            raise 