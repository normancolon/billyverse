import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from scipy import stats

@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float
    correlation: float
    volatility: float

@dataclass
class PositionLimit:
    symbol: str
    max_position_size: float
    max_notional_value: float
    max_leverage: float

class RiskManager:
    def __init__(self,
                 initial_capital: float,
                 max_position_pct: float = 0.2,
                 max_portfolio_var: float = 0.02,
                 max_leverage: float = 2.0,
                 risk_free_rate: float = 0.02):
        """
        Risk management system
        
        Args:
            initial_capital: Starting capital
            max_position_pct: Maximum position size as percentage of capital
            max_portfolio_var: Maximum portfolio VaR (95%)
            max_leverage: Maximum allowed leverage
            risk_free_rate: Annual risk-free rate
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_portfolio_var = max_portfolio_var
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate
        
        # Position limits
        self.position_limits: Dict[str, PositionLimit] = {}
        
        # Risk metrics history
        self.metrics_history: List[Tuple[datetime, RiskMetrics]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_limits(self, 
                                symbol: str,
                                price: float,
                                volatility: float) -> PositionLimit:
        """Calculates position limits for a symbol"""
        # Maximum position size based on capital
        max_position_size = self.initial_capital * self.max_position_pct / price
        
        # Adjust for volatility
        volatility_factor = np.exp(-volatility)  # Reduce limits for high volatility
        adjusted_position_size = max_position_size * volatility_factor
        
        # Maximum notional value
        max_notional = self.initial_capital * self.max_position_pct
        
        # Maximum leverage based on volatility
        max_leverage = self.max_leverage * volatility_factor
        
        return PositionLimit(
            symbol=symbol,
            max_position_size=adjusted_position_size,
            max_notional_value=max_notional,
            max_leverage=max_leverage
        )
        
    def update_position_limits(self, market_data: Dict[str, pd.DataFrame]):
        """Updates position limits based on market conditions"""
        for symbol, data in market_data.items():
            # Calculate volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Update limits
            self.position_limits[symbol] = self.calculate_position_limits(
                symbol=symbol,
                price=data['close'].iloc[-1],
                volatility=volatility
            )
            
    def calculate_var(self, 
                     returns: pd.Series,
                     confidence_level: float = 0.95,
                     time_horizon: int = 1) -> float:
        """Calculates Value at Risk"""
        # Parametric VaR assuming normal distribution
        var = stats.norm.ppf(1 - confidence_level) * returns.std() * np.sqrt(time_horizon)
        return abs(var)
        
    def calculate_expected_shortfall(self,
                                   returns: pd.Series,
                                   confidence_level: float = 0.95) -> float:
        """Calculates Expected Shortfall (CVaR)"""
        var = self.calculate_var(returns, confidence_level)
        return abs(returns[returns <= -var].mean())
        
    def calculate_portfolio_metrics(self,
                                  positions: Dict[str, float],
                                  market_data: Dict[str, pd.DataFrame]) -> RiskMetrics:
        """Calculates portfolio risk metrics"""
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=market_data[list(market_data.keys())[0]].index)
        
        for symbol, amount in positions.items():
            if symbol in market_data:
                price_data = market_data[symbol]['close']
                position_value = amount * price_data
                returns = price_data.pct_change().fillna(0)
                portfolio_returns += (position_value.shift(1) * returns) / self.initial_capital
                
        # Calculate metrics
        var_95 = self.calculate_var(portfolio_returns, 0.95)
        var_99 = self.calculate_var(portfolio_returns, 0.99)
        es = self.calculate_expected_shortfall(portfolio_returns)
        
        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate/252
        sharpe = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = abs(drawdowns.min())
        
        # Market beta (using SPY as market proxy)
        if 'SPY' in market_data:
            market_returns = market_data['SPY']['close'].pct_change()
            beta = stats.linregress(market_returns, portfolio_returns).slope
            correlation = portfolio_returns.corr(market_returns)
        else:
            beta = 1.0
            correlation = 0.0
            
        # Volatility
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        metrics = RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=es,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            beta=beta,
            correlation=correlation,
            volatility=volatility
        )
        
        # Store metrics history
        self.metrics_history.append((datetime.now(), metrics))
        
        return metrics
        
    def check_trade(self,
                    symbol: str,
                    side: str,
                    amount: float,
                    price: float,
                    current_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Checks if a trade complies with risk limits"""
        if symbol not in self.position_limits:
            return False, "Symbol not found in position limits"
            
        limits = self.position_limits[symbol]
        current_position = current_positions.get(symbol, 0)
        
        # Check position size
        new_position = current_position + (amount if side == 'buy' else -amount)
        if abs(new_position) > limits.max_position_size:
            return False, f"Position size {abs(new_position)} exceeds limit {limits.max_position_size}"
            
        # Check notional value
        notional_value = abs(new_position * price)
        if notional_value > limits.max_notional_value:
            return False, f"Notional value {notional_value} exceeds limit {limits.max_notional_value}"
            
        # Calculate total leverage
        total_notional = sum(
            abs(pos * self.position_limits[sym].max_notional_value)
            for sym, pos in current_positions.items()
        ) + notional_value
        
        leverage = total_notional / self.initial_capital
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.2f} exceeds limit {self.max_leverage}"
            
        return True, "Trade approved"
        
    def get_metrics_report(self) -> str:
        """Generates a risk metrics report"""
        if not self.metrics_history:
            return "No risk metrics available"
            
        latest_time, latest_metrics = self.metrics_history[-1]
        
        report = "=== Risk Metrics Report ===\n\n"
        report += f"Timestamp: {latest_time}\n\n"
        
        report += "Value at Risk:\n"
        report += f"95% VaR: {latest_metrics.var_95:.2%}\n"
        report += f"99% VaR: {latest_metrics.var_99:.2%}\n"
        report += f"Expected Shortfall: {latest_metrics.expected_shortfall:.2%}\n\n"
        
        report += "Performance Metrics:\n"
        report += f"Sharpe Ratio: {latest_metrics.sharpe_ratio:.2f}\n"
        report += f"Maximum Drawdown: {latest_metrics.max_drawdown:.2%}\n"
        report += f"Beta: {latest_metrics.beta:.2f}\n"
        report += f"Market Correlation: {latest_metrics.correlation:.2f}\n"
        report += f"Annualized Volatility: {latest_metrics.volatility:.2%}\n"
        
        return report 