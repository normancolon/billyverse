import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

@dataclass
class Trade:
    timestamp: datetime
    symbol: str
    side: str
    amount: float
    price: float
    fees: float
    pnl: float = 0.0

@dataclass
class Position:
    symbol: str
    amount: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

class BacktestEngine:
    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 initial_capital: float = 1000000.0,
                 fee_rate: float = 0.001,
                 slippage: float = 0.0002,
                 risk_free_rate: float = 0.02):
        """
        Backtesting engine for strategy validation
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each asset
            initial_capital: Starting capital for the backtest
            fee_rate: Trading fee rate (as decimal)
            slippage: Estimated slippage (as decimal)
            risk_free_rate: Annual risk-free rate for performance metrics
        """
        self.data = data
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate
        
        # Trading history
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.equity_curve = []
        
        # Performance metrics
        self.metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def execute_trade(self, timestamp: datetime, symbol: str, 
                     side: str, amount: float, price: float) -> Optional[Trade]:
        """Executes a trade and updates positions"""
        # Apply slippage
        executed_price = price * (1 + self.slippage) if side == "BUY" else price * (1 - self.slippage)
        
        # Calculate fees
        fees = amount * executed_price * self.fee_rate
        total_cost = amount * executed_price + fees
        
        # Check if we have enough capital
        if side == "BUY" and total_cost > self.current_capital:
            self.logger.warning(f"Insufficient capital for trade: {total_cost} > {self.current_capital}")
            return None
            
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0, 0, 0)
            
        position = self.positions[symbol]
        
        if side == "BUY":
            # Update average price
            total_cost_basis = position.amount * position.avg_price + amount * executed_price
            new_amount = position.amount + amount
            position.avg_price = total_cost_basis / new_amount if new_amount > 0 else 0
            position.amount = new_amount
            self.current_capital -= total_cost
        else:  # SELL
            if amount > position.amount:
                self.logger.warning(f"Insufficient position for sale: {amount} > {position.amount}")
                return None
                
            # Calculate realized PnL
            pnl = amount * (executed_price - position.avg_price) - fees
            position.realized_pnl += pnl
            position.amount -= amount
            self.current_capital += amount * executed_price - fees
            
        # Record trade
        trade = Trade(timestamp, symbol, side, amount, executed_price, fees, pnl)
        self.trades.append(trade)
        
        return trade
        
    def update_positions(self, timestamp: datetime, prices: Dict[str, float]):
        """Updates unrealized PnL for all positions"""
        total_equity = self.current_capital
        
        for symbol, position in self.positions.items():
            if position.amount == 0:
                continue
                
            current_price = prices[symbol]
            position.unrealized_pnl = position.amount * (current_price - position.avg_price)
            total_equity += position.unrealized_pnl
            
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity
        })
        
    def run_backtest(self, strategy: Callable):
        """Runs the backtest using the provided strategy"""
        # Get common timestamp index
        timestamps = sorted(set.intersection(*[set(df.index) for df in self.data.values()]))
        
        for timestamp in timestamps:
            # Get current prices
            prices = {symbol: df.loc[timestamp, 'close'] 
                     for symbol, df in self.data.items()}
            
            # Update positions
            self.update_positions(timestamp, prices)
            
            # Execute strategy
            strategy(timestamp, self.data, prices, self)
            
        # Calculate performance metrics
        self._calculate_metrics()
        
    def _calculate_metrics(self):
        """Calculates performance metrics for the backtest"""
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        returns = equity_df['equity'].pct_change().dropna()
        
        # Basic metrics
        self.metrics['total_return'] = (self.current_capital / self.initial_capital - 1)
        self.metrics['num_trades'] = len(self.trades)
        
        if len(returns) > 0:
            # Risk metrics
            self.metrics['volatility'] = returns.std() * np.sqrt(252)  # Annualized
            self.metrics['sharpe_ratio'] = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252)
            self.metrics['max_drawdown'] = self._calculate_max_drawdown(equity_df['equity'])
            
            # Trading metrics
            profitable_trades = [t for t in self.trades if t.pnl > 0]
            self.metrics['win_rate'] = len(profitable_trades) / len(self.trades) if self.trades else 0
            self.metrics['profit_factor'] = (
                sum(t.pnl for t in profitable_trades) / 
                abs(sum(t.pnl for t in self.trades if t.pnl < 0))
                if sum(t.pnl for t in self.trades if t.pnl < 0) != 0 else float('inf')
            )
            
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculates maximum drawdown from peak"""
        rolling_max = equity.expanding().max()
        drawdowns = equity / rolling_max - 1
        return abs(drawdowns.min())
        
    def plot_results(self, save_path: Optional[str] = None):
        """Plots backtest results"""
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        equity_df.set_index('timestamp').plot(y='equity', ax=ax1)
        ax1.set_title('Equity Curve')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Daily returns distribution
        returns = equity_df.set_index('timestamp')['equity'].pct_change().dropna()
        sns.histplot(returns, kde=True, ax=ax2)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
    def get_trade_history(self) -> pd.DataFrame:
        """Returns trade history as a DataFrame"""
        return pd.DataFrame([vars(t) for t in self.trades])
        
    def get_metrics_report(self) -> str:
        """Generates a formatted performance metrics report"""
        report = "=== Backtest Results ===\n\n"
        
        # Portfolio metrics
        report += "Portfolio Metrics:\n"
        report += f"Initial Capital: ${self.initial_capital:,.2f}\n"
        report += f"Final Capital: ${self.current_capital:,.2f}\n"
        report += f"Total Return: {self.metrics['total_return']:.2%}\n"
        report += f"Volatility (Ann.): {self.metrics['volatility']:.2%}\n"
        report += f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
        report += f"Max Drawdown: {self.metrics['max_drawdown']:.2%}\n\n"
        
        # Trading metrics
        report += "Trading Metrics:\n"
        report += f"Number of Trades: {self.metrics['num_trades']}\n"
        report += f"Win Rate: {self.metrics['win_rate']:.2%}\n"
        report += f"Profit Factor: {self.metrics['profit_factor']:.2f}\n"
        
        return report 