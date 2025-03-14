import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from models.ai.portfolio_optimizer import DeepPortfolioOptimizer
from models.ai.dynamic_hedging import DynamicHedgingAgent
from models.backtesting.engine import BacktestEngine
import matplotlib.pyplot as plt

def fetch_market_data(symbols: list, start_date: str, end_date: str) -> dict:
    """Fetches historical market data for given symbols"""
    data = {}
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        data[symbol] = df
    return data

def prepare_state_features(data: dict, timestamp: datetime) -> np.ndarray:
    """Prepares state features for the hedging agent"""
    features = []
    
    for symbol, df in data.items():
        # Price features
        returns = df['Close'].pct_change()
        vol = df['Close'].pct_change().rolling(20).std()
        rsi = calculate_rsi(df['Close'])
        
        # Get current values
        current_return = returns.loc[timestamp]
        current_vol = vol.loc[timestamp]
        current_rsi = rsi.loc[timestamp]
        
        features.extend([current_return, current_vol, current_rsi])
    
    return np.array(features)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def ai_trading_strategy(timestamp: datetime, data: dict, prices: dict, backtest: BacktestEngine):
    """AI-driven trading strategy combining portfolio optimization and dynamic hedging"""
    # Portfolio optimization
    portfolio_optimizer = DeepPortfolioOptimizer(n_assets=len(data))
    market_data = pd.DataFrame({symbol: df['Close'] for symbol, df in data.items()})
    optimal_weights = portfolio_optimizer.predict(market_data.iloc[-30:])  # Use last 30 days
    
    # Dynamic hedging
    state_dim = len(data) * 3  # 3 features per asset
    action_dim = 10  # Discrete hedging levels
    hedging_agent = DynamicHedgingAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Prepare state for hedging decision
    state = prepare_state_features(data, timestamp)
    hedge_ratio = hedging_agent.calculate_hedge_ratio(state)
    
    # Execute trades based on AI decisions
    current_positions = {symbol: pos.amount for symbol, pos in backtest.positions.items()}
    
    for i, (symbol, weight) in enumerate(zip(data.keys(), optimal_weights[-1])):
        target_position = backtest.current_capital * weight * (1 - hedge_ratio)
        current_position = current_positions.get(symbol, 0)
        
        # Calculate required trade
        price = prices[symbol]
        trade_amount = (target_position - current_position * price) / price
        
        if abs(trade_amount) > 0:
            side = "BUY" if trade_amount > 0 else "SELL"
            backtest.execute_trade(
                timestamp=timestamp,
                symbol=symbol,
                side=side,
                amount=abs(trade_amount),
                price=price
            )

def main():
    # Configuration
    symbols = ['SPY', 'QQQ', 'GLD', 'TLT', 'VGK']  # Example ETFs
    start_date = '2022-01-01'
    end_date = '2023-12-31'
    initial_capital = 1000000.0
    
    # Fetch market data
    print("Fetching market data...")
    data = fetch_market_data(symbols, start_date, end_date)
    
    # Initialize backtesting engine
    backtest = BacktestEngine(
        data=data,
        initial_capital=initial_capital,
        fee_rate=0.001,
        slippage=0.0002
    )
    
    # Run backtest
    print("Running backtest...")
    backtest.run_backtest(ai_trading_strategy)
    
    # Display results
    print("\nBacktest Results:")
    print(backtest.get_metrics_report())
    
    # Plot results
    backtest.plot_results()
    
    # Save trade history
    trade_history = backtest.get_trade_history()
    trade_history.to_csv('results/trade_history.csv', index=False)
    print("\nTrade history saved to results/trade_history.csv")

if __name__ == "__main__":
    main() 