import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import logging
from dotenv import load_dotenv

from models.ai.portfolio_optimizer import DeepPortfolioOptimizer
from models.ai.dynamic_hedging import DynamicHedgingAgent
from models.exchange.connector import ExchangeConnector
from models.risk.manager import RiskManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveTradingSystem:
    def __init__(self,
                 exchange_id: str,
                 symbols: list,
                 initial_capital: float,
                 model_path: str = 'models/saved'):
        """
        Live trading system integrating all components
        
        Args:
            exchange_id: Exchange identifier
            symbols: List of trading symbols
            initial_capital: Initial capital
            model_path: Path to saved models
        """
        # Load environment variables
        load_dotenv()
        
        # Initialize components
        self.exchange = ExchangeConnector(
            exchange_id=exchange_id,
            api_key=os.getenv('EXCHANGE_API_KEY'),
            api_secret=os.getenv('EXCHANGE_API_SECRET'),
            sandbox=True  # Use sandbox for testing
        )
        
        self.portfolio_optimizer = DeepPortfolioOptimizer(n_assets=len(symbols))
        self.portfolio_optimizer.load_model(f"{model_path}/portfolio_optimizer")
        
        self.hedging_agent = DynamicHedgingAgent(
            state_dim=len(symbols) * 3,
            action_dim=10
        )
        self.hedging_agent.load_model(f"{model_path}/hedging_agent")
        
        self.risk_manager = RiskManager(
            initial_capital=initial_capital,
            max_position_pct=0.2,
            max_portfolio_var=0.02,
            max_leverage=2.0
        )
        
        self.symbols = symbols
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, float] = {}
        
    async def update_market_data(self):
        """Updates market data for all symbols"""
        for symbol in self.symbols:
            try:
                # Get historical data
                data = await self.exchange.get_historical_data(
                    symbol=symbol,
                    timeframe='1h',
                    limit=1000
                )
                self.market_data[symbol] = data
                
                # Get current ticker
                ticker = await self.exchange.get_ticker(symbol)
                
                # Update latest price
                self.market_data[symbol].loc[datetime.now()] = {
                    'open': ticker['last'],
                    'high': ticker['last'],
                    'low': ticker['last'],
                    'close': ticker['last'],
                    'volume': ticker['volume']
                }
                
            except Exception as e:
                logger.error(f"Failed to update market data for {symbol}: {str(e)}")
                
    async def update_positions(self):
        """Updates current positions"""
        try:
            balances = await self.exchange.get_account_balance()
            self.positions = {
                symbol: float(balances.get(symbol.split('/')[0], 0))
                for symbol in self.symbols
            }
        except Exception as e:
            logger.error(f"Failed to update positions: {str(e)}")
            
    def prepare_features(self, symbol: str) -> np.ndarray:
        """Prepares features for AI models"""
        data = self.market_data[symbol]
        
        # Price features
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        rsi = self.calculate_rsi(data['close'])
        
        features = np.array([
            returns.iloc[-1],
            volatility.iloc[-1],
            rsi.iloc[-1]
        ])
        
        return features
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculates RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    async def execute_trades(self, optimal_weights: np.ndarray, hedge_ratio: float):
        """Executes trades based on AI decisions"""
        current_prices = {
            symbol: self.market_data[symbol]['close'].iloc[-1]
            for symbol in self.symbols
        }
        
        for i, (symbol, weight) in enumerate(zip(self.symbols, optimal_weights)):
            try:
                # Calculate target position
                price = current_prices[symbol]
                target_value = self.risk_manager.initial_capital * weight * (1 - hedge_ratio)
                target_position = target_value / price
                current_position = self.positions.get(symbol, 0)
                
                # Calculate required trade
                trade_amount = target_position - current_position
                
                if abs(trade_amount) > 0:
                    # Check risk limits
                    side = "buy" if trade_amount > 0 else "sell"
                    approved, message = self.risk_manager.check_trade(
                        symbol=symbol,
                        side=side,
                        amount=abs(trade_amount),
                        price=price,
                        current_positions=self.positions
                    )
                    
                    if approved:
                        # Execute trade
                        order = await self.exchange.place_order(
                            symbol=symbol,
                            side=side,
                            amount=abs(trade_amount),
                            price=price,
                            order_type='limit'
                        )
                        logger.info(f"Executed trade: {order}")
                    else:
                        logger.warning(f"Trade rejected: {message}")
                        
            except Exception as e:
                logger.error(f"Failed to execute trade for {symbol}: {str(e)}")
                
    async def trading_loop(self):
        """Main trading loop"""
        while True:
            try:
                # Update data
                await self.update_market_data()
                await self.update_positions()
                
                # Update risk limits
                self.risk_manager.update_position_limits(self.market_data)
                
                # Calculate portfolio metrics
                metrics = self.risk_manager.calculate_portfolio_metrics(
                    self.positions,
                    self.market_data
                )
                logger.info("\n" + self.risk_manager.get_metrics_report())
                
                # Get AI decisions
                market_data = pd.DataFrame({
                    symbol: df['close'] for symbol, df in self.market_data.items()
                })
                optimal_weights = self.portfolio_optimizer.predict(market_data.iloc[-30:])
                
                # Prepare state for hedging
                state = np.concatenate([
                    self.prepare_features(symbol)
                    for symbol in self.symbols
                ])
                hedge_ratio = self.hedging_agent.calculate_hedge_ratio(state)
                
                # Execute trades
                await self.execute_trades(optimal_weights[-1], hedge_ratio)
                
                # Wait for next iteration
                await asyncio.sleep(60)  # 1-minute interval
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(60)
                
async def main():
    # Configuration
    exchange_id = 'binance'
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    initial_capital = 100000.0
    
    # Initialize trading system
    system = LiveTradingSystem(
        exchange_id=exchange_id,
        symbols=symbols,
        initial_capital=initial_capital
    )
    
    # Run trading loop
    logger.info("Starting live trading system...")
    await system.trading_loop()

if __name__ == "__main__":
    asyncio.run(main()) 