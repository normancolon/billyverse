import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from models.portfolio.risk import RiskManager, RiskMetrics, HedgeParameters
from models.portfolio.risk_mitigation import (
    StopLossRL,
    DerivativesHedging,
    AIPositionSizing,
    StopLossState
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.risk_mitigation")

def fetch_market_data(
    symbols: list,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """Fetch historical market data"""
    try:
        # Download data
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False
        )
        
        # Calculate returns
        returns = data['Adj Close'].pct_change().dropna()
        
        return returns, data['Adj Close']
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return None, None

def create_vol_surface(
    underlying_prices: pd.Series,
    base_vol: float = 0.2
) -> pd.DataFrame:
    """Create simple volatility surface for example"""
    try:
        current_price = underlying_prices.iloc[-1]
        expiries = [30, 60, 90]  # Days to expiry
        strikes = np.linspace(0.8 * current_price, 1.2 * current_price, 5)
        
        surface_data = []
        for dte in expiries:
            for strike in strikes:
                # Simple vol smile
                moneyness = np.log(strike / current_price)
                implied_vol = base_vol + 0.05 * moneyness**2
                
                surface_data.append({
                    'days_to_expiry': dte,
                    'strike': strike,
                    'implied_vol': implied_vol,
                    'underlying_price': current_price,
                    'time_to_expiry': dte / 365.0
                })
        
        return pd.DataFrame(surface_data)
        
    except Exception as e:
        logger.error(f"Error creating vol surface: {str(e)}")
        return pd.DataFrame()

def run_stop_loss_example():
    """Example using RL-based stop-loss"""
    try:
        # Define portfolio
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Fetch data
        returns, prices = fetch_market_data(
            symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None or prices is None:
            return
        
        # Initialize stop-loss model
        stop_loss_model = StopLossRL(
            state_dim=8,
            action_dim=3,
            learning_rate=0.001
        )
        
        # Create training data
        training_data = []
        for symbol in symbols:
            price_series = prices[symbol]
            returns_series = returns[symbol]
            
            for i in range(100, len(price_series)):
                # Calculate features
                current_price = price_series.iloc[i]
                volatility = returns_series.iloc[i-20:i].std()
                trend = (
                    price_series.iloc[i] /
                    price_series.iloc[i-20] - 1
                )
                drawdown = (
                    current_price /
                    price_series.iloc[i-20:i].max() - 1
                )
                
                # Create state
                state = StopLossState(
                    price=current_price,
                    volatility=volatility,
                    trend_strength=trend,
                    drawdown=drawdown,
                    profit_loss=returns_series.iloc[i],
                    time_in_trade=20,
                    market_regime=0,
                    risk_score=volatility
                )
                
                # Simple reward function
                reward = -abs(drawdown) if drawdown < -0.05 else 0.01
                
                # Add to training data
                next_state = StopLossState(
                    price=price_series.iloc[i+1],
                    volatility=volatility,
                    trend_strength=trend,
                    drawdown=drawdown,
                    profit_loss=returns_series.iloc[i+1],
                    time_in_trade=21,
                    market_regime=0,
                    risk_score=volatility
                )
                
                training_data.append((state, 0, reward, next_state, False))
        
        # Train model
        stop_loss_model.replay_buffer = training_data[-10000:]
        losses = []
        for _ in range(100):
            loss = stop_loss_model.train(batch_size=32)
            losses.append(loss)
            if _ % 10 == 0:
                stop_loss_model.update_target_network()
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Stop-Loss Model Training Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig("results/stop_loss_training.png")
        plt.close()
        
        # Test model
        test_results = []
        for symbol in symbols:
            price_series = prices[symbol].iloc[-100:]
            returns_series = returns[symbol].iloc[-100:]
            
            positions = []
            for i in range(20, len(price_series)):
                state = StopLossState(
                    price=price_series.iloc[i],
                    volatility=returns_series.iloc[i-20:i].std(),
                    trend_strength=price_series.iloc[i] / price_series.iloc[i-20] - 1,
                    drawdown=price_series.iloc[i] / price_series.iloc[i-20:i].max() - 1,
                    profit_loss=returns_series.iloc[i],
                    time_in_trade=20,
                    market_regime=0,
                    risk_score=returns_series.iloc[i-20:i].std()
                )
                
                action = stop_loss_model.get_action(state)
                positions.append(action)
            
            test_results.append({
                'symbol': symbol,
                'positions': positions,
                'returns': returns_series.iloc[-80:].values
            })
        
        # Plot test results
        plt.figure(figsize=(15, 10))
        for i, result in enumerate(test_results):
            plt.subplot(len(test_results), 1, i+1)
            plt.plot(result['returns'].cumsum(), label='Returns')
            plt.plot(
                np.array(result['positions']) == 2,
                'r.',
                label='Exit Signals'
            )
            plt.title(f"Stop-Loss Signals - {result['symbol']}")
            plt.legend()
        plt.tight_layout()
        plt.savefig("results/stop_loss_signals.png")
        plt.close()
        
        return stop_loss_model
        
    except Exception as e:
        logger.error(f"Error in stop-loss example: {str(e)}")
        return None

def run_hedging_example(stop_loss_model: StopLossRL = None):
    """Example using derivatives hedging"""
    try:
        # Define portfolio
        symbols = ['SPY', 'QQQ', 'IWM']  # Use ETFs for derivatives example
        
        # Fetch data
        returns, prices = fetch_market_data(
            symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None or prices is None:
            return
        
        # Initialize risk manager
        risk_manager = RiskManager(
            returns=returns,
            benchmark_returns=returns['SPY'],
            risk_limits={
                'max_var_95': 0.02,
                'max_leverage': 1.5,
                'max_concentration': 0.3
            }
        )
        
        # Create volatility surfaces
        vol_surface = {
            symbol: create_vol_surface(prices[symbol])
            for symbol in symbols
        }
        
        # Initialize hedging
        hedging = DerivativesHedging(
            risk_manager=risk_manager,
            vol_surface=vol_surface,
            min_hedge_ratio=0.5,
            max_hedge_ratio=0.8
        )
        
        # Test hedging over time
        hedge_results = []
        for i in range(100, len(returns), 5):  # Check every 5 days
            # Calculate portfolio metrics
            window_returns = returns.iloc[i-100:i]
            portfolio_weights = {
                symbol: 1/len(symbols)
                for symbol in symbols
            }
            
            metrics = risk_manager.calculate_risk_metrics(portfolio_weights)
            if metrics is None:
                continue
            
            # Update hedges
            new_hedges = hedging.update_hedges(metrics)
            
            # Track results
            hedge_results.append({
                'date': returns.index[i],
                'portfolio_delta': hedging.greek_history[-1]['portfolio_delta'],
                'hedge_delta': hedging.greek_history[-1]['hedge_delta'],
                'portfolio_vega': hedging.greek_history[-1]['portfolio_vega'],
                'hedge_vega': hedging.greek_history[-1]['hedge_vega']
            })
        
        # Plot hedging results
        results_df = pd.DataFrame(hedge_results)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(results_df['date'], results_df['portfolio_delta'], label='Portfolio Delta')
        plt.plot(results_df['date'], results_df['hedge_delta'], label='Hedge Delta')
        plt.title("Delta Hedging")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(results_df['date'], results_df['portfolio_vega'], label='Portfolio Vega')
        plt.plot(results_df['date'], results_df['hedge_vega'], label='Hedge Vega')
        plt.title("Vega Hedging")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("results/hedging_results.png")
        plt.close()
        
        return risk_manager, hedging
        
    except Exception as e:
        logger.error(f"Error in hedging example: {str(e)}")
        return None, None

def run_position_sizing_example(
    stop_loss_model: StopLossRL,
    risk_manager: RiskManager
):
    """Example using AI position sizing"""
    try:
        # Define portfolio
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        
        # Fetch data
        returns, prices = fetch_market_data(
            symbols,
            '2020-01-01',
            datetime.now().strftime('%Y-%m-%d')
        )
        
        if returns is None or prices is None:
            return
        
        # Initialize position sizing
        position_sizer = AIPositionSizing(
            risk_manager=risk_manager,
            stop_loss_model=stop_loss_model,
            max_position_size=0.3,
            min_position_size=0.05,
            kelly_fraction=0.5
        )
        
        # Test position sizing
        sizing_results = []
        for i in range(100, len(returns), 5):  # Check every 5 days
            # Generate mock signals
            signals = {}
            for symbol in symbols:
                price_series = prices[symbol].iloc[i-100:i]
                returns_series = returns[symbol].iloc[i-100:i]
                
                # Calculate signal features
                volatility = returns_series.std()
                trend = price_series.iloc[-1] / price_series.iloc[0] - 1
                
                # Simple signal generation
                confidence = np.random.uniform(0.5, 0.9)
                if trend > 0:
                    confidence *= (1 + trend)
                else:
                    confidence *= (1 - abs(trend))
                
                signals[symbol] = {
                    'confidence': min(confidence, 0.95),
                    'price': price_series.iloc[-1],
                    'volatility': volatility,
                    'trend_strength': trend
                }
            
            # Get position sizes
            new_positions = position_sizer.update_positions(
                portfolio={symbol: 0.2 for symbol in symbols},
                signals=signals
            )
            
            # Track results
            sizing_results.append({
                'date': returns.index[i],
                **new_positions
            })
        
        # Plot position sizing results
        results_df = pd.DataFrame(sizing_results)
        
        plt.figure(figsize=(12, 6))
        plt.stackplot(
            results_df['date'],
            [results_df[col] for col in symbols],
            labels=symbols
        )
        plt.title("AI Position Sizing")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("results/position_sizing.png")
        plt.close()
        
        # Plot sizing metrics distribution
        metrics_df = pd.DataFrame(position_sizer.sizing_metrics)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.boxplot(data=metrics_df, x='asset', y='base_size')
        plt.title("Position Size Distribution")
        plt.xticks(rotation=45)
        
        plt.subplot(1, 3, 2)
        sns.scatterplot(
            data=metrics_df,
            x='volatility',
            y='base_size',
            hue='stop_loss_action'
        )
        plt.title("Size vs Volatility")
        
        plt.subplot(1, 3, 3)
        sns.scatterplot(
            data=metrics_df,
            x='trend_strength',
            y='base_size',
            hue='stop_loss_action'
        )
        plt.title("Size vs Trend")
        
        plt.tight_layout()
        plt.savefig("results/sizing_analysis.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in position sizing example: {str(e)}")

def main():
    """Run all risk mitigation examples"""
    try:
        # Create results directory
        Path("results").mkdir(exist_ok=True)
        
        # Run examples
        logger.info("Training stop-loss model...")
        stop_loss_model = run_stop_loss_example()
        
        if stop_loss_model:
            logger.info("\nTesting derivatives hedging...")
            risk_manager, hedging = run_hedging_example(stop_loss_model)
            
            if risk_manager and hedging:
                logger.info("\nTesting position sizing...")
                run_position_sizing_example(stop_loss_model, risk_manager)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 