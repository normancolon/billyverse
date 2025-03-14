import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
import logging

logger = logging.getLogger("billieverse.processing")

class MarketDataProcessor:
    """Process and engineer features from market data"""
    
    def __init__(self):
        self.logger = logger
    
    def process_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process OHLCV data and add technical indicators"""
        try:
            # Create copy to avoid modifying original data
            df = df.copy()
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            df['price_range'] = df['high'] - df['low']
            df['price_range_pct'] = df['price_range'] / df['close']
            
            # Volume features
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume_std']
            
            # Trend indicators
            sma = SMAIndicator(close=df['close'], window=20)
            ema = EMAIndicator(close=df['close'], window=20)
            macd = MACD(close=df['close'])
            
            df['sma'] = sma.sma_indicator()
            df['ema'] = ema.ema_indicator()
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            
            # Momentum indicators
            rsi = RSIIndicator(close=df['close'])
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            
            df['rsi'] = rsi.rsi()
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volatility indicators
            bb = BollingerBands(close=df['close'])
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
            
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
            df['atr'] = atr.average_true_range()
            
            # Volume price indicators
            vwap = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            df['vwap'] = vwap.volume_weighted_average_price()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data: {str(e)}")
            raise
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """Normalize features using specified method"""
        try:
            df = df.copy()
            
            if columns is None:
                # Exclude non-numeric columns and timestamp index
                columns = df.select_dtypes(include=[np.number]).columns
            
            if method == 'zscore':
                for col in columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[f'{col}_norm'] = (df[col] - mean) / std
            
            elif method == 'minmax':
                for col in columns:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error normalizing features: {str(e)}")
            raise
    
    def process_orderbook(self, orderbook: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book data and extract features"""
        try:
            bids = np.array(orderbook['bids'])
            asks = np.array(orderbook['asks'])
            
            bid_prices = bids[:, 0]
            bid_volumes = bids[:, 1]
            ask_prices = asks[:, 0]
            ask_volumes = asks[:, 1]
            
            features = {
                'symbol': orderbook['symbol'],
                'timestamp': orderbook['timestamp'],
                'bid_ask_spread': ask_prices[0] - bid_prices[0],
                'bid_ask_spread_pct': (ask_prices[0] - bid_prices[0]) / bid_prices[0],
                'total_bid_volume': np.sum(bid_volumes),
                'total_ask_volume': np.sum(ask_volumes),
                'volume_imbalance': np.sum(bid_volumes) - np.sum(ask_volumes),
                'mid_price': (ask_prices[0] + bid_prices[0]) / 2,
                'weighted_mid_price': (
                    np.sum(bid_prices * bid_volumes) + np.sum(ask_prices * ask_volumes)
                ) / (np.sum(bid_volumes) + np.sum(ask_volumes))
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook: {str(e)}")
            raise
    
    def calculate_market_impact(
        self,
        orderbook: Dict[str, Any],
        trade_size: float
    ) -> Dict[str, float]:
        """Calculate potential market impact for a given trade size"""
        try:
            bids = np.array(orderbook['bids'])
            asks = np.array(orderbook['asks'])
            
            def calculate_impact(orders: np.ndarray, size: float, side: str) -> float:
                cumulative_volume = np.cumsum(orders[:, 1])
                impact_idx = np.searchsorted(cumulative_volume, size)
                
                if impact_idx >= len(orders):
                    return float('inf')
                
                if side == 'buy':
                    return (orders[impact_idx, 0] - orders[0, 0]) / orders[0, 0]
                else:
                    return (orders[0, 0] - orders[impact_idx, 0]) / orders[0, 0]
            
            return {
                'buy_impact': calculate_impact(asks, trade_size, 'buy'),
                'sell_impact': calculate_impact(bids, trade_size, 'sell')
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {str(e)}")
            raise 