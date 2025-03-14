import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from models.hft.base import BaseHFTStrategy, Signal, MarketState

logger = logging.getLogger("billieverse.models.hft.market_maker")

@dataclass
class MarketMakingParams:
    """Parameters for market making strategy"""
    spread_multiplier: float = 1.2  # Target spread as multiple of market spread
    inventory_target: float = 0.0  # Target inventory position
    inventory_limit: float = 100.0  # Maximum allowed inventory
    min_spread: float = 0.0001  # Minimum spread to quote
    max_spread: float = 0.01  # Maximum spread to quote
    quote_size: float = 1.0  # Base quote size
    skew_factor: float = 0.5  # How much to skew quotes based on inventory
    mean_reversion_factor: float = 0.1  # How much to adjust based on VWAP
    volatility_adjustment: float = 1.0  # How much to widen spreads in volatility
    order_imbalance_factor: float = 0.2  # How much to adjust based on imbalance

class MarketMaker(BaseHFTStrategy):
    """Market making strategy for HFT"""
    
    def __init__(
        self,
        symbols: List[str],
        position_limits: Dict[str, float],
        risk_limits: Dict[str, float],
        params: Optional[MarketMakingParams] = None,
        **kwargs
    ):
        super().__init__(symbols, position_limits, risk_limits, **kwargs)
        
        self.params = params or MarketMakingParams()
        
        # Initialize quote tracking
        self.active_quotes: Dict[str, Dict] = {}
        self.quote_history: List[Dict] = []
    
    def _calculate_quote_prices(
        self,
        market_state: MarketState
    ) -> Tuple[float, float]:
        """Calculate bid and ask prices for quotes"""
        try:
            # Get base spread
            base_spread = market_state.spread * self.params.spread_multiplier
            
            # Adjust spread based on volatility
            volatility_spread = base_spread * (
                1 + market_state.volatility * self.params.volatility_adjustment
            )
            
            # Ensure spread is within limits
            final_spread = np.clip(
                volatility_spread,
                self.params.min_spread,
                self.params.max_spread
            )
            
            # Calculate mid price adjustments
            inventory_adjustment = (
                self.positions[market_state.symbol] -
                self.params.inventory_target
            ) * self.params.skew_factor
            
            mean_reversion = (
                market_state.vwap - market_state.mid_price
            ) * self.params.mean_reversion_factor
            
            imbalance_adjustment = (
                market_state.order_imbalance *
                self.params.order_imbalance_factor *
                market_state.mid_price
            )
            
            # Calculate final mid price
            adjusted_mid = (
                market_state.mid_price +
                mean_reversion +
                imbalance_adjustment -
                inventory_adjustment
            )
            
            # Calculate quote prices
            bid_price = adjusted_mid - final_spread / 2
            ask_price = adjusted_mid + final_spread / 2
            
            return bid_price, ask_price
            
        except Exception as e:
            logger.error(f"Error calculating quote prices: {str(e)}")
            return market_state.bid, market_state.ask
    
    def _calculate_quote_sizes(
        self,
        market_state: MarketState,
        side: str
    ) -> float:
        """Calculate quote size based on market conditions"""
        try:
            # Get base quote size
            base_size = self.params.quote_size
            
            # Adjust for inventory
            inventory_ratio = abs(
                self.positions[market_state.symbol] /
                self.params.inventory_limit
            )
            inventory_factor = 1 - inventory_ratio
            
            # Adjust for market conditions
            if side == 'buy':
                # Increase buy size when price is below VWAP
                price_factor = max(
                    0,
                    (market_state.vwap - market_state.mid_price) /
                    market_state.vwap
                )
                # Increase buy size when ask liquidity is low
                liquidity_factor = market_state.bid_size / (
                    market_state.bid_size + market_state.ask_size
                )
            else:
                # Increase sell size when price is above VWAP
                price_factor = max(
                    0,
                    (market_state.mid_price - market_state.vwap) /
                    market_state.vwap
                )
                # Increase sell size when bid liquidity is low
                liquidity_factor = market_state.ask_size / (
                    market_state.bid_size + market_state.ask_size
                )
            
            # Calculate final size
            quote_size = base_size * (
                1 +
                inventory_factor +
                price_factor +
                liquidity_factor
            )
            
            return quote_size
            
        except Exception as e:
            logger.error(f"Error calculating quote size: {str(e)}")
            return self.params.quote_size
    
    def _should_cancel_quotes(
        self,
        market_state: MarketState
    ) -> bool:
        """Determine if quotes should be cancelled"""
        try:
            # Check inventory limits
            if abs(self.positions[market_state.symbol]) >= self.params.inventory_limit:
                return True
            
            # Check spread width
            if market_state.spread >= self.params.max_spread:
                return True
            
            # Check volatility
            if market_state.volatility >= 0.01:  # 1% volatility threshold
                return True
            
            # Check order imbalance
            if abs(market_state.order_imbalance) >= 0.7:  # 70% imbalance threshold
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking quote cancellation: {str(e)}")
            return True
    
    def generate_signals(self) -> List[Signal]:
        """Generate market making signals"""
        try:
            signals = []
            
            for symbol in self.symbols:
                if not self.market_states[symbol]:
                    continue
                
                market_state = self.market_states[symbol][-1]
                
                # Check if we should cancel quotes
                if self._should_cancel_quotes(market_state):
                    if symbol in self.active_quotes:
                        signals.append(Signal(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            signal_type='cancel_quotes',
                            direction=0,
                            strength=1.0,
                            confidence=1.0,
                            horizon=0,
                            metadata={'quotes': self.active_quotes[symbol]}
                        ))
                    continue
                
                # Calculate quote prices
                bid_price, ask_price = self._calculate_quote_prices(market_state)
                
                # Calculate quote sizes
                bid_size = self._calculate_quote_sizes(market_state, 'buy')
                ask_size = self._calculate_quote_sizes(market_state, 'sell')
                
                # Generate quote signals
                quotes = {
                    'bid': {'price': bid_price, 'size': bid_size},
                    'ask': {'price': ask_price, 'size': ask_size}
                }
                
                signals.append(Signal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type='market_making',
                    direction=0,
                    strength=1.0,
                    confidence=1.0,
                    horizon=100,  # 100ms quote lifetime
                    metadata={'quotes': quotes}
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def execute_signals(self, signals: List[Signal]) -> None:
        """Execute market making signals"""
        try:
            start_time = datetime.now()
            
            for signal in signals:
                if signal.signal_type == 'cancel_quotes':
                    # Cancel existing quotes
                    if signal.symbol in self.active_quotes:
                        # In practice, send cancellation to exchange
                        del self.active_quotes[signal.symbol]
                
                elif signal.signal_type == 'market_making':
                    quotes = signal.metadata['quotes']
                    
                    # Place new quotes
                    self.active_quotes[signal.symbol] = {
                        'bid': {
                            'price': quotes['bid']['price'],
                            'size': quotes['bid']['size'],
                            'timestamp': datetime.now()
                        },
                        'ask': {
                            'price': quotes['ask']['price'],
                            'size': quotes['ask']['size'],
                            'timestamp': datetime.now()
                        }
                    }
                    
                    # Record quotes
                    self.quote_history.append({
                        'timestamp': datetime.now(),
                        'symbol': signal.symbol,
                        'bid_price': quotes['bid']['price'],
                        'bid_size': quotes['bid']['size'],
                        'ask_price': quotes['ask']['price'],
                        'ask_size': quotes['ask']['size']
                    })
                    
                    # Calculate and update metrics
                    latency = (
                        datetime.now() - start_time
                    ).total_seconds() * 1_000_000  # μs
                    
                    self.update_metrics(
                        trade_qty=quotes['bid']['size'] + quotes['ask']['size'],
                        trade_price=(
                            quotes['bid']['price'] + quotes['ask']['price']
                        ) / 2,
                        trade_type='quote',
                        latency=latency
                    )
            
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
    
    def handle_market_data(self, market_data: Dict) -> None:
        """Handle incoming market data"""
        try:
            # Update market state
            self.update_market_state(
                symbol=market_data['symbol'],
                bid=market_data['bid'],
                ask=market_data['ask'],
                bid_size=market_data['bid_size'],
                ask_size=market_data['ask_size'],
                last_price=market_data['last_price'],
                volume=market_data['volume']
            )
            
            # Generate and execute signals
            signals = self.generate_signals()
            if signals:
                self.execute_signals(signals)
            
        except Exception as e:
            logger.error(f"Error handling market data: {str(e)}")
    
    def handle_order_update(self, order_update: Dict) -> None:
        """Handle order updates"""
        try:
            symbol = order_update['symbol']
            
            # Update positions based on fills
            if order_update['status'] == 'filled':
                filled_qty = order_update['filled_quantity']
                side_multiplier = 1 if order_update['side'] == 'buy' else -1
                self.positions[symbol] += filled_qty * side_multiplier
                
                # Remove filled quotes
                if symbol in self.active_quotes:
                    side = order_update['side']
                    if side == 'buy':
                        del self.active_quotes[symbol]['bid']
                    else:
                        del self.active_quotes[symbol]['ask']
                    
                    # Remove symbol if no quotes left
                    if not self.active_quotes[symbol]:
                        del self.active_quotes[symbol]
            
        except Exception as e:
            logger.error(f"Error handling order update: {str(e)}") 