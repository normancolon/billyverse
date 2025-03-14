import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass

from models.hft.base import BaseHFTStrategy, Signal, MarketState

logger = logging.getLogger("billieverse.models.hft.arbitrage")

@dataclass
class ArbitrageParams:
    """Parameters for arbitrage strategy"""
    min_spread: float = 0.0001  # Minimum spread to trade
    min_profit: float = 0.0002  # Minimum profit per trade
    max_position: float = 100.0  # Maximum position per venue
    trade_size: float = 1.0  # Base trade size
    execution_timeout: int = 100  # Timeout in milliseconds
    slippage_tolerance: float = 0.0001  # Maximum allowed slippage
    venue_costs: Dict[str, float] = None  # Trading costs per venue

@dataclass
class VenueState:
    """State of a trading venue"""
    venue_id: str
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    timestamp: datetime
    latency: float  # microseconds
    fees: float  # basis points
    liquidity: float  # Available liquidity score

class ArbitrageStrategy(BaseHFTStrategy):
    """Statistical arbitrage strategy for HFT"""
    
    def __init__(
        self,
        symbols: List[str],
        position_limits: Dict[str, float],
        risk_limits: Dict[str, float],
        venues: List[Dict],
        params: Optional[ArbitrageParams] = None,
        **kwargs
    ):
        super().__init__(symbols, position_limits, risk_limits, **kwargs)
        
        self.venues = venues
        self.params = params or ArbitrageParams()
        if self.params.venue_costs is None:
            self.params.venue_costs = {
                venue['id']: venue.get('trading_fee', 0.001)
                for venue in venues
            }
        
        # Initialize venue states
        self.venue_states: Dict[str, Dict[str, VenueState]] = {
            venue['id']: {} for venue in venues
        }
        
        # Initialize arbitrage tracking
        self.active_arbitrages: List[Dict] = []
        self.arbitrage_history: List[Dict] = []
    
    def _update_venue_state(
        self,
        venue_id: str,
        market_data: Dict
    ) -> None:
        """Update state for a specific venue"""
        try:
            symbol = market_data['symbol']
            
            state = VenueState(
                venue_id=venue_id,
                symbol=symbol,
                bid=market_data['bid'],
                ask=market_data['ask'],
                bid_size=market_data['bid_size'],
                ask_size=market_data['ask_size'],
                timestamp=datetime.now(),
                latency=market_data.get('latency', 1000),  # Default 1ms
                fees=self.params.venue_costs[venue_id],
                liquidity=min(market_data['bid_size'], market_data['ask_size'])
            )
            
            self.venue_states[venue_id][symbol] = state
            
        except Exception as e:
            logger.error(f"Error updating venue state: {str(e)}")
    
    def _find_arbitrage_opportunities(
        self,
        symbol: str
    ) -> List[Dict]:
        """Find arbitrage opportunities across venues"""
        try:
            opportunities = []
            
            # Get all venue pairs
            venues = list(self.venue_states.keys())
            for i in range(len(venues)):
                for j in range(i + 1, len(venues)):
                    venue1, venue2 = venues[i], venues[j]
                    
                    # Get venue states
                    state1 = self.venue_states[venue1].get(symbol)
                    state2 = self.venue_states[venue2].get(symbol)
                    
                    if not (state1 and state2):
                        continue
                    
                    # Calculate spreads
                    buy1_sell2 = state2.bid - state1.ask  # Buy on venue1, sell on venue2
                    buy2_sell1 = state1.bid - state2.ask  # Buy on venue2, sell on venue1
                    
                    # Calculate costs
                    total_cost = (
                        state1.fees + state2.fees +
                        self.params.slippage_tolerance
                    )
                    
                    # Check opportunities
                    if buy1_sell2 > self.params.min_spread:
                        profit = buy1_sell2 - total_cost
                        if profit > self.params.min_profit:
                            opportunities.append({
                                'type': 'buy1_sell2',
                                'buy_venue': venue1,
                                'sell_venue': venue2,
                                'buy_price': state1.ask,
                                'sell_price': state2.bid,
                                'spread': buy1_sell2,
                                'profit': profit,
                                'timestamp': datetime.now()
                            })
                    
                    if buy2_sell1 > self.params.min_spread:
                        profit = buy2_sell1 - total_cost
                        if profit > self.params.min_profit:
                            opportunities.append({
                                'type': 'buy2_sell1',
                                'buy_venue': venue2,
                                'sell_venue': venue1,
                                'buy_price': state2.ask,
                                'sell_price': state1.bid,
                                'spread': buy2_sell1,
                                'profit': profit,
                                'timestamp': datetime.now()
                            })
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage opportunities: {str(e)}")
            return []
    
    def _calculate_trade_size(
        self,
        opportunity: Dict,
        symbol: str
    ) -> float:
        """Calculate optimal trade size for arbitrage"""
        try:
            # Get venue states
            buy_state = self.venue_states[opportunity['buy_venue']][symbol]
            sell_state = self.venue_states[opportunity['sell_venue']][symbol]
            
            # Calculate size constraints
            max_buy = min(
                buy_state.ask_size,
                self.params.max_position - self.positions.get(symbol, 0)
            )
            max_sell = min(
                sell_state.bid_size,
                self.params.max_position + self.positions.get(symbol, 0)
            )
            
            # Calculate optimal size
            size = min(
                max_buy,
                max_sell,
                self.params.trade_size,
                opportunity['profit'] * 1000  # Scale size with profit
            )
            
            return max(size, 0)
            
        except Exception as e:
            logger.error(f"Error calculating trade size: {str(e)}")
            return 0
    
    def generate_signals(self) -> List[Signal]:
        """Generate arbitrage signals"""
        try:
            signals = []
            
            for symbol in self.symbols:
                # Find opportunities
                opportunities = self._find_arbitrage_opportunities(symbol)
                
                for opp in opportunities:
                    # Calculate trade size
                    size = self._calculate_trade_size(opp, symbol)
                    if size <= 0:
                        continue
                    
                    # Generate signals
                    buy_signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type='arbitrage_buy',
                        direction=1,
                        strength=opp['profit'] / self.params.min_profit,
                        confidence=min(1.0, opp['spread'] / self.params.min_spread),
                        horizon=self.params.execution_timeout,
                        metadata={
                            'venue': opp['buy_venue'],
                            'price': opp['buy_price'],
                            'size': size,
                            'opportunity_id': id(opp)
                        }
                    )
                    
                    sell_signal = Signal(
                        timestamp=datetime.now(),
                        symbol=symbol,
                        signal_type='arbitrage_sell',
                        direction=-1,
                        strength=opp['profit'] / self.params.min_profit,
                        confidence=min(1.0, opp['spread'] / self.params.min_spread),
                        horizon=self.params.execution_timeout,
                        metadata={
                            'venue': opp['sell_venue'],
                            'price': opp['sell_price'],
                            'size': size,
                            'opportunity_id': id(opp)
                        }
                    )
                    
                    signals.extend([buy_signal, sell_signal])
                    
                    # Track arbitrage
                    self.active_arbitrages.append({
                        'id': id(opp),
                        'opportunity': opp,
                        'size': size,
                        'status': 'pending',
                        'start_time': datetime.now()
                    })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def execute_signals(self, signals: List[Signal]) -> None:
        """Execute arbitrage signals"""
        try:
            start_time = datetime.now()
            
            # Group signals by opportunity
            signal_groups = {}
            for signal in signals:
                opp_id = signal.metadata['opportunity_id']
                if opp_id not in signal_groups:
                    signal_groups[opp_id] = []
                signal_groups[opp_id].append(signal)
            
            # Execute each opportunity
            for opp_id, opp_signals in signal_groups.items():
                # Find arbitrage record
                arb = next(
                    (a for a in self.active_arbitrages if a['id'] == opp_id),
                    None
                )
                if not arb:
                    continue
                
                # Execute trades
                executed_qty = 0
                avg_buy_price = 0
                avg_sell_price = 0
                
                for signal in opp_signals:
                    # In practice, send orders to venues
                    executed_qty = signal.metadata['size']
                    
                    if signal.signal_type == 'arbitrage_buy':
                        avg_buy_price = signal.metadata['price']
                    else:
                        avg_sell_price = signal.metadata['price']
                    
                    # Update positions
                    self.positions[signal.symbol] += (
                        executed_qty if signal.direction > 0 else -executed_qty
                    )
                
                # Calculate metrics
                latency = (
                    datetime.now() - start_time
                ).total_seconds() * 1_000_000  # μs
                
                profit = executed_qty * (avg_sell_price - avg_buy_price)
                
                # Update arbitrage record
                arb['status'] = 'executed'
                arb['executed_qty'] = executed_qty
                arb['profit'] = profit
                arb['latency'] = latency
                arb['end_time'] = datetime.now()
                
                # Move to history
                self.arbitrage_history.append(arb)
                self.active_arbitrages.remove(arb)
                
                # Update metrics
                self.update_metrics(
                    trade_qty=executed_qty * 2,  # Count both legs
                    trade_price=(avg_buy_price + avg_sell_price) / 2,
                    trade_type='arbitrage',
                    latency=latency
                )
            
        except Exception as e:
            logger.error(f"Error executing signals: {str(e)}")
    
    def handle_market_data(self, market_data: Dict) -> None:
        """Handle incoming market data"""
        try:
            # Update venue state
            venue_id = market_data['venue_id']
            self._update_venue_state(venue_id, market_data)
            
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
            # Update positions based on fills
            if order_update['status'] == 'filled':
                symbol = order_update['symbol']
                filled_qty = order_update['filled_quantity']
                side_multiplier = 1 if order_update['side'] == 'buy' else -1
                self.positions[symbol] += filled_qty * side_multiplier
                
                # Update arbitrage status if applicable
                opp_id = order_update.get('opportunity_id')
                if opp_id:
                    arb = next(
                        (a for a in self.active_arbitrages if a['id'] == opp_id),
                        None
                    )
                    if arb:
                        arb['status'] = 'partial' if arb['status'] == 'pending' else 'completed'
            
        except Exception as e:
            logger.error(f"Error handling order update: {str(e)}") 