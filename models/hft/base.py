import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import asyncio
from collections import deque

logger = logging.getLogger("billieverse.models.hft")

@dataclass
class MarketState:
    """Container for market state information"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    vwap: float
    order_imbalance: float
    volatility: float
    spread: float
    mid_price: float
    microprice: float  # Volume-weighted mid price

@dataclass
class Signal:
    """Container for trading signals"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'market_making', 'arbitrage', 'momentum', etc.
    direction: int  # 1 for long, -1 for short, 0 for neutral
    strength: float  # Signal strength (0 to 1)
    confidence: float  # Signal confidence (0 to 1)
    horizon: int  # Expected holding period in milliseconds
    metadata: Dict  # Additional signal information

@dataclass
class HFTMetrics:
    """Container for HFT performance metrics"""
    total_trades: int
    total_volume: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_holding_time: float  # milliseconds
    avg_latency: float  # microseconds
    inventory_time: float  # Average time holding inventory
    execution_quality: float  # Implementation shortfall
    market_impact: float  # Average price impact
    fill_rate: float  # Order fill rate

class BaseHFTStrategy(ABC):
    """Base class for HFT strategies"""
    
    def __init__(
        self,
        symbols: List[str],
        position_limits: Dict[str, float],
        risk_limits: Dict[str, float],
        latency_threshold: float = 1000,  # microseconds
        market_data_buffer: int = 1000,
        signal_buffer: int = 100
    ):
        self.symbols = symbols
        self.position_limits = position_limits
        self.risk_limits = risk_limits
        self.latency_threshold = latency_threshold
        
        # Initialize storage
        self.market_states: Dict[str, deque] = {
            symbol: deque(maxlen=market_data_buffer)
            for symbol in symbols
        }
        self.signals: deque = deque(maxlen=signal_buffer)
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in symbols}
        self.metrics = HFTMetrics(
            total_trades=0,
            total_volume=0.0,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            avg_holding_time=0.0,
            avg_latency=0.0,
            inventory_time=0.0,
            execution_quality=0.0,
            market_impact=0.0,
            fill_rate=0.0
        )
        
        # Initialize performance tracking
        self.pnl_history: List[float] = []
        self.latency_history: List[float] = []
        self.trade_history: List[Dict] = []
        
        self.logger = logger
        
        # Start monitoring tasks
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start monitoring tasks"""
        loop = asyncio.get_event_loop()
        loop.create_task(self._monitor_latency())
        loop.create_task(self._monitor_risk())
    
    async def _monitor_latency(self):
        """Monitor and log latency"""
        while True:
            try:
                if self.latency_history:
                    avg_latency = np.mean(self.latency_history[-100:])
                    if avg_latency > self.latency_threshold:
                        self.logger.warning(
                            f"High latency detected: {avg_latency:.2f} μs"
                        )
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring latency: {str(e)}")
    
    async def _monitor_risk(self):
        """Monitor risk limits"""
        while True:
            try:
                # Check position limits
                for symbol, position in self.positions.items():
                    if abs(position) > self.position_limits[symbol]:
                        self.logger.warning(
                            f"Position limit exceeded for {symbol}: {position}"
                        )
                
                # Check PnL drawdown
                if self.pnl_history:
                    current_drawdown = (
                        max(self.pnl_history) - self.pnl_history[-1]
                    ) / max(abs(max(self.pnl_history)), 1e-6)
                    
                    if current_drawdown > self.risk_limits['max_drawdown']:
                        self.logger.warning(
                            f"Max drawdown exceeded: {current_drawdown:.2%}"
                        )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring risk: {str(e)}")
    
    def update_market_state(
        self,
        symbol: str,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
        last_price: float,
        volume: float
    ) -> None:
        """Update market state"""
        try:
            # Calculate derived metrics
            spread = ask - bid
            mid_price = (bid + ask) / 2
            microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
            
            # Calculate order imbalance
            order_imbalance = (bid_size - ask_size) / (bid_size + ask_size)
            
            # Calculate volatility (using last 100 prices)
            prices = [
                state.last_price
                for state in self.market_states[symbol]
                if state is not None
            ]
            volatility = np.std(prices[-100:]) if len(prices) >= 100 else 0.0
            
            # Calculate VWAP
            vwap = np.average(
                prices,
                weights=[
                    state.volume
                    for state in self.market_states[symbol]
                    if state is not None
                ]
            ) if prices else last_price
            
            # Create market state
            state = MarketState(
                timestamp=datetime.now(),
                symbol=symbol,
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                last_price=last_price,
                volume=volume,
                vwap=vwap,
                order_imbalance=order_imbalance,
                volatility=volatility,
                spread=spread,
                mid_price=mid_price,
                microprice=microprice
            )
            
            # Update state history
            self.market_states[symbol].append(state)
            
        except Exception as e:
            self.logger.error(f"Error updating market state: {str(e)}")
    
    def add_signal(self, signal: Signal) -> None:
        """Add trading signal"""
        try:
            self.signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {str(e)}")
    
    def update_metrics(
        self,
        trade_qty: float,
        trade_price: float,
        trade_type: str,
        latency: float
    ) -> None:
        """Update performance metrics"""
        try:
            # Update basic metrics
            self.metrics.total_trades += 1
            self.metrics.total_volume += trade_qty
            
            # Update latency metrics
            self.latency_history.append(latency)
            self.metrics.avg_latency = np.mean(self.latency_history[-100:])
            
            # Record trade
            trade = {
                'timestamp': datetime.now(),
                'quantity': trade_qty,
                'price': trade_price,
                'type': trade_type,
                'latency': latency
            }
            self.trade_history.append(trade)
            
            # Update other metrics if enough data
            if len(self.trade_history) >= 2:
                # Calculate PnL
                pnl = self._calculate_pnl()
                self.pnl_history.append(pnl)
                self.metrics.total_pnl = pnl
                
                # Calculate Sharpe ratio
                returns = np.diff(self.pnl_history)
                self.metrics.sharpe_ratio = (
                    np.mean(returns) / np.std(returns)
                    if len(returns) > 1 else 0.0
                )
                
                # Calculate max drawdown
                cummax = np.maximum.accumulate(self.pnl_history)
                drawdown = (cummax - self.pnl_history) / cummax
                self.metrics.max_drawdown = np.max(drawdown)
                
                # Calculate win rate
                gains = [t['price'] > t_prev['price'] for t, t_prev in zip(
                    self.trade_history[1:],
                    self.trade_history[:-1]
                )]
                self.metrics.win_rate = np.mean(gains)
                
                # Calculate average holding time
                holding_times = [
                    (t['timestamp'] - t_prev['timestamp']).total_seconds() * 1000
                    for t, t_prev in zip(
                        self.trade_history[1:],
                        self.trade_history[:-1]
                    )
                ]
                self.metrics.avg_holding_time = np.mean(holding_times)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def _calculate_pnl(self) -> float:
        """Calculate current PnL"""
        try:
            pnl = 0.0
            for symbol in self.symbols:
                if not self.market_states[symbol]:
                    continue
                
                current_price = self.market_states[symbol][-1].mid_price
                position = self.positions[symbol]
                pnl += position * current_price
            
            return pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating PnL: {str(e)}")
            return 0.0
    
    @abstractmethod
    def generate_signals(self) -> List[Signal]:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def execute_signals(self, signals: List[Signal]) -> None:
        """Execute trading signals"""
        pass
    
    @abstractmethod
    def handle_market_data(self, market_data: Dict) -> None:
        """Handle incoming market data"""
        pass
    
    @abstractmethod
    def handle_order_update(self, order_update: Dict) -> None:
        """Handle order updates"""
        pass 