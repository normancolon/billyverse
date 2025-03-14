import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("billieverse.models.execution")

@dataclass
class OrderState:
    """Container for order state information"""
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    filled_quantity: float
    remaining_quantity: float
    price: float
    type: str  # 'market', 'limit', etc.
    status: str  # 'pending', 'partial', 'filled', 'cancelled'
    timestamp: datetime
    execution_details: List[Dict]  # List of execution details (price, quantity, timestamp)
    transaction_costs: float
    metadata: Dict  # Additional order metadata

@dataclass
class ExecutionMetrics:
    """Container for execution performance metrics"""
    implementation_shortfall: float  # Difference between arrival price and execution price
    market_impact: float  # Price impact of the trade
    timing_cost: float  # Cost due to price movements
    transaction_costs: float  # Total transaction costs
    execution_time: float  # Time taken to execute
    fill_rate: float  # Rate of order filling
    slippage: float  # Difference between expected and actual execution price

class BaseOrderExecutor(ABC):
    """Base class for order execution"""
    
    def __init__(
        self,
        max_orders: int = 1000,
        transaction_cost_model: Optional[Dict] = None,
        risk_limits: Optional[Dict] = None
    ):
        self.max_orders = max_orders
        self.transaction_cost_model = transaction_cost_model or {
            'fixed': 0.0,  # Fixed cost per trade
            'variable': 0.001  # Variable cost as percentage
        }
        self.risk_limits = risk_limits or {
            'max_order_size': float('inf'),
            'max_position': float('inf'),
            'max_drawdown': float('inf')
        }
        
        # Initialize storage
        self.orders: List[OrderState] = []
        self.current_positions: Dict[str, float] = {}
        self.execution_metrics: Dict[str, ExecutionMetrics] = {}
        
        self.logger = logger
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> bool:
        """Validate order parameters against risk limits"""
        try:
            # Check basic parameters
            if quantity <= 0:
                self.logger.error("Order quantity must be positive")
                return False
            
            if side not in ['buy', 'sell']:
                self.logger.error("Invalid order side")
                return False
            
            # Check against risk limits
            if quantity > self.risk_limits['max_order_size']:
                self.logger.error("Order size exceeds maximum limit")
                return False
            
            # Calculate potential position
            current_pos = self.current_positions.get(symbol, 0)
            potential_pos = current_pos + quantity if side == 'buy' else current_pos - quantity
            
            if abs(potential_pos) > self.risk_limits['max_position']:
                self.logger.error("Position would exceed maximum limit")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating order: {str(e)}")
            return False
    
    def calculate_transaction_costs(
        self,
        quantity: float,
        price: float
    ) -> float:
        """Calculate transaction costs for an order"""
        try:
            fixed_cost = self.transaction_cost_model['fixed']
            variable_cost = self.transaction_cost_model['variable'] * quantity * price
            return fixed_cost + variable_cost
            
        except Exception as e:
            self.logger.error(f"Error calculating transaction costs: {str(e)}")
            return 0.0
    
    def update_metrics(
        self,
        order: OrderState,
        market_price: float
    ) -> None:
        """Update execution metrics for an order"""
        try:
            if not order.execution_details:
                return
            
            # Calculate metrics
            arrival_price = order.execution_details[0]['price']
            vwap = sum(ex['price'] * ex['quantity'] for ex in order.execution_details) / order.filled_quantity
            
            metrics = ExecutionMetrics(
                implementation_shortfall=vwap - arrival_price,
                market_impact=market_price - arrival_price if order.side == 'buy' else arrival_price - market_price,
                timing_cost=vwap - market_price,
                transaction_costs=order.transaction_costs,
                execution_time=(order.execution_details[-1]['timestamp'] - order.timestamp).total_seconds(),
                fill_rate=order.filled_quantity / order.quantity,
                slippage=abs(vwap - order.price) / order.price if order.price else 0.0
            )
            
            self.execution_metrics[order.order_id] = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {str(e)}")
    
    def update_positions(
        self,
        symbol: str,
        quantity: float,
        side: str
    ) -> None:
        """Update current positions"""
        try:
            current_pos = self.current_positions.get(symbol, 0)
            delta = quantity if side == 'buy' else -quantity
            self.current_positions[symbol] = current_pos + delta
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {str(e)}")
    
    @abstractmethod
    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        **kwargs
    ) -> Optional[OrderState]:
        """Execute an order"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        pass
    
    @abstractmethod
    def get_market_state(self, symbol: str) -> Dict:
        """Get current market state for a symbol"""
        pass 