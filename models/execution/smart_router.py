import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

from models.execution.base import BaseOrderExecutor, OrderState

logger = logging.getLogger("billieverse.models.execution.router")

@dataclass
class VenueMetrics:
    """Container for venue performance metrics"""
    venue_id: str
    avg_spread: float  # Average bid-ask spread
    avg_latency: float  # Average execution latency (ms)
    fill_rate: float  # Order fill rate
    cost_score: float  # Transaction cost score
    liquidity_score: float  # Available liquidity score
    reliability_score: float  # Venue reliability score

class SmartOrderRouter(BaseOrderExecutor):
    """Smart order router for optimal execution across venues"""
    
    def __init__(
        self,
        venues: List[Dict],
        max_venues: int = 5,
        update_interval: int = 60,  # seconds
        executor_threads: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.venues = venues
        self.max_venues = max_venues
        self.update_interval = update_interval
        
        # Initialize venue metrics
        self.venue_metrics: Dict[str, VenueMetrics] = {}
        for venue in venues:
            self.venue_metrics[venue['id']] = VenueMetrics(
                venue_id=venue['id'],
                avg_spread=0.0,
                avg_latency=0.0,
                fill_rate=1.0,
                cost_score=1.0,
                liquidity_score=1.0,
                reliability_score=1.0
            )
        
        # Initialize thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=executor_threads)
        
        # Start metrics update loop
        self._start_metrics_update()
    
    def _start_metrics_update(self):
        """Start background metrics update loop"""
        async def update_loop():
            while True:
                await self._update_venue_metrics()
                await asyncio.sleep(self.update_interval)
        
        loop = asyncio.get_event_loop()
        loop.create_task(update_loop())
    
    async def _update_venue_metrics(self):
        """Update venue performance metrics"""
        try:
            for venue_id, metrics in self.venue_metrics.items():
                # Get venue state
                venue_state = await self._get_venue_state(venue_id)
                
                # Update spread metrics
                if 'spread' in venue_state:
                    metrics.avg_spread = (
                        0.9 * metrics.avg_spread +
                        0.1 * venue_state['spread']
                    )
                
                # Update latency metrics
                if 'latency' in venue_state:
                    metrics.avg_latency = (
                        0.9 * metrics.avg_latency +
                        0.1 * venue_state['latency']
                    )
                
                # Update fill rate
                if 'fill_rate' in venue_state:
                    metrics.fill_rate = (
                        0.9 * metrics.fill_rate +
                        0.1 * venue_state['fill_rate']
                    )
                
                # Calculate scores
                metrics.cost_score = self._calculate_cost_score(metrics)
                metrics.liquidity_score = self._calculate_liquidity_score(metrics)
                metrics.reliability_score = self._calculate_reliability_score(metrics)
                
        except Exception as e:
            self.logger.error(f"Error updating venue metrics: {str(e)}")
    
    async def _get_venue_state(self, venue_id: str) -> Dict:
        """Get current state of a venue"""
        try:
            # Implement venue-specific state fetching here
            # For now, return dummy data
            return {
                'spread': 0.1,
                'latency': 50.0,  # ms
                'fill_rate': 0.95,
                'liquidity': 1000000.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting venue state: {str(e)}")
            return {}
    
    def _calculate_cost_score(self, metrics: VenueMetrics) -> float:
        """Calculate venue cost score"""
        try:
            # Lower spread and latency is better
            spread_score = 1.0 / (1.0 + metrics.avg_spread)
            latency_score = 1.0 / (1.0 + metrics.avg_latency / 1000.0)
            
            return 0.7 * spread_score + 0.3 * latency_score
            
        except Exception as e:
            self.logger.error(f"Error calculating cost score: {str(e)}")
            return 0.0
    
    def _calculate_liquidity_score(self, metrics: VenueMetrics) -> float:
        """Calculate venue liquidity score"""
        try:
            # Higher fill rate is better
            return metrics.fill_rate
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {str(e)}")
            return 0.0
    
    def _calculate_reliability_score(self, metrics: VenueMetrics) -> float:
        """Calculate venue reliability score"""
        try:
            # Consider latency and fill rate
            latency_score = 1.0 / (1.0 + metrics.avg_latency / 1000.0)
            return 0.6 * metrics.fill_rate + 0.4 * latency_score
            
        except Exception as e:
            self.logger.error(f"Error calculating reliability score: {str(e)}")
            return 0.0
    
    def _select_venues(
        self,
        quantity: float,
        side: str
    ) -> List[Tuple[str, float]]:
        """Select venues and allocate quantity"""
        try:
            # Calculate venue scores
            venue_scores = []
            for venue_id, metrics in self.venue_metrics.items():
                score = (
                    0.4 * metrics.cost_score +
                    0.3 * metrics.liquidity_score +
                    0.3 * metrics.reliability_score
                )
                venue_scores.append((venue_id, score))
            
            # Sort venues by score
            venue_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top venues
            selected_venues = venue_scores[:self.max_venues]
            
            # Allocate quantity based on scores
            total_score = sum(score for _, score in selected_venues)
            allocations = [
                (venue_id, (score / total_score) * quantity)
                for venue_id, score in selected_venues
            ]
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error selecting venues: {str(e)}")
            return []
    
    async def _execute_on_venue(
        self,
        venue_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> Optional[Dict]:
        """Execute order on a specific venue"""
        try:
            # Implement venue-specific execution here
            # For now, simulate execution
            execution_price = price or 100.0
            filled_quantity = quantity * 0.95  # 95% fill rate
            
            return {
                'venue_id': venue_id,
                'price': execution_price,
                'quantity': filled_quantity,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing on venue: {str(e)}")
            return None
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        **kwargs
    ) -> Optional[OrderState]:
        """Execute order across multiple venues"""
        try:
            if not self.validate_order(symbol, side, quantity, price):
                return None
            
            # Initialize order
            order_id = f"order_{datetime.now().timestamp()}"
            order = OrderState(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=0.0,
                remaining_quantity=quantity,
                price=price,
                type=order_type,
                status='pending',
                timestamp=datetime.now(),
                execution_details=[],
                transaction_costs=0.0,
                metadata={'venues': []}
            )
            
            # Select venues and allocate quantity
            venue_allocations = self._select_venues(quantity, side)
            
            # Execute on each venue
            async def execute_all():
                executions = []
                for venue_id, venue_quantity in venue_allocations:
                    execution = await self._execute_on_venue(
                        venue_id,
                        symbol,
                        side,
                        venue_quantity,
                        price
                    )
                    if execution:
                        executions.append(execution)
                return executions
            
            loop = asyncio.get_event_loop()
            executions = loop.run_until_complete(execute_all())
            
            # Update order state
            total_filled = 0.0
            total_cost = 0.0
            
            for execution in executions:
                filled_quantity = execution['quantity']
                execution_price = execution['price']
                
                total_filled += filled_quantity
                total_cost += filled_quantity * execution_price
                
                # Add execution details
                order.execution_details.append({
                    'venue_id': execution['venue_id'],
                    'price': execution_price,
                    'quantity': filled_quantity,
                    'timestamp': execution['timestamp']
                })
                
                # Add venue to metadata
                order.metadata['venues'].append(execution['venue_id'])
            
            # Update order status
            order.filled_quantity = total_filled
            order.remaining_quantity = quantity - total_filled
            order.status = 'filled' if order.remaining_quantity <= 0 else 'partial'
            
            # Calculate transaction costs
            order.transaction_costs = self.calculate_transaction_costs(
                total_filled,
                total_cost / total_filled if total_filled > 0 else price or 0.0
            )
            
            # Update positions and metrics
            self.update_positions(symbol, order.filled_quantity, side)
            self.update_metrics(order, price or 0.0)
            
            # Store order
            self.orders.append(order)
            if len(self.orders) > self.max_orders:
                self.orders = self.orders[-self.max_orders:]
            
            return order
            
        except Exception as e:
            self.logger.error(f"Error executing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            for order in self.orders:
                if order.order_id == order_id and order.status in ['pending', 'partial']:
                    # Cancel on all venues
                    for venue_id in order.metadata.get('venues', []):
                        # Implement venue-specific cancellation here
                        pass
                    
                    order.status = 'cancelled'
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_market_state(self, symbol: str) -> Dict:
        """Get aggregated market state across venues"""
        try:
            # Aggregate state from all venues
            states = []
            for venue in self.venues:
                venue_state = asyncio.run(self._get_venue_state(venue['id']))
                if venue_state:
                    states.append(venue_state)
            
            if not states:
                return {}
            
            # Calculate aggregated metrics
            return {
                'price': np.mean([s.get('price', 0.0) for s in states]),
                'volume': sum(s.get('volume', 0.0) for s in states),
                'bid': max(s.get('bid', 0.0) for s in states),
                'ask': min(s.get('ask', float('inf')) for s in states),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {} 