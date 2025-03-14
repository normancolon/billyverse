import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Type, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models.execution.base import BaseOrderExecutor, OrderState, ExecutionMetrics

logger = logging.getLogger("billieverse.models.execution.backtester")

@dataclass
class BacktestResults:
    """Container for backtest results"""
    total_orders: int
    total_volume: float
    total_cost: float
    avg_implementation_shortfall: float
    avg_market_impact: float
    avg_timing_cost: float
    avg_transaction_costs: float
    avg_execution_time: float
    avg_fill_rate: float
    avg_slippage: float
    execution_timeline: pd.DataFrame
    venue_performance: pd.DataFrame
    order_metrics: pd.DataFrame

class ExecutionBacktester:
    """Backtester for order execution strategies"""
    
    def __init__(
        self,
        market_data: pd.DataFrame,
        executor_class: Type[BaseOrderExecutor],
        executor_params: Dict,
        simulation_params: Dict = None
    ):
        self.market_data = market_data
        self.executor_class = executor_class
        self.executor_params = executor_params
        
        self.simulation_params = simulation_params or {
            'latency': 50,  # ms
            'spread': 0.001,  # 10 bps
            'market_impact': 0.1,  # Price impact factor
            'fill_rate': 0.95,  # 95% fill rate
            'transaction_cost': 0.001  # 10 bps
        }
        
        self.logger = logger
        self.results = None
    
    def _simulate_market_impact(
        self,
        price: float,
        quantity: float,
        total_volume: float
    ) -> float:
        """Simulate market impact of an order"""
        try:
            impact_factor = self.simulation_params['market_impact']
            relative_size = quantity / total_volume
            return price * (1 + impact_factor * relative_size)
            
        except Exception as e:
            self.logger.error(f"Error simulating market impact: {str(e)}")
            return price
    
    def _simulate_execution(
        self,
        order: OrderState,
        market_state: Dict
    ) -> Tuple[float, float]:
        """Simulate order execution"""
        try:
            # Apply spread
            base_price = market_state['price']
            spread = self.simulation_params['spread'] * base_price
            
            if order.side == 'buy':
                execution_price = base_price + spread / 2
            else:
                execution_price = base_price - spread / 2
            
            # Apply market impact
            impacted_price = self._simulate_market_impact(
                execution_price,
                order.quantity,
                market_state['volume']
            )
            
            # Apply fill rate
            filled_quantity = order.quantity * self.simulation_params['fill_rate']
            
            return impacted_price, filled_quantity
            
        except Exception as e:
            self.logger.error(f"Error simulating execution: {str(e)}")
            return market_state['price'], 0.0
    
    def _generate_orders(self) -> List[Dict]:
        """Generate test orders"""
        try:
            orders = []
            timestamps = self.market_data.index
            
            for i in range(0, len(timestamps), 10):  # Generate order every 10 periods
                price = self.market_data.iloc[i]['price']
                volume = self.market_data.iloc[i]['volume']
                
                # Generate random order
                side = 'buy' if np.random.random() > 0.5 else 'sell'
                quantity = np.random.uniform(0.1, 0.3) * volume
                
                orders.append({
                    'timestamp': timestamps[i],
                    'symbol': 'BTC-USD',
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'type': 'market'
                })
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Error generating orders: {str(e)}")
            return []
    
    def _get_market_state(self, timestamp: datetime) -> Dict:
        """Get market state at timestamp"""
        try:
            state = self.market_data.loc[timestamp]
            return {
                'price': state['price'],
                'volume': state['volume'],
                'bid': state['price'] * (1 - self.simulation_params['spread'] / 2),
                'ask': state['price'] * (1 + self.simulation_params['spread'] / 2),
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {}
    
    def run_backtest(self) -> BacktestResults:
        """Run backtest simulation"""
        try:
            # Initialize executor
            executor = self.executor_class(**self.executor_params)
            
            # Generate test orders
            orders = self._generate_orders()
            
            # Initialize metrics storage
            execution_timeline = []
            order_metrics = []
            venue_metrics = {}
            
            total_volume = 0.0
            total_cost = 0.0
            
            # Execute orders
            for order_data in orders:
                timestamp = order_data['timestamp']
                
                # Get market state
                market_state = self._get_market_state(timestamp)
                if not market_state:
                    continue
                
                # Execute order
                order = executor.execute_order(
                    symbol=order_data['symbol'],
                    side=order_data['side'],
                    quantity=order_data['quantity'],
                    order_type=order_data['type'],
                    price=order_data['price']
                )
                
                if not order:
                    continue
                
                # Record execution details
                for detail in order.execution_details:
                    execution_timeline.append({
                        'timestamp': detail['timestamp'],
                        'venue_id': detail['venue_id'],
                        'price': detail['price'],
                        'quantity': detail['quantity'],
                        'side': order.side
                    })
                    
                    # Update venue metrics
                    if detail['venue_id'] not in venue_metrics:
                        venue_metrics[detail['venue_id']] = {
                            'total_volume': 0.0,
                            'total_cost': 0.0,
                            'num_orders': 0,
                            'total_slippage': 0.0
                        }
                    
                    venue = venue_metrics[detail['venue_id']]
                    venue['total_volume'] += detail['quantity']
                    venue['total_cost'] += detail['quantity'] * detail['price']
                    venue['num_orders'] += 1
                    venue['total_slippage'] += abs(detail['price'] - order_data['price'])
                
                # Update totals
                total_volume += order.filled_quantity
                total_cost += sum(
                    detail['price'] * detail['quantity']
                    for detail in order.execution_details
                )
                
                # Record order metrics
                metrics = executor.execution_metrics.get(order.order_id)
                if metrics:
                    order_metrics.append({
                        'order_id': order.order_id,
                        'timestamp': order.timestamp,
                        'implementation_shortfall': metrics.implementation_shortfall,
                        'market_impact': metrics.market_impact,
                        'timing_cost': metrics.timing_cost,
                        'transaction_costs': metrics.transaction_costs,
                        'execution_time': metrics.execution_time,
                        'fill_rate': metrics.fill_rate,
                        'slippage': metrics.slippage
                    })
            
            # Create DataFrames
            timeline_df = pd.DataFrame(execution_timeline)
            order_metrics_df = pd.DataFrame(order_metrics)
            
            venue_performance = []
            for venue_id, metrics in venue_metrics.items():
                venue_performance.append({
                    'venue_id': venue_id,
                    'total_volume': metrics['total_volume'],
                    'avg_cost': metrics['total_cost'] / metrics['total_volume'],
                    'num_orders': metrics['num_orders'],
                    'avg_slippage': metrics['total_slippage'] / metrics['num_orders']
                })
            venue_performance_df = pd.DataFrame(venue_performance)
            
            # Calculate aggregate metrics
            self.results = BacktestResults(
                total_orders=len(orders),
                total_volume=total_volume,
                total_cost=total_cost,
                avg_implementation_shortfall=order_metrics_df['implementation_shortfall'].mean(),
                avg_market_impact=order_metrics_df['market_impact'].mean(),
                avg_timing_cost=order_metrics_df['timing_cost'].mean(),
                avg_transaction_costs=order_metrics_df['transaction_costs'].mean(),
                avg_execution_time=order_metrics_df['execution_time'].mean(),
                avg_fill_rate=order_metrics_df['fill_rate'].mean(),
                avg_slippage=order_metrics_df['slippage'].mean(),
                execution_timeline=timeline_df,
                venue_performance=venue_performance_df,
                order_metrics=order_metrics_df
            )
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def plot_results(self, save_dir: Path) -> None:
        """Plot backtest results"""
        try:
            if not self.results:
                raise ValueError("No backtest results available")
            
            plots_dir = save_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn')
            
            # Plot execution timeline
            plt.figure(figsize=(12, 6))
            timeline = self.results.execution_timeline
            plt.scatter(
                timeline['timestamp'],
                timeline['price'],
                c=timeline['side'].map({'buy': 'green', 'sell': 'red'}),
                alpha=0.6
            )
            plt.title("Execution Timeline")
            plt.xlabel("Time")
            plt.ylabel("Price")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "execution_timeline.png")
            plt.close()
            
            # Plot venue performance
            plt.figure(figsize=(10, 6))
            venue_perf = self.results.venue_performance
            sns.barplot(data=venue_perf, x='venue_id', y='total_volume')
            plt.title("Volume by Venue")
            plt.xlabel("Venue")
            plt.ylabel("Total Volume")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "venue_volume.png")
            plt.close()
            
            # Plot metrics distribution
            metrics = self.results.order_metrics
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            sns.histplot(metrics['implementation_shortfall'], ax=axes[0, 0])
            axes[0, 0].set_title("Implementation Shortfall Distribution")
            
            sns.histplot(metrics['market_impact'], ax=axes[0, 1])
            axes[0, 1].set_title("Market Impact Distribution")
            
            sns.histplot(metrics['fill_rate'], ax=axes[1, 0])
            axes[1, 0].set_title("Fill Rate Distribution")
            
            sns.histplot(metrics['slippage'], ax=axes[1, 1])
            axes[1, 1].set_title("Slippage Distribution")
            
            plt.tight_layout()
            plt.savefig(plots_dir / "metrics_distribution.png")
            plt.close()
            
            # Save summary statistics
            with open(save_dir / "backtest_summary.txt", "w") as f:
                f.write("Backtest Results Summary\n")
                f.write("=" * 30 + "\n\n")
                
                f.write(f"Total Orders: {self.results.total_orders}\n")
                f.write(f"Total Volume: {self.results.total_volume:.2f}\n")
                f.write(f"Total Cost: {self.results.total_cost:.2f}\n\n")
                
                f.write("Average Metrics:\n")
                f.write(f"- Implementation Shortfall: {self.results.avg_implementation_shortfall:.4f}\n")
                f.write(f"- Market Impact: {self.results.avg_market_impact:.4f}\n")
                f.write(f"- Timing Cost: {self.results.avg_timing_cost:.4f}\n")
                f.write(f"- Transaction Costs: {self.results.avg_transaction_costs:.4f}\n")
                f.write(f"- Execution Time: {self.results.avg_execution_time:.2f}s\n")
                f.write(f"- Fill Rate: {self.results.avg_fill_rate:.2%}\n")
                f.write(f"- Slippage: {self.results.avg_slippage:.4f}\n")
            
        except Exception as e:
            self.logger.error(f"Error plotting results: {str(e)}")
            raise 