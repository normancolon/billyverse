import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import gym
from collections import deque
import random

from models.execution.base import BaseOrderExecutor, OrderState

logger = logging.getLogger("billieverse.models.execution.rl")

class MarketEnvironment(gym.Env):
    """Custom gym environment for order execution"""
    
    def __init__(
        self,
        initial_state: Dict,
        time_limit: int = 100,
        transaction_cost_model: Dict = None
    ):
        super().__init__()
        
        # Action space: percentage of remaining quantity to execute
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # State space: market features
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),  # [price, volume, volatility, spread, etc.]
            dtype=np.float32
        )
        
        self.initial_state = initial_state
        self.time_limit = time_limit
        self.transaction_cost_model = transaction_cost_model or {
            'fixed': 0.0,
            'variable': 0.001
        }
        
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        self.current_step = 0
        self.remaining_quantity = self.initial_state['quantity']
        self.initial_price = self.initial_state['price']
        self.current_price = self.initial_price
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Calculate execution quantity
        exec_quantity = min(
            self.remaining_quantity,
            action * self.remaining_quantity
        )
        
        # Simulate price impact
        price_impact = 0.1 * (exec_quantity / self.initial_state['quantity'])
        execution_price = self.current_price * (1 + price_impact)
        
        # Calculate costs
        transaction_cost = (
            self.transaction_cost_model['fixed'] +
            self.transaction_cost_model['variable'] * exec_quantity * execution_price
        )
        
        # Update state
        self.remaining_quantity -= exec_quantity
        self.current_price = execution_price
        
        # Calculate reward
        implementation_shortfall = (execution_price - self.initial_price) * exec_quantity
        reward = -(implementation_shortfall + transaction_cost)
        
        # Check if done
        self.done = (
            self.current_step >= self.time_limit or
            self.remaining_quantity <= 0
        )
        
        return self._get_observation(), reward, self.done, {}
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        return np.array([
            self.current_price / self.initial_price,  # Normalized price
            self.remaining_quantity / self.initial_state['quantity'],  # Remaining quantity
            self.current_step / self.time_limit,  # Time progress
            # Add more market features here
        ], dtype=np.float32)

class DQN(nn.Module):
    """Deep Q-Network for order execution"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RLOrderExecutor(BaseOrderExecutor):
    """Reinforcement learning-based order executor"""
    
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 10,  # Discretized action space
        hidden_dim: int = 64,
        memory_size: int = 10000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        learning_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize networks
        self.policy_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net = DQN(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        # Initialize environment
        self.env = None
    
    def _update_epsilon(self):
        """Update exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _get_action(self, state: np.ndarray) -> float:
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.random()  # Random action between 0 and 1
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.argmax().item()
            return action_idx / (self.action_dim - 1)  # Convert to [0, 1]
    
    def _optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor([x[0] for x in batch])
        action_batch = torch.FloatTensor([x[1] for x in batch])
        reward_batch = torch.FloatTensor([x[2] for x in batch])
        next_state_batch = torch.FloatTensor([x[3] for x in batch])
        done_batch = torch.FloatTensor([x[4] for x in batch])
        
        # Compute Q(s_t, a)
        current_q = self.policy_net(state_batch).gather(1, action_batch.long().unsqueeze(1))
        
        # Compute V(s_{t+1})
        next_q = self.target_net(next_state_batch).max(1)[0].detach()
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        **kwargs
    ) -> Optional[OrderState]:
        """Execute order using RL agent"""
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
                metadata={}
            )
            
            # Initialize environment
            market_state = self.get_market_state(symbol)
            self.env = MarketEnvironment(
                initial_state={
                    'quantity': quantity,
                    'price': price or market_state['price']
                },
                transaction_cost_model=self.transaction_cost_model
            )
            
            # Execute order using RL agent
            state = self.env.reset()
            total_reward = 0
            
            while not self.env.done:
                # Get action from policy
                action = self._get_action(state)
                
                # Execute action
                next_state, reward, done, _ = self.env.step(action)
                
                # Store transition
                self.memory.append((state, action, reward, next_state, done))
                
                # Update state and metrics
                state = next_state
                total_reward += reward
                
                # Optimize model
                self._optimize_model()
                
                # Update exploration rate
                self._update_epsilon()
                
                # Update order state
                executed_quantity = quantity - self.env.remaining_quantity
                order.filled_quantity = executed_quantity
                order.remaining_quantity = self.env.remaining_quantity
                order.status = 'filled' if done else 'partial'
                
                # Add execution details
                order.execution_details.append({
                    'price': self.env.current_price,
                    'quantity': executed_quantity,
                    'timestamp': datetime.now()
                })
            
            # Update positions and metrics
            self.update_positions(symbol, order.filled_quantity, side)
            self.update_metrics(order, market_state['price'])
            
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
                    order.status = 'cancelled'
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_market_state(self, symbol: str) -> Dict:
        """Get current market state"""
        try:
            # Implement market data fetching here
            # For now, return dummy data
            return {
                'price': 100.0,
                'volume': 1000.0,
                'bid': 99.5,
                'ask': 100.5,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market state: {str(e)}")
            return {} 