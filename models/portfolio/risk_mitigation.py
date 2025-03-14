import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow.keras import layers
import logging
from scipy.stats import norm
from gym import spaces
import py_vollib.black_scholes as bs
import py_vollib.black_scholes.greeks.analytical as greeks

from models.portfolio.risk import RiskManager, RiskMetrics

logger = logging.getLogger("billieverse.models.portfolio.risk_mitigation")

@dataclass
class StopLossState:
    """State representation for RL stop-loss model"""
    price: float
    volatility: float
    trend_strength: float
    drawdown: float
    profit_loss: float
    time_in_trade: int
    market_regime: int  # 0: normal, 1: high vol, 2: crisis
    risk_score: float

@dataclass
class DerivativePosition:
    """Container for derivative position details"""
    instrument_type: str  # 'option' or 'future'
    underlying: str
    strike: float
    expiry: pd.Timestamp
    position: float
    delta: float
    gamma: float
    vega: float
    theta: float
    implied_vol: float

class StopLossRL:
    """Reinforcement learning model for dynamic stop-loss"""
    
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 3,  # 0: hold, 1: tighten stop, 2: exit
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = 10000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-network
        self.q_network = self._build_network(state_dim, action_dim)
        self.target_network = self._build_network(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # Initialize experience replay buffer
        self.replay_buffer = []
        self.buffer_size = buffer_size
        
        # Initialize tracking
        self.training_history = []
    
    def _build_network(
        self,
        state_dim: int,
        action_dim: int
    ) -> tf.keras.Model:
        """Build Q-network architecture"""
        model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_dim,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(action_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def get_action(
        self,
        state: StopLossState,
        training: bool = False
    ) -> int:
        """Get action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_array = np.array([
            state.price, state.volatility, state.trend_strength,
            state.drawdown, state.profit_loss, state.time_in_trade,
            state.market_regime, state.risk_score
        ]).reshape(1, -1)
        
        q_values = self.q_network.predict(state_array, verbose=0)
        return np.argmax(q_values[0])
    
    def train(
        self,
        batch_size: int = 32
    ) -> float:
        """Train Q-network using experience replay"""
        if len(self.replay_buffer) < batch_size:
            return 0.0
        
        # Sample batch
        batch = np.random.choice(
            len(self.replay_buffer),
            batch_size,
            replace=False
        )
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in batch:
            s, a, r, ns, d = self.replay_buffer[idx]
            states.append([
                s.price, s.volatility, s.trend_strength,
                s.drawdown, s.profit_loss, s.time_in_trade,
                s.market_regime, s.risk_score
            ])
            actions.append(a)
            rewards.append(r)
            next_states.append([
                ns.price, ns.volatility, ns.trend_strength,
                ns.drawdown, ns.profit_loss, ns.time_in_trade,
                ns.market_regime, ns.risk_score
            ])
            dones.append(d)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Get target Q-values
        future_qs = self.target_network.predict(next_states, verbose=0)
        target_qs = self.q_network.predict(states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                target_qs[i][actions[i]] = rewards[i] + \
                    self.gamma * np.max(future_qs[i])
        
        # Train network
        loss = self.q_network.train_on_batch(states, target_qs)
        self.training_history.append(loss)
        
        return loss
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

class DerivativesHedging:
    """Derivatives hedging and risk management"""
    
    def __init__(
        self,
        risk_manager: RiskManager,
        vol_surface: Optional[Dict[str, pd.DataFrame]] = None,
        min_hedge_ratio: float = 0.5,
        max_hedge_ratio: float = 1.0,
        rebalance_threshold: float = 0.1
    ):
        self.risk_manager = risk_manager
        self.vol_surface = vol_surface
        self.min_hedge_ratio = min_hedge_ratio
        self.max_hedge_ratio = max_hedge_ratio
        self.rebalance_threshold = rebalance_threshold
        
        # Initialize positions
        self.option_positions: Dict[str, DerivativePosition] = {}
        self.future_positions: Dict[str, DerivativePosition] = {}
        
        # Initialize tracking
        self.hedge_history: List[Dict[str, DerivativePosition]] = []
        self.greek_history: List[Dict[str, float]] = []
    
    def calculate_option_greeks(
        self,
        underlying_price: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        option_type: str = 'c',
        r: float = 0.02
    ) -> Dict[str, float]:
        """Calculate option Greeks"""
        try:
            # Calculate Greeks using py_vollib
            delta = greeks.delta(
                option_type, underlying_price, strike,
                time_to_expiry, r, volatility
            )
            gamma = greeks.gamma(
                option_type, underlying_price, strike,
                time_to_expiry, r, volatility
            )
            vega = greeks.vega(
                option_type, underlying_price, strike,
                time_to_expiry, r, volatility
            )
            theta = greeks.theta(
                option_type, underlying_price, strike,
                time_to_expiry, r, volatility
            )
            
            return {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta
            }
            
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            return {}
    
    def select_hedge_options(
        self,
        portfolio_delta: float,
        portfolio_vega: float
    ) -> List[DerivativePosition]:
        """Select optimal options for hedging"""
        try:
            selected_options = []
            remaining_delta = -portfolio_delta  # Target opposite delta
            remaining_vega = -portfolio_vega  # Target opposite vega
            
            if self.vol_surface is None:
                return []
            
            for underlying, surface in self.vol_surface.items():
                for _, row in surface.iterrows():
                    # Calculate Greeks
                    greeks = self.calculate_option_greeks(
                        row['underlying_price'],
                        row['strike'],
                        row['time_to_expiry'],
                        row['implied_vol']
                    )
                    
                    if not greeks:
                        continue
                    
                    # Check if option helps hedge exposure
                    if (abs(remaining_delta) > 0.1 and
                        abs(remaining_vega) > 0.1):
                        # Calculate position size
                        size = min(
                            abs(remaining_delta / greeks['delta']),
                            abs(remaining_vega / greeks['vega'])
                        )
                        
                        if size > 0.01:  # Minimum position size
                            position = DerivativePosition(
                                instrument_type='option',
                                underlying=underlying,
                                strike=row['strike'],
                                expiry=row.name,
                                position=size,
                                delta=greeks['delta'],
                                gamma=greeks['gamma'],
                                vega=greeks['vega'],
                                theta=greeks['theta'],
                                implied_vol=row['implied_vol']
                            )
                            
                            selected_options.append(position)
                            remaining_delta -= size * greeks['delta']
                            remaining_vega -= size * greeks['vega']
                            
                            if (abs(remaining_delta) < 0.1 and
                                abs(remaining_vega) < 0.1):
                                break
            
            return selected_options
            
        except Exception as e:
            logger.error(f"Error selecting hedge options: {str(e)}")
            return []
    
    def update_hedges(
        self,
        portfolio_risk: RiskMetrics
    ) -> Dict[str, List[DerivativePosition]]:
        """Update derivative hedges"""
        try:
            # Calculate portfolio Greeks
            portfolio_delta = portfolio_risk.beta  # Approximate delta with beta
            portfolio_vega = portfolio_risk.volatility * \
                np.sqrt(252)  # Annualized vol
            
            # Select hedge options
            new_options = self.select_hedge_options(
                portfolio_delta,
                portfolio_vega
            )
            
            # Calculate hedge ratio
            total_hedge_ratio = sum(
                abs(opt.delta * opt.position)
                for opt in new_options
            ) / abs(portfolio_delta)
            
            if (total_hedge_ratio < self.min_hedge_ratio or
                total_hedge_ratio > self.max_hedge_ratio):
                # Adjust position sizes
                scale = (
                    (self.min_hedge_ratio + self.max_hedge_ratio) / 2 /
                    total_hedge_ratio
                )
                for opt in new_options:
                    opt.position *= scale
            
            # Update positions
            self.option_positions = {
                f"{opt.underlying}_{opt.strike}_{opt.expiry}": opt
                for opt in new_options
            }
            
            # Track history
            self.hedge_history.append(self.option_positions)
            self.greek_history.append({
                'portfolio_delta': portfolio_delta,
                'portfolio_vega': portfolio_vega,
                'hedge_delta': sum(
                    opt.delta * opt.position
                    for opt in new_options
                ),
                'hedge_vega': sum(
                    opt.vega * opt.position
                    for opt in new_options
                )
            })
            
            return {
                'options': list(self.option_positions.values()),
                'futures': list(self.future_positions.values())
            }
            
        except Exception as e:
            logger.error(f"Error updating hedges: {str(e)}")
            return {'options': [], 'futures': []}

class AIPositionSizing:
    """AI-driven position sizing"""
    
    def __init__(
        self,
        risk_manager: RiskManager,
        stop_loss_model: StopLossRL,
        max_position_size: float = 0.2,
        min_position_size: float = 0.02,
        kelly_fraction: float = 0.5,
        confidence_threshold: float = 0.6
    ):
        self.risk_manager = risk_manager
        self.stop_loss_model = stop_loss_model
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.kelly_fraction = kelly_fraction
        self.confidence_threshold = confidence_threshold
        
        # Initialize tracking
        self.position_history: List[Dict[str, float]] = []
        self.sizing_metrics: List[Dict[str, float]] = []
    
    def calculate_kelly_size(
        self,
        win_prob: float,
        win_loss_ratio: float
    ) -> float:
        """Calculate position size using Kelly Criterion"""
        try:
            kelly_size = (win_prob * win_loss_ratio - (1 - win_prob)) / \
                win_loss_ratio
            
            # Apply fraction and limits
            position_size = kelly_size * self.kelly_fraction
            position_size = np.clip(
                position_size,
                self.min_position_size,
                self.max_position_size
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating Kelly size: {str(e)}")
            return self.min_position_size
    
    def get_position_size(
        self,
        asset: str,
        signal_confidence: float,
        current_price: float,
        volatility: float,
        trend_strength: float
    ) -> float:
        """Get optimal position size using AI signals"""
        try:
            # Skip if confidence below threshold
            if signal_confidence < self.confidence_threshold:
                return 0.0
            
            # Create state for stop-loss model
            state = StopLossState(
                price=current_price,
                volatility=volatility,
                trend_strength=trend_strength,
                drawdown=0.0,
                profit_loss=0.0,
                time_in_trade=0,
                market_regime=0,  # Determine regime from volatility
                risk_score=self.risk_manager.risk_metrics_history[-1].var_95
                if self.risk_manager.risk_metrics_history
                else 0.0
            )
            
            # Get stop-loss action
            stop_loss_action = self.stop_loss_model.get_action(state)
            
            # Calculate win probability and ratio
            win_prob = signal_confidence * (1 - volatility)  # Adjust for vol
            win_loss_ratio = 1.5 + trend_strength  # Adjust for trend
            
            # Calculate base position size
            base_size = self.calculate_kelly_size(win_prob, win_loss_ratio)
            
            # Adjust size based on stop-loss signal
            if stop_loss_action == 2:  # Exit signal
                return 0.0
            elif stop_loss_action == 1:  # Tighten stop
                base_size *= 0.7
            
            # Track metrics
            self.sizing_metrics.append({
                'asset': asset,
                'confidence': signal_confidence,
                'win_prob': win_prob,
                'win_loss_ratio': win_loss_ratio,
                'base_size': base_size,
                'stop_loss_action': stop_loss_action,
                'volatility': volatility,
                'trend_strength': trend_strength
            })
            
            return base_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
    def update_positions(
        self,
        portfolio: Dict[str, float],
        signals: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Update position sizes based on signals"""
        try:
            new_positions = {}
            
            for asset, signal in signals.items():
                size = self.get_position_size(
                    asset,
                    signal['confidence'],
                    signal['price'],
                    signal['volatility'],
                    signal['trend_strength']
                )
                
                if size > 0:
                    new_positions[asset] = size
            
            # Normalize to sum to 1
            total_size = sum(new_positions.values())
            if total_size > 0:
                new_positions = {
                    asset: size / total_size
                    for asset, size in new_positions.items()
                }
            
            # Track history
            self.position_history.append(new_positions)
            
            return new_positions
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")
            return portfolio  # Return current portfolio if error 