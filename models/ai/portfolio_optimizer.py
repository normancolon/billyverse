import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DeepPortfolioOptimizer:
    def __init__(self, 
                 n_assets: int,
                 hidden_layers: List[int] = [256, 128, 64],
                 learning_rate: float = 0.001):
        self.n_assets = n_assets
        self.model = self._build_model(hidden_layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.scaler = StandardScaler()
        
    def _build_model(self, hidden_layers: List[int]) -> tf.keras.Model:
        """Builds the deep neural network for portfolio optimization"""
        inputs = tf.keras.Input(shape=(self.n_assets * 5,))  # price, volume, volatility, sharpe, correlation
        x = inputs
        
        for units in hidden_layers:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
        outputs = tf.keras.layers.Dense(self.n_assets, activation='softmax')(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Prepares feature matrix from market data"""
        features = []
        
        for asset in range(self.n_assets):
            asset_data = market_data[asset]
            
            # Price features
            features.append(asset_data['close'].pct_change().fillna(0))
            
            # Volume features
            features.append(asset_data['volume'].pct_change().fillna(0))
            
            # Volatility (20-day rolling)
            features.append(asset_data['close'].pct_change().rolling(20).std().fillna(0))
            
            # Sharpe ratio (20-day rolling)
            returns = asset_data['close'].pct_change()
            sharpe = (returns.rolling(20).mean() / returns.rolling(20).std()).fillna(0)
            features.append(sharpe)
            
            # Correlation with other assets
            corr = market_data[asset]['close'].pct_change().corr(
                market_data.mean(axis=1).pct_change()
            )
            features.append(np.full_like(sharpe, corr))
            
        feature_matrix = np.column_stack(features)
        return self.scaler.fit_transform(feature_matrix)
    
    @tf.function
    def train_step(self, 
                   features: tf.Tensor, 
                   returns: tf.Tensor,
                   risk_aversion: float = 1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """Single training step with custom portfolio optimization loss"""
        with tf.GradientTape() as tape:
            weights = self.model(features, training=True)
            
            # Portfolio returns
            portfolio_returns = tf.reduce_sum(weights * returns, axis=1)
            expected_return = tf.reduce_mean(portfolio_returns)
            
            # Portfolio risk (variance)
            portfolio_variance = tf.reduce_mean(tf.square(portfolio_returns - expected_return))
            
            # Sharpe ratio-inspired loss (negative because we want to maximize)
            loss = -(expected_return - risk_aversion * portfolio_variance)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss, weights
    
    def train(self, 
              market_data: pd.DataFrame,
              epochs: int = 100,
              batch_size: int = 32,
              risk_aversion: float = 1.0) -> Dict[str, List[float]]:
        """Trains the model on historical market data"""
        features = self.prepare_features(market_data)
        returns = market_data.pct_change().fillna(0).values
        
        n_batches = len(features) // batch_size
        history = {'loss': [], 'weights': []}
        
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_weights = []
            
            for batch in range(n_batches):
                batch_features = features[batch*batch_size:(batch+1)*batch_size]
                batch_returns = returns[batch*batch_size:(batch+1)*batch_size]
                
                loss, weights = self.train_step(
                    tf.convert_to_tensor(batch_features, dtype=tf.float32),
                    tf.convert_to_tensor(batch_returns, dtype=tf.float32),
                    risk_aversion
                )
                
                epoch_loss += loss.numpy()
                epoch_weights.append(weights.numpy())
            
            history['loss'].append(epoch_loss / n_batches)
            history['weights'].append(np.mean(epoch_weights, axis=0))
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {history['loss'][-1]:.4f}")
        
        return history
    
    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """Predicts optimal portfolio weights for given market data"""
        features = self.prepare_features(market_data)
        features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
        return self.model(features_tensor).numpy()
    
    def save_model(self, path: str):
        """Saves the model weights and scaler"""
        self.model.save_weights(path + '_weights')
        np.save(path + '_scaler', self.scaler.get_params())
        
    def load_model(self, path: str):
        """Loads the model weights and scaler"""
        self.model.load_weights(path + '_weights')
        scaler_params = np.load(path + '_scaler.npy', allow_pickle=True)
        self.scaler.set_params(**scaler_params.item()) 