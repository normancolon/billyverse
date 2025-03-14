from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import json
from dataclasses import dataclass
from web3 import Web3

logger = logging.getLogger("billieverse.compliance.monitoring")

@dataclass
class ComplianceViolation:
    """Represents a compliance violation"""
    type: str  # Type of violation (e.g., "KYC", "AML", "INSIDER_TRADING")
    severity: str  # Severity level ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    description: str  # Detailed description of the violation
    timestamp: datetime  # When the violation was detected
    transaction_hash: Optional[str]  # Related transaction hash if applicable
    wallet_address: Optional[str]  # Related wallet address if applicable
    risk_score: float  # Risk score between 0 and 1

class ComplianceMonitor:
    """AI-driven compliance monitoring system"""
    
    def __init__(
        self,
        web3: Web3,
        model_path: Optional[str] = None,
        risk_threshold: float = 0.7
    ):
        self.web3 = web3
        self.risk_threshold = risk_threshold
        
        # Initialize models
        self.transaction_model = None
        self.kyc_model = None
        self.insider_model = None
        
        if model_path:
            self._load_models(model_path)
        
        # Initialize anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        
        # Track violations
        self.violations: List[ComplianceViolation] = []
    
    def _load_models(self, model_path: str):
        """Load pre-trained compliance models"""
        try:
            # Load transaction monitoring model
            self.transaction_model = tf.keras.models.load_model(
                f"{model_path}/transaction_model"
            )
            
            # Load KYC verification model
            self.kyc_model = tf.keras.models.load_model(
                f"{model_path}/kyc_model"
            )
            
            # Load insider trading detection model
            self.insider_model = tf.keras.models.load_model(
                f"{model_path}/insider_model"
            )
            
            logger.info("Successfully loaded compliance models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _create_transaction_model(self) -> tf.keras.Model:
        """Create transaction monitoring model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_kyc_model(self) -> tf.keras.Model:
        """Create KYC verification model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Low, Medium, High risk
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_insider_model(self) -> tf.keras.Model:
        """Create insider trading detection model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(None, 30), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def check_transaction_compliance(
        self,
        transaction: Dict,
        wallet_history: List[Dict]
    ) -> Optional[ComplianceViolation]:
        """Check transaction for compliance violations"""
        try:
            # Extract features
            features = self._extract_transaction_features(
                transaction,
                wallet_history
            )
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Check for anomalies
            is_anomaly = self.anomaly_detector.predict([features])[0] == -1
            
            if is_anomaly:
                # Get risk score from transaction model
                risk_score = float(
                    self.transaction_model.predict(features_scaled)[0][0]
                )
                
                if risk_score > self.risk_threshold:
                    return ComplianceViolation(
                        type="SUSPICIOUS_TRANSACTION",
                        severity="HIGH" if risk_score > 0.9 else "MEDIUM",
                        description=(
                            f"Suspicious transaction pattern detected "
                            f"(risk score: {risk_score:.2f})"
                        ),
                        timestamp=datetime.now(),
                        transaction_hash=transaction.get('hash'),
                        wallet_address=transaction.get('from'),
                        risk_score=risk_score
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking transaction compliance: {str(e)}")
            raise
    
    async def verify_kyc_aml(
        self,
        wallet_address: str,
        kyc_data: Dict
    ) -> Optional[ComplianceViolation]:
        """Verify KYC/AML compliance"""
        try:
            # Extract KYC features
            features = self._extract_kyc_features(kyc_data)
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Get risk classification
            risk_probs = self.kyc_model.predict(features_scaled)[0]
            risk_class = np.argmax(risk_probs)
            risk_score = float(risk_probs[risk_class])
            
            if risk_class > 0:  # Medium or High risk
                severity = "CRITICAL" if risk_class == 2 else "HIGH"
                return ComplianceViolation(
                    type="KYC_AML",
                    severity=severity,
                    description=(
                        f"KYC/AML verification failed "
                        f"(risk score: {risk_score:.2f})"
                    ),
                    timestamp=datetime.now(),
                    wallet_address=wallet_address,
                    risk_score=risk_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error verifying KYC/AML: {str(e)}")
            raise
    
    async def detect_insider_trading(
        self,
        transactions: List[Dict],
        market_data: pd.DataFrame
    ) -> List[ComplianceViolation]:
        """Detect potential insider trading patterns"""
        try:
            violations = []
            
            # Group transactions by wallet
            wallet_txs = {}
            for tx in transactions:
                wallet = tx['from']
                if wallet not in wallet_txs:
                    wallet_txs[wallet] = []
                wallet_txs[wallet].append(tx)
            
            # Analyze each wallet's transactions
            for wallet, txs in wallet_txs.items():
                # Extract temporal features
                features = self._extract_insider_features(txs, market_data)
                
                if len(features) > 0:
                    # Scale features
                    features_scaled = np.array([features])
                    
                    # Get prediction
                    risk_score = float(
                        self.insider_model.predict(features_scaled)[0][0]
                    )
                    
                    if risk_score > self.risk_threshold:
                        violations.append(
                            ComplianceViolation(
                                type="INSIDER_TRADING",
                                severity="CRITICAL",
                                description=(
                                    f"Potential insider trading pattern detected "
                                    f"(risk score: {risk_score:.2f})"
                                ),
                                timestamp=datetime.now(),
                                wallet_address=wallet,
                                risk_score=risk_score
                            )
                        )
            
            return violations
            
        except Exception as e:
            logger.error(f"Error detecting insider trading: {str(e)}")
            raise
    
    def _extract_transaction_features(
        self,
        transaction: Dict,
        wallet_history: List[Dict]
    ) -> List[float]:
        """Extract features from transaction and wallet history"""
        features = []
        
        # Transaction features
        features.extend([
            float(transaction.get('value', 0)) / 1e18,  # Convert from wei
            float(transaction.get('gas', 0)),
            float(transaction.get('gasPrice', 0)) / 1e9,  # Convert to gwei
            len(transaction.get('input', '0x')) / 2 - 1  # Input data length
        ])
        
        # Wallet history features
        if wallet_history:
            # Calculate transaction patterns
            times = [tx['timestamp'] for tx in wallet_history]
            values = [float(tx['value']) / 1e18 for tx in wallet_history]
            
            features.extend([
                len(wallet_history),  # Number of transactions
                np.mean(values),  # Average transaction value
                np.std(values),  # Transaction value volatility
                np.median(np.diff(times)),  # Median time between transactions
                len(set(tx['to'] for tx in wallet_history))  # Unique recipients
            ])
        else:
            features.extend([0, 0, 0, 0, 0])  # No history
        
        return features
    
    def _extract_kyc_features(
        self,
        kyc_data: Dict
    ) -> List[float]:
        """Extract features from KYC data"""
        features = []
        
        # Identity verification features
        features.extend([
            kyc_data.get('id_score', 0),  # ID document score
            kyc_data.get('face_match_score', 0),  # Facial recognition score
            kyc_data.get('address_match_score', 0),  # Address verification score
            int(kyc_data.get('politically_exposed', False)),
            int(kyc_data.get('sanctions_match', False))
        ])
        
        # Transaction history features
        history = kyc_data.get('transaction_history', {})
        features.extend([
            history.get('total_volume', 0),
            history.get('avg_transaction_size', 0),
            history.get('unique_counterparties', 0),
            history.get('cross_border_ratio', 0),
            history.get('high_risk_country_ratio', 0)
        ])
        
        return features
    
    def _extract_insider_features(
        self,
        transactions: List[Dict],
        market_data: pd.DataFrame
    ) -> List[float]:
        """Extract features for insider trading detection"""
        features = []
        
        if not transactions:
            return features
        
        # Sort transactions by timestamp
        transactions.sort(key=lambda x: x['timestamp'])
        
        # Calculate transaction timing features
        for i in range(len(transactions)-1):
            tx1, tx2 = transactions[i:i+2]
            
            # Get market data around transaction
            tx_time = pd.Timestamp(tx1['timestamp'], unit='s')
            market_window = market_data[
                (market_data.index >= tx_time - pd.Timedelta(hours=24)) &
                (market_data.index <= tx_time + pd.Timedelta(hours=24))
            ]
            
            if not market_window.empty:
                # Calculate market features
                price_change = (
                    market_window['price'].pct_change()
                    .fillna(0)
                    .values
                )
                volume_change = (
                    market_window['volume'].pct_change()
                    .fillna(0)
                    .values
                )
                
                features.extend([
                    np.mean(price_change),
                    np.std(price_change),
                    np.mean(volume_change),
                    np.std(volume_change),
                    float(tx2['timestamp'] - tx1['timestamp']) / 3600,  # Hours
                    float(tx2['value']) / float(tx1['value'])
                    if float(tx1['value']) > 0 else 0
                ])
        
        return features
    
    def add_violation(
        self,
        violation: ComplianceViolation
    ):
        """Add compliance violation to tracking"""
        self.violations.append(violation)
        
        # Log violation
        logger.warning(
            f"Compliance violation detected: {violation.type} "
            f"(Severity: {violation.severity}, "
            f"Risk Score: {violation.risk_score:.2f})"
        )
    
    def get_violations(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_severity: str = "LOW"
    ) -> List[ComplianceViolation]:
        """Get compliance violations within time range"""
        severity_levels = {
            "LOW": 0,
            "MEDIUM": 1,
            "HIGH": 2,
            "CRITICAL": 3
        }
        min_level = severity_levels[min_severity]
        
        filtered = []
        for v in self.violations:
            if (
                severity_levels[v.severity] >= min_level and
                (not start_time or v.timestamp >= start_time) and
                (not end_time or v.timestamp <= end_time)
            ):
                filtered.append(v)
        
        return filtered 