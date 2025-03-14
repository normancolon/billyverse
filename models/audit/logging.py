from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
import shap
import lime
import lime.lime_tabular
from web3 import Web3
import json
import logging
from dataclasses import dataclass, asdict
import hashlib
from eth_account.messages import encode_defunct

logger = logging.getLogger("billieverse.audit.logging")

@dataclass
class ModelDecision:
    """Represents an AI model's decision with explanations"""
    timestamp: datetime
    model_id: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    feature_importance: Dict[str, float]
    shap_values: Optional[List[float]]
    lime_explanation: Optional[Dict[str, float]]
    metadata: Dict[str, Any]

@dataclass
class AuditLog:
    """Represents an immutable audit log entry"""
    timestamp: datetime
    log_type: str  # "MODEL_DECISION", "TRADE_EXECUTION", "COMPLIANCE_CHECK"
    content: Dict[str, Any]
    hash: str  # Hash of previous log + current content
    signature: Optional[str]  # Signed by private key
    transaction_hash: Optional[str]  # Blockchain transaction hash

class AIAuditLogger:
    """AI-powered audit and logging system"""
    
    def __init__(
        self,
        web3: Web3,
        contract_address: str,
        private_key: str,
        storage_path: str
    ):
        self.web3 = web3
        self.contract_address = Web3.to_checksum_address(contract_address)
        self.private_key = private_key
        self.storage_path = storage_path
        
        # Initialize audit chain
        self.audit_logs: List[AuditLog] = []
        self._load_existing_logs()
        
        # Initialize explainers
        self.shap_explainers = {}
        self.lime_explainers = {}
    
    def _load_existing_logs(self):
        """Load existing audit logs from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                logs_data = json.load(f)
                
            for log_data in logs_data:
                log_data['timestamp'] = datetime.fromisoformat(
                    log_data['timestamp']
                )
                self.audit_logs.append(AuditLog(**log_data))
                
            logger.info(f"Loaded {len(self.audit_logs)} existing audit logs")
            
        except FileNotFoundError:
            logger.info("No existing audit logs found")
        except Exception as e:
            logger.error(f"Error loading audit logs: {str(e)}")
    
    def _save_logs(self):
        """Save audit logs to storage"""
        try:
            logs_data = []
            for log in self.audit_logs:
                log_dict = asdict(log)
                log_dict['timestamp'] = log.timestamp.isoformat()
                logs_data.append(log_dict)
                
            with open(self.storage_path, 'w') as f:
                json.dump(logs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving audit logs: {str(e)}")
    
    def _calculate_hash(
        self,
        content: Dict[str, Any],
        previous_hash: Optional[str] = None
    ) -> str:
        """Calculate hash of log content"""
        hash_input = json.dumps(content, sort_keys=True)
        if previous_hash:
            hash_input = previous_hash + hash_input
            
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _sign_hash(self, hash_str: str) -> str:
        """Sign hash with private key"""
        message = encode_defunct(text=hash_str)
        signed = self.web3.eth.account.sign_message(
            message,
            private_key=self.private_key
        )
        return signed.signature.hex()
    
    async def _store_on_blockchain(
        self,
        log_hash: str,
        signature: str
    ) -> str:
        """Store log hash and signature on blockchain"""
        try:
            # Create transaction
            nonce = await self.web3.eth.get_transaction_count(
                self.web3.eth.account.from_key(self.private_key).address
            )
            
            tx = {
                'nonce': nonce,
                'to': self.contract_address,
                'value': 0,
                'gas': 100000,
                'gasPrice': await self.web3.eth.gas_price,
                'data': self.web3.keccak(
                    text=f"{log_hash}{signature}"
                ).hex()
            }
            
            # Sign and send transaction
            signed_tx = self.web3.eth.account.sign_transaction(
                tx,
                self.private_key
            )
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(
                tx_hash
            )
            
            return receipt['transactionHash'].hex()
            
        except Exception as e:
            logger.error(f"Error storing on blockchain: {str(e)}")
            return None
    
    def initialize_explainers(
        self,
        model: tf.keras.Model,
        model_id: str,
        background_data: np.ndarray,
        feature_names: List[str]
    ):
        """Initialize SHAP and LIME explainers for a model"""
        try:
            # Initialize SHAP explainer
            self.shap_explainers[model_id] = shap.KernelExplainer(
                model.predict,
                background_data
            )
            
            # Initialize LIME explainer
            self.lime_explainers[model_id] = lime.lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=feature_names,
                mode='regression'
            )
            
            logger.info(f"Initialized explainers for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
    
    def explain_decision(
        self,
        model: tf.keras.Model,
        model_id: str,
        input_data: np.ndarray,
        feature_names: List[str]
    ) -> ModelDecision:
        """Generate explanations for model decision"""
        try:
            # Get model prediction
            prediction = model.predict(input_data)[0]
            
            # Calculate feature importance
            importance = {}
            if hasattr(model, 'layers') and len(model.layers) > 0:
                weights = model.layers[-1].get_weights()[0]
                for i, name in enumerate(feature_names):
                    importance[name] = float(abs(weights[i][0]))
            
            # Generate SHAP explanation
            shap_values = None
            if model_id in self.shap_explainers:
                shap_values = self.shap_explainers[model_id].shap_values(
                    input_data
                )[0]
            
            # Generate LIME explanation
            lime_explanation = None
            if model_id in self.lime_explainers:
                exp = self.lime_explainers[model_id].explain_instance(
                    input_data[0],
                    model.predict
                )
                lime_explanation = dict(exp.as_list())
            
            return ModelDecision(
                timestamp=datetime.now(),
                model_id=model_id,
                model_version=getattr(model, 'version', 'unknown'),
                input_features=dict(zip(feature_names, input_data[0].tolist())),
                prediction=float(prediction),
                confidence=float(max(prediction, 1 - prediction)),
                feature_importance=importance,
                shap_values=shap_values.tolist() if shap_values is not None else None,
                lime_explanation=lime_explanation,
                metadata={}
            )
            
        except Exception as e:
            logger.error(f"Error explaining decision: {str(e)}")
            raise
    
    async def log_decision(
        self,
        decision: ModelDecision,
        store_on_chain: bool = True
    ):
        """Log model decision with blockchain storage"""
        try:
            # Prepare log content
            content = asdict(decision)
            content['timestamp'] = decision.timestamp.isoformat()
            
            # Calculate hash
            previous_hash = (
                self.audit_logs[-1].hash if self.audit_logs else None
            )
            log_hash = self._calculate_hash(content, previous_hash)
            
            # Sign hash
            signature = self._sign_hash(log_hash)
            
            # Store on blockchain if requested
            tx_hash = None
            if store_on_chain:
                tx_hash = await self._store_on_blockchain(log_hash, signature)
            
            # Create audit log
            audit_log = AuditLog(
                timestamp=decision.timestamp,
                log_type="MODEL_DECISION",
                content=content,
                hash=log_hash,
                signature=signature,
                transaction_hash=tx_hash
            )
            
            # Add to logs and save
            self.audit_logs.append(audit_log)
            self._save_logs()
            
            logger.info(
                f"Logged model decision with hash {log_hash} "
                f"and transaction {tx_hash}"
            )
            
        except Exception as e:
            logger.error(f"Error logging decision: {str(e)}")
            raise
    
    def verify_log_integrity(
        self,
        start_index: Optional[int] = None,
        end_index: Optional[int] = None
    ) -> bool:
        """Verify integrity of audit log chain"""
        try:
            logs_to_verify = self.audit_logs[start_index:end_index]
            previous_hash = None
            
            for log in logs_to_verify:
                # Verify hash
                calculated_hash = self._calculate_hash(
                    log.content,
                    previous_hash
                )
                if calculated_hash != log.hash:
                    logger.error(f"Hash mismatch for log at {log.timestamp}")
                    return False
                
                # Verify signature
                if log.signature:
                    signer = self.web3.eth.account.recover_message(
                        encode_defunct(text=log.hash),
                        signature=log.signature
                    )
                    expected_signer = self.web3.eth.account.from_key(
                        self.private_key
                    ).address
                    
                    if signer != expected_signer:
                        logger.error(
                            f"Signature verification failed for log at "
                            f"{log.timestamp}"
                        )
                        return False
                
                previous_hash = log.hash
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying log integrity: {str(e)}")
            return False
    
    def get_model_decisions(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ModelDecision]:
        """Get model decisions within time range"""
        decisions = []
        
        for log in self.audit_logs:
            if log.log_type != "MODEL_DECISION":
                continue
                
            timestamp = datetime.fromisoformat(log.content['timestamp'])
            
            if (
                (not model_id or log.content['model_id'] == model_id) and
                (not start_time or timestamp >= start_time) and
                (not end_time or timestamp <= end_time)
            ):
                decisions.append(ModelDecision(**log.content))
        
        return decisions
    
    def analyze_model_performance(
        self,
        model_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze model performance from audit logs"""
        decisions = self.get_model_decisions(model_id, start_time, end_time)
        
        if not decisions:
            return {}
        
        # Calculate metrics
        predictions = [d.prediction for d in decisions]
        confidences = [d.confidence for d in decisions]
        
        # Aggregate feature importance
        importance_sums = {}
        importance_counts = {}
        
        for decision in decisions:
            for feature, importance in decision.feature_importance.items():
                importance_sums[feature] = (
                    importance_sums.get(feature, 0) + importance
                )
                importance_counts[feature] = (
                    importance_counts.get(feature, 0) + 1
                )
        
        avg_importance = {
            feature: importance_sums[feature] / importance_counts[feature]
            for feature in importance_sums
        }
        
        return {
            'total_decisions': len(decisions),
            'avg_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'avg_prediction': float(np.mean(predictions)),
            'std_prediction': float(np.std(predictions)),
            'feature_importance': avg_importance,
            'time_range': {
                'start': min(d.timestamp for d in decisions),
                'end': max(d.timestamp for d in decisions)
            }
        } 