from typing import Dict, List, Optional, Union, Any
import json
import numpy as np
import tensorflow as tf
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_account.messages import encode_defunct
import zk_snarks
from dataclasses import dataclass
import logging
from pathlib import Path
import asyncio

logger = logging.getLogger("billieverse.web3.ai_contracts")

# Solidity contract templates
MODEL_CONTRACT_TEMPLATE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AIModel {
    address public owner;
    bytes32 public modelHash;
    mapping(bytes32 => bool) public predictions;
    mapping(bytes32 => uint256) public confidenceScores;
    
    event PredictionMade(
        bytes32 indexed inputHash,
        bytes32 indexed outputHash,
        uint256 confidence,
        uint256 timestamp
    );
    
    constructor(bytes32 _modelHash) {
        owner = msg.sender;
        modelHash = _modelHash;
    }
    
    function recordPrediction(
        bytes32 inputHash,
        bytes32 outputHash,
        uint256 confidence
    ) external {
        require(msg.sender == owner, "Only owner can record predictions");
        predictions[inputHash] = true;
        confidenceScores[inputHash] = confidence;
        emit PredictionMade(inputHash, outputHash, confidence, block.timestamp);
    }
    
    function verifyPrediction(
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata proof
    ) external view returns (bool) {
        require(predictions[inputHash], "Prediction not found");
        // Verify zero-knowledge proof
        return true; // Placeholder for ZK proof verification
    }
}
'''

TRADING_AGENT_TEMPLATE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IAIModel {
    function verifyPrediction(
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata proof
    ) external view returns (bool);
}

contract TradingAgent {
    address public owner;
    address public aiModel;
    uint256 public minConfidence;
    uint256 public maxTradeAmount;
    
    event TradeExecuted(
        address indexed token,
        bool isBuy,
        uint256 amount,
        uint256 price,
        bytes32 predictionHash
    );
    
    constructor(
        address _aiModel,
        uint256 _minConfidence,
        uint256 _maxTradeAmount
    ) {
        owner = msg.sender;
        aiModel = _aiModel;
        minConfidence = _minConfidence;
        maxTradeAmount = _maxTradeAmount;
    }
    
    function executeTrade(
        address token,
        bool isBuy,
        uint256 amount,
        uint256 price,
        bytes32 inputHash,
        bytes32 outputHash,
        bytes calldata proof
    ) external {
        require(msg.sender == owner, "Only owner can execute trades");
        require(amount <= maxTradeAmount, "Amount exceeds limit");
        require(
            IAIModel(aiModel).verifyPrediction(inputHash, outputHash, proof),
            "Invalid prediction proof"
        );
        
        // Execute trade logic here
        emit TradeExecuted(token, isBuy, amount, price, outputHash);
    }
}
'''

@dataclass
class ModelDeployment:
    """Represents a deployed AI model on blockchain"""
    contract_address: str
    model_hash: str
    deployment_tx: str
    abi: Dict
    bytecode: str

@dataclass
class TradingAgentDeployment:
    """Represents a deployed trading agent"""
    contract_address: str
    model_address: str
    deployment_tx: str
    abi: Dict
    bytecode: str

class Web3AIDeployer:
    """Deploys and manages AI models on blockchain"""
    
    def __init__(
        self,
        web3: Web3,
        private_key: str,
        chain_id: int,
        gas_price_gwei: int = 50
    ):
        self.web3 = web3
        self.account = Account.from_key(private_key)
        self.chain_id = chain_id
        self.gas_price = Web3.to_wei(gas_price_gwei, 'gwei')
        
        # Store deployed contracts
        self.deployed_models: Dict[str, ModelDeployment] = {}
        self.deployed_agents: Dict[str, TradingAgentDeployment] = {}
    
    def _compile_contract(
        self,
        source_code: str,
        contract_name: str
    ) -> Dict:
        """Compile Solidity contract"""
        try:
            # Note: In production, use proper Solidity compiler
            # This is a simplified version
            from solcx import compile_source
            
            compiled = compile_source(
                source_code,
                output_values=['abi', 'bin']
            )
            contract_interface = compiled[f'<stdin>:{contract_name}']
            
            return {
                'abi': contract_interface['abi'],
                'bytecode': contract_interface['bin']
            }
            
        except Exception as e:
            logger.error(f"Error compiling contract: {str(e)}")
            raise
    
    def _get_contract_instance(
        self,
        address: str,
        abi: Dict
    ) -> Contract:
        """Get contract instance"""
        return self.web3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi
        )
    
    def _hash_model(self, model: tf.keras.Model) -> str:
        """Create deterministic hash of model architecture and weights"""
        try:
            # Get model architecture
            config = model.get_config()
            
            # Get model weights
            weights = []
            for layer in model.layers:
                layer_weights = layer.get_weights()
                if layer_weights:
                    weights.extend([w.tobytes() for w in layer_weights])
            
            # Create combined hash
            combined = json.dumps(config, sort_keys=True).encode()
            for w in weights:
                combined += w
            
            return Web3.keccak(combined).hex()
            
        except Exception as e:
            logger.error(f"Error hashing model: {str(e)}")
            raise
    
    def _create_zk_proof(
        self,
        model: tf.keras.Model,
        inputs: np.ndarray,
        prediction: np.ndarray
    ) -> bytes:
        """Create zero-knowledge proof of model prediction"""
        try:
            # Note: This is a placeholder for actual ZK-SNARK implementation
            # In production, use a proper ZK-SNARK library
            circuit = {
                'model_hash': self._hash_model(model),
                'input_shape': inputs.shape,
                'output_shape': prediction.shape
            }
            
            witness = {
                'inputs': inputs.tolist(),
                'prediction': prediction.tolist()
            }
            
            # Generate proof (placeholder)
            proof = zk_snarks.generate_proof(circuit, witness)
            
            return proof
            
        except Exception as e:
            logger.error(f"Error creating ZK proof: {str(e)}")
            raise
    
    async def deploy_model(
        self,
        model: tf.keras.Model,
        model_name: str
    ) -> ModelDeployment:
        """Deploy AI model as smart contract"""
        try:
            # Compile contract
            contract_interface = self._compile_contract(
                MODEL_CONTRACT_TEMPLATE,
                'AIModel'
            )
            
            # Create contract instance
            contract = self.web3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bytecode']
            )
            
            # Get model hash
            model_hash = self._hash_model(model)
            
            # Build constructor transaction
            nonce = await self.web3.eth.get_transaction_count(
                self.account.address
            )
            
            constructor_tx = contract.constructor(
                model_hash
            ).build_transaction({
                'chainId': self.chain_id,
                'gas': 2000000,
                'gasPrice': self.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(constructor_tx)
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(
                tx_hash
            )
            
            deployment = ModelDeployment(
                contract_address=receipt['contractAddress'],
                model_hash=model_hash,
                deployment_tx=tx_hash.hex(),
                abi=contract_interface['abi'],
                bytecode=contract_interface['bytecode']
            )
            
            self.deployed_models[model_name] = deployment
            
            logger.info(
                f"Model deployed at {deployment.contract_address}"
            )
            return deployment
            
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise
    
    async def deploy_trading_agent(
        self,
        model_name: str,
        min_confidence: float = 0.8,
        max_trade_amount: float = 1.0
    ) -> TradingAgentDeployment:
        """Deploy trading agent contract"""
        try:
            if model_name not in self.deployed_models:
                raise ValueError(f"Model {model_name} not deployed")
            
            # Compile contract
            contract_interface = self._compile_contract(
                TRADING_AGENT_TEMPLATE,
                'TradingAgent'
            )
            
            # Create contract instance
            contract = self.web3.eth.contract(
                abi=contract_interface['abi'],
                bytecode=contract_interface['bytecode']
            )
            
            # Build constructor transaction
            nonce = await self.web3.eth.get_transaction_count(
                self.account.address
            )
            
            constructor_tx = contract.constructor(
                self.deployed_models[model_name].contract_address,
                int(min_confidence * 100),
                Web3.to_wei(max_trade_amount, 'ether')
            ).build_transaction({
                'chainId': self.chain_id,
                'gas': 2000000,
                'gasPrice': self.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(constructor_tx)
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(
                tx_hash
            )
            
            deployment = TradingAgentDeployment(
                contract_address=receipt['contractAddress'],
                model_address=self.deployed_models[model_name].contract_address,
                deployment_tx=tx_hash.hex(),
                abi=contract_interface['abi'],
                bytecode=contract_interface['bytecode']
            )
            
            self.deployed_agents[model_name] = deployment
            
            logger.info(
                f"Trading agent deployed at {deployment.contract_address}"
            )
            return deployment
            
        except Exception as e:
            logger.error(f"Error deploying trading agent: {str(e)}")
            raise
    
    async def record_prediction(
        self,
        model_name: str,
        inputs: np.ndarray,
        prediction: np.ndarray,
        confidence: float
    ):
        """Record model prediction on blockchain"""
        try:
            if model_name not in self.deployed_models:
                raise ValueError(f"Model {model_name} not deployed")
            
            deployment = self.deployed_models[model_name]
            contract = self._get_contract_instance(
                deployment.contract_address,
                deployment.abi
            )
            
            # Create hashes
            input_hash = Web3.keccak(inputs.tobytes()).hex()
            output_hash = Web3.keccak(prediction.tobytes()).hex()
            
            # Build transaction
            nonce = await self.web3.eth.get_transaction_count(
                self.account.address
            )
            
            tx = contract.functions.recordPrediction(
                input_hash,
                output_hash,
                int(confidence * 100)
            ).build_transaction({
                'chainId': self.chain_id,
                'gas': 200000,
                'gasPrice': self.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for receipt
            await self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Prediction recorded with hash {output_hash}")
            
        except Exception as e:
            logger.error(f"Error recording prediction: {str(e)}")
            raise
    
    async def execute_trade(
        self,
        model_name: str,
        token_address: str,
        is_buy: bool,
        amount: float,
        price: float,
        inputs: np.ndarray,
        prediction: np.ndarray
    ):
        """Execute trade through trading agent"""
        try:
            if model_name not in self.deployed_agents:
                raise ValueError(f"Trading agent for {model_name} not deployed")
            
            deployment = self.deployed_agents[model_name]
            contract = self._get_contract_instance(
                deployment.contract_address,
                deployment.abi
            )
            
            # Create hashes and proof
            input_hash = Web3.keccak(inputs.tobytes()).hex()
            output_hash = Web3.keccak(prediction.tobytes()).hex()
            proof = self._create_zk_proof(
                self.deployed_models[model_name],
                inputs,
                prediction
            )
            
            # Build transaction
            nonce = await self.web3.eth.get_transaction_count(
                self.account.address
            )
            
            tx = contract.functions.executeTrade(
                Web3.to_checksum_address(token_address),
                is_buy,
                Web3.to_wei(amount, 'ether'),
                int(price * 1e8),  # Convert to 8 decimal places
                input_hash,
                output_hash,
                proof
            ).build_transaction({
                'chainId': self.chain_id,
                'gas': 500000,
                'gasPrice': self.gas_price,
                'nonce': nonce,
                'from': self.account.address
            })
            
            # Sign and send transaction
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = await self.web3.eth.send_raw_transaction(
                signed_tx.rawTransaction
            )
            
            # Wait for receipt
            receipt = await self.web3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(
                f"Trade executed: {amount} {'bought' if is_buy else 'sold'} "
                f"at {price}"
            )
            
            return receipt
            
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            raise 