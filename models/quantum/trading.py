from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms import QAOA, VQE
from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import TwoLocal
from qiskit_finance.applications import PortfolioOptimization
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from dimod import Binary, BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite
import logging
from dataclasses import dataclass

logger = logging.getLogger("billieverse.quantum.trading")

@dataclass
class QuantumPortfolio:
    """Represents a quantum-optimized portfolio"""
    weights: np.ndarray
    expected_return: float
    risk: float
    sharpe_ratio: float
    quantum_circuit: Optional[QuantumCircuit]
    optimization_result: Dict

@dataclass
class QuantumPrediction:
    """Represents a quantum ML prediction"""
    prediction: float
    confidence: float
    quantum_state: np.ndarray
    circuit_depth: int
    execution_time: float

class QuantumTrading:
    """Quantum-enhanced trading system"""
    
    def __init__(
        self,
        backend: str = 'qiskit',  # 'qiskit' or 'dwave'
        api_token: Optional[str] = None,
        n_qubits: int = 10
    ):
        self.backend = backend
        self.n_qubits = n_qubits
        
        # Initialize quantum backend
        if backend == 'qiskit':
            self.quantum_instance = qiskit.Aer.get_backend('aer_simulator')
        elif backend == 'dwave':
            self.sampler = EmbeddingComposite(
                DWaveSampler(token=api_token)
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        
        # Initialize quantum ML models
        self.vqc = None  # Variational Quantum Classifier
        self.qsvc = None  # Quantum Support Vector Classifier
        self.quantum_kernel = None
    
    def _create_portfolio_circuit(
        self,
        returns: np.ndarray,
        risk_matrix: np.ndarray,
        risk_factor: float = 1.0
    ) -> QuantumCircuit:
        """Create quantum circuit for portfolio optimization"""
        n_assets = len(returns)
        
        # Create quantum registers
        qr_assets = QuantumRegister(n_assets, 'assets')
        qr_aux = QuantumRegister(self.n_qubits - n_assets, 'aux')
        cr = ClassicalRegister(n_assets, 'measurements')
        
        circuit = QuantumCircuit(qr_assets, qr_aux, cr)
        
        # Prepare initial state
        for i in range(n_assets):
            circuit.h(qr_assets[i])  # Superposition
        
        # Add QAOA layers
        for i in range(n_assets):
            # Return maximization
            circuit.rz(returns[i], qr_assets[i])
            
            # Risk minimization
            for j in range(i+1, n_assets):
                circuit.cx(qr_assets[i], qr_assets[j])
                circuit.rz(risk_matrix[i,j] * risk_factor, qr_assets[j])
                circuit.cx(qr_assets[i], qr_assets[j])
        
        # Measurement
        circuit.measure(qr_assets, cr)
        
        return circuit
    
    def _create_prediction_circuit(
        self,
        features: np.ndarray,
        n_layers: int = 2
    ) -> QuantumCircuit:
        """Create quantum circuit for prediction"""
        n_features = len(features)
        
        # Create quantum registers
        qr = QuantumRegister(self.n_qubits, 'qr')
        cr = ClassicalRegister(1, 'prediction')
        
        circuit = QuantumCircuit(qr, cr)
        
        # Encode features
        for i in range(min(n_features, self.n_qubits)):
            circuit.ry(features[i], qr[i])
        
        # Add variational layers
        for _ in range(n_layers):
            # Entangling layer
            for i in range(self.n_qubits-1):
                circuit.cx(qr[i], qr[i+1])
            
            # Rotation layer
            for i in range(self.n_qubits):
                circuit.ry(features[i % n_features], qr[i])
        
        # Measurement
        circuit.measure(qr[0], cr)
        
        return circuit
    
    def optimize_portfolio(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        risk_tolerance: float = 0.5,
        n_shots: int = 1000
    ) -> QuantumPortfolio:
        """Optimize portfolio using quantum computing"""
        try:
            n_assets = len(returns)
            
            if self.backend == 'qiskit':
                # Create QAOA instance
                circuit = self._create_portfolio_circuit(
                    returns,
                    covariance,
                    risk_tolerance
                )
                
                # Run optimization
                result = qiskit.execute(
                    circuit,
                    self.quantum_instance,
                    shots=n_shots
                ).result()
                
                # Process results
                counts = result.get_counts()
                best_state = max(counts.items(), key=lambda x: x[1])[0]
                weights = np.array([int(x) for x in best_state]) / n_shots
                
            elif self.backend == 'dwave':
                # Create BQM
                bqm = BinaryQuadraticModel('BINARY')
                
                # Add return terms
                for i in range(n_assets):
                    bqm.add_variable(
                        Binary(f'x{i}'),
                        -returns[i]  # Negative for minimization
                    )
                
                # Add risk terms
                for i in range(n_assets):
                    for j in range(i+1, n_assets):
                        bqm.add_interaction(
                            Binary(f'x{i}'),
                            Binary(f'x{j}'),
                            risk_tolerance * covariance[i,j]
                        )
                
                # Sample solution
                sampleset = self.sampler.sample(
                    bqm,
                    num_reads=n_shots
                )
                
                # Get best solution
                best_sample = sampleset.first.sample
                weights = np.array([
                    best_sample[f'x{i}'] for i in range(n_assets)
                ])
            
            # Calculate portfolio metrics
            expected_return = np.dot(weights, returns)
            risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            sharpe_ratio = expected_return / risk if risk > 0 else 0
            
            return QuantumPortfolio(
                weights=weights,
                expected_return=float(expected_return),
                risk=float(risk),
                sharpe_ratio=float(sharpe_ratio),
                quantum_circuit=circuit if self.backend == 'qiskit' else None,
                optimization_result={
                    'n_shots': n_shots,
                    'risk_tolerance': risk_tolerance,
                    'backend': self.backend
                }
            )
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            raise
    
    def train_quantum_ml(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_layers: int = 2,
        n_shots: int = 1000
    ):
        """Train quantum ML models"""
        try:
            if self.backend == 'qiskit':
                # Create quantum feature map
                feature_map = TwoLocal(
                    self.n_qubits,
                    ['ry', 'rz'],
                    'cx',
                    reps=n_layers
                )
                
                # Initialize quantum kernel
                self.quantum_kernel = QuantumKernel(
                    feature_map=feature_map,
                    quantum_instance=self.quantum_instance
                )
                
                # Train quantum SVC
                self.qsvc = QSVC(
                    quantum_kernel=self.quantum_kernel
                )
                self.qsvc.fit(X_train, y_train)
                
                # Train variational quantum classifier
                self.vqc = VQC(
                    feature_map=feature_map,
                    ansatz=TwoLocal(
                        self.n_qubits,
                        ['ry', 'rz'],
                        'cx',
                        reps=n_layers
                    ),
                    optimizer=SPSA(maxiter=100),
                    quantum_instance=self.quantum_instance
                )
                self.vqc.fit(X_train, y_train)
                
            elif self.backend == 'dwave':
                logger.warning(
                    "Quantum ML training not implemented for D-Wave backend"
                )
            
            logger.info("Successfully trained quantum ML models")
            
        except Exception as e:
            logger.error(f"Error training quantum ML models: {str(e)}")
            raise
    
    def predict(
        self,
        features: np.ndarray,
        method: str = 'vqc'  # 'vqc' or 'qsvc'
    ) -> QuantumPrediction:
        """Make prediction using quantum ML"""
        try:
            start_time = datetime.now()
            
            if self.backend == 'qiskit':
                if method == 'vqc' and self.vqc is not None:
                    # Predict using VQC
                    circuit = self._create_prediction_circuit(
                        features,
                        n_layers=2
                    )
                    prediction = float(self.vqc.predict(features.reshape(1, -1))[0])
                    confidence = float(
                        max(self.vqc.predict_proba(features.reshape(1, -1))[0])
                    )
                    quantum_state = self.vqc._circuit.bind_parameters(
                        self.vqc._optimal_params
                    ).data
                    
                elif method == 'qsvc' and self.qsvc is not None:
                    # Predict using QSVC
                    prediction = float(self.qsvc.predict(features.reshape(1, -1))[0])
                    confidence = float(
                        max(self.qsvc.predict_proba(features.reshape(1, -1))[0])
                    )
                    quantum_state = self.quantum_kernel.evaluate(
                        features.reshape(1, -1),
                        features.reshape(1, -1)
                    )
                    
                else:
                    raise ValueError(f"Invalid method or model not trained: {method}")
                
            elif self.backend == 'dwave':
                raise NotImplementedError(
                    "Quantum ML prediction not implemented for D-Wave backend"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return QuantumPrediction(
                prediction=prediction,
                confidence=confidence,
                quantum_state=quantum_state,
                circuit_depth=circuit.depth() if self.backend == 'qiskit' else 0,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in quantum prediction: {str(e)}")
            raise
    
    def assess_risk(
        self,
        portfolio: QuantumPortfolio,
        market_data: pd.DataFrame,
        n_scenarios: int = 1000
    ) -> Dict[str, float]:
        """Assess risk using quantum-inspired algorithms"""
        try:
            # Calculate VaR and CVaR using quantum superposition
            if self.backend == 'qiskit':
                # Create quantum circuit for risk assessment
                qr = QuantumRegister(self.n_qubits, 'qr')
                cr = ClassicalRegister(self.n_qubits, 'cr')
                circuit = QuantumCircuit(qr, cr)
                
                # Encode portfolio weights
                for i, weight in enumerate(portfolio.weights):
                    if i < self.n_qubits:
                        circuit.ry(weight * np.pi, qr[i])
                
                # Add entangling layers
                for i in range(self.n_qubits - 1):
                    circuit.cx(qr[i], qr[i+1])
                
                # Measure
                circuit.measure(qr, cr)
                
                # Execute
                result = qiskit.execute(
                    circuit,
                    self.quantum_instance,
                    shots=n_scenarios
                ).result()
                
                # Calculate risk metrics
                counts = result.get_counts()
                losses = []
                for state, count in counts.items():
                    # Convert quantum state to loss
                    loss = sum(
                        int(x) * w * r for x, w, r in zip(
                            state,
                            portfolio.weights,
                            market_data['returns'].values
                        )
                    )
                    losses.extend([loss] * count)
                
                losses = np.array(losses)
                var_95 = np.percentile(losses, 95)
                cvar_95 = np.mean(losses[losses >= var_95])
                
            elif self.backend == 'dwave':
                # Use quantum annealing for scenario generation
                bqm = BinaryQuadraticModel('BINARY')
                
                # Add risk terms
                for i, weight in enumerate(portfolio.weights):
                    bqm.add_variable(
                        Binary(f'x{i}'),
                        weight * market_data['returns'].values[i]
                    )
                
                # Sample scenarios
                sampleset = self.sampler.sample(
                    bqm,
                    num_reads=n_scenarios
                )
                
                # Calculate risk metrics
                losses = []
                for sample in sampleset.record:
                    loss = sum(
                        x * w * r for x, w, r in zip(
                            sample.sample,
                            portfolio.weights,
                            market_data['returns'].values
                        )
                    )
                    losses.extend([loss] * sample.num_occurrences)
                
                losses = np.array(losses)
                var_95 = np.percentile(losses, 95)
                cvar_95 = np.mean(losses[losses >= var_95])
            
            return {
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'max_loss': float(np.max(losses)),
                'volatility': float(np.std(losses)),
                'n_scenarios': n_scenarios
            }
            
        except Exception as e:
            logger.error(f"Error in risk assessment: {str(e)}")
            raise 