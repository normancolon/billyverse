import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from web3 import Web3
import tensorflow as tf

from models.compliance.monitoring import ComplianceMonitor, ComplianceViolation
from infrastructure.config import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.compliance")

# Create results directory
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

def create_compliance_models():
    """Create and train simple compliance models"""
    models_dir = results_dir / "compliance_models"
    models_dir.mkdir(exist_ok=True)
    
    # 1. Transaction Monitoring Model
    tx_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    tx_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train on dummy data
    n_samples = 1000
    X_tx = np.random.random((n_samples, 20))
    y_tx = (np.random.random(n_samples) > 0.8).astype(float)  # 20% anomalies
    
    tx_model.fit(X_tx, y_tx, epochs=5, verbose=0)
    tx_model.save(models_dir / "transaction_model")
    
    # 2. KYC Model
    kyc_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    kyc_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    # Train on dummy data
    X_kyc = np.random.random((n_samples, 50))
    y_kyc = np.eye(3)[np.random.choice(3, n_samples, p=[0.7, 0.2, 0.1])]
    
    kyc_model.fit(X_kyc, y_kyc, epochs=5, verbose=0)
    kyc_model.save(models_dir / "kyc_model")
    
    # 3. Insider Trading Model
    insider_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(None, 30)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    insider_model.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train on dummy data
    X_insider = np.random.random((n_samples, 10, 30))
    y_insider = (np.random.random(n_samples) > 0.95).astype(float)  # 5% insider patterns
    
    insider_model.fit(X_insider, y_insider, epochs=5, verbose=0)
    insider_model.save(models_dir / "insider_model")
    
    return str(models_dir)

def generate_sample_transaction():
    """Generate a sample transaction"""
    return {
        'hash': f"0x{''.join([str(np.random.randint(0, 9)) for _ in range(64)])}",
        'from': f"0x{''.join([str(np.random.randint(0, 9)) for _ in range(40)])}",
        'to': f"0x{''.join([str(np.random.randint(0, 9)) for _ in range(40)])}",
        'value': str(int(np.random.random() * 10**18)),  # 0-1 ETH
        'gas': str(int(21000 + np.random.random() * 100000)),
        'gasPrice': str(int(1e9 * (20 + np.random.random() * 100))),  # 20-120 gwei
        'input': '0x',
        'timestamp': int(datetime.now().timestamp() - np.random.random() * 86400)
    }

def generate_sample_kyc_data(risk_level='low'):
    """Generate sample KYC data"""
    if risk_level == 'high':
        base_score = 0.3
        pep_prob = 0.8
        sanctions_prob = 0.6
    elif risk_level == 'medium':
        base_score = 0.6
        pep_prob = 0.3
        sanctions_prob = 0.2
    else:
        base_score = 0.9
        pep_prob = 0.1
        sanctions_prob = 0.05
    
    return {
        'id_score': base_score + np.random.random() * 0.1,
        'face_match_score': base_score + np.random.random() * 0.1,
        'address_match_score': base_score + np.random.random() * 0.1,
        'politically_exposed': np.random.random() < pep_prob,
        'sanctions_match': np.random.random() < sanctions_prob,
        'transaction_history': {
            'total_volume': np.random.random() * 1000,
            'avg_transaction_size': np.random.random() * 10,
            'unique_counterparties': int(np.random.random() * 100),
            'cross_border_ratio': np.random.random(),
            'high_risk_country_ratio': np.random.random() * 0.3
        }
    }

def generate_sample_market_data(days=30):
    """Generate sample market data"""
    periods = days * 24  # Hourly data
    timestamps = pd.date_range(
        end=datetime.now(),
        periods=periods,
        freq='H'
    )
    
    # Generate price movement
    base_price = 1800  # Base ETH price
    price = base_price + np.random.randn(periods).cumsum() * 10
    
    # Generate volume
    volume = np.abs(np.random.normal(1000, 200, periods))
    
    return pd.DataFrame({
        'price': price,
        'volume': volume
    }, index=timestamps)

async def run_compliance_example():
    """Run compliance monitoring example"""
    try:
        logger.info("Starting compliance monitoring example...")
        
        # Connect to Ethereum network
        web3 = Web3(Web3.HTTPProvider(os.getenv('ETH_NODE_URL')))
        
        if not web3.is_connected():
            raise Exception("Failed to connect to Ethereum network")
        
        # Create and train models
        model_path = create_compliance_models()
        
        # Initialize compliance monitor
        monitor = ComplianceMonitor(
            web3,
            model_path,
            risk_threshold=0.7
        )
        
        # 1. Transaction Compliance
        logger.info("\nChecking transaction compliance...")
        
        # Generate sample transactions
        normal_tx = generate_sample_transaction()
        suspicious_tx = generate_sample_transaction()
        suspicious_tx['value'] = str(int(1e20))  # Unusually large amount
        
        # Check transactions
        wallet_history = [generate_sample_transaction() for _ in range(10)]
        
        violation = await monitor.check_transaction_compliance(
            normal_tx,
            wallet_history
        )
        if violation:
            monitor.add_violation(violation)
        
        violation = await monitor.check_transaction_compliance(
            suspicious_tx,
            wallet_history
        )
        if violation:
            monitor.add_violation(violation)
        
        # 2. KYC/AML Verification
        logger.info("\nPerforming KYC/AML checks...")
        
        # Check different risk levels
        for risk_level in ['low', 'medium', 'high']:
            wallet = f"0x{''.join([str(np.random.randint(0, 9)) for _ in range(40)])}"
            kyc_data = generate_sample_kyc_data(risk_level)
            
            violation = await monitor.verify_kyc_aml(wallet, kyc_data)
            if violation:
                monitor.add_violation(violation)
        
        # 3. Insider Trading Detection
        logger.info("\nAnalyzing for insider trading patterns...")
        
        # Generate market data
        market_data = generate_sample_market_data()
        
        # Generate suspicious trading pattern
        insider_txs = []
        base_time = int(datetime.now().timestamp())
        
        # Add trades before price movement
        for i in range(3):
            tx = generate_sample_transaction()
            tx['timestamp'] = base_time - 86400 + i * 3600
            tx['value'] = str(int(1e18))  # 1 ETH
            insider_txs.append(tx)
        
        # Add trades after price movement
        for i in range(3):
            tx = generate_sample_transaction()
            tx['timestamp'] = base_time + i * 3600
            tx['value'] = str(int(2e18))  # 2 ETH
            insider_txs.append(tx)
        
        violations = await monitor.detect_insider_trading(
            insider_txs,
            market_data
        )
        for violation in violations:
            monitor.add_violation(violation)
        
        # Get violation summary
        all_violations = monitor.get_violations(min_severity="MEDIUM")
        
        logger.info("\nCompliance Monitoring Summary:")
        logger.info(f"Total violations detected: {len(all_violations)}")
        
        severity_count = {
            "MEDIUM": 0,
            "HIGH": 0,
            "CRITICAL": 0
        }
        
        type_count = {
            "SUSPICIOUS_TRANSACTION": 0,
            "KYC_AML": 0,
            "INSIDER_TRADING": 0
        }
        
        for v in all_violations:
            severity_count[v.severity] += 1
            type_count[v.type] += 1
        
        logger.info("\nViolations by Severity:")
        for severity, count in severity_count.items():
            logger.info(f"  {severity}: {count}")
        
        logger.info("\nViolations by Type:")
        for type_, count in type_count.items():
            logger.info(f"  {type_}: {count}")
        
        logger.info("\nCompliance monitoring example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in compliance example: {str(e)}")
        raise

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run example
    asyncio.run(run_compliance_example()) 