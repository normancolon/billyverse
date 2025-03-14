import os
import asyncio
import logging
from datetime import datetime
from decimal import Decimal

from dotenv import load_dotenv
from web3 import Web3

from models.quantum.trading import QuantumTrading
from models.web3.ai_contracts import Web3AIDeployer
from models.compliance.monitoring import ComplianceMonitor
from models.audit.logging import AIAuditLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TradingSystem:
    def __init__(self):
        # Initialize Web3
        self.w3 = Web3(Web3.HTTPProvider(os.getenv('ETH_NODE_URL')))
        self.account = self.w3.eth.account.from_key(os.getenv('PRIVATE_KEY'))
        
        # Initialize components
        self.quantum_trader = QuantumTrading(
            backend='qiskit',
            n_qubits=10
        )
        
        self.deployer = Web3AIDeployer(
            web3=self.w3,
            private_key=os.getenv('PRIVATE_KEY')
        )
        
        self.compliance = ComplianceMonitor(
            web3=self.w3,
            risk_threshold=0.8
        )
        
        self.audit_logger = AIAuditLogger(
            web3=self.w3,
            contract_address=os.getenv('AUDIT_CONTRACT_ADDRESS'),
            private_key=os.getenv('PRIVATE_KEY')
        )

    async def analyze_market(self):
        """Analyze market conditions and generate predictions"""
        logger.info("Analyzing market conditions...")
        
        # Get market data from multiple DEXs
        uniswap_prices = await self.get_pool_prices('uniswap')
        sushiswap_prices = await self.get_pool_prices('sushiswap')
        
        # Generate predictions using quantum ML
        predictions = self.quantum_trader.predict(
            features={
                'uniswap_prices': uniswap_prices,
                'sushiswap_prices': sushiswap_prices,
                'timestamp': datetime.now().timestamp()
            }
        )
        
        # Log the prediction with explanations
        await self.audit_logger.log_decision(
            model_id='quantum_ml_v1',
            input_features=predictions['features'],
            prediction=predictions['prediction'],
            confidence=predictions['confidence']
        )
        
        return predictions

    async def optimize_portfolio(self, predictions):
        """Optimize portfolio based on predictions"""
        logger.info("Optimizing portfolio allocation...")
        
        # Use quantum optimization for portfolio allocation
        optimal_portfolio = self.quantum_trader.optimize_portfolio(
            returns=predictions['expected_returns'],
            covariance=predictions['risk_matrix'],
            risk_tolerance=0.5  # Moderate risk tolerance
        )
        
        # Verify compliance
        compliance_check = await self.compliance.check_portfolio(
            portfolio=optimal_portfolio,
            account=self.account.address
        )
        
        if not compliance_check['compliant']:
            logger.warning(f"Compliance issues detected: {compliance_check['violations']}")
            return None
            
        return optimal_portfolio

    async def find_arbitrage(self):
        """Find arbitrage opportunities across DEXs"""
        logger.info("Searching for arbitrage opportunities...")
        
        opportunities = self.quantum_trader.find_arbitrage(
            exchanges=['uniswap', 'sushiswap', 'curve'],
            min_profit_threshold=0.01,  # 1% minimum profit
            max_path_length=3
        )
        
        return opportunities

    async def execute_trades(self, portfolio, opportunities):
        """Execute trades based on portfolio optimization and arbitrage"""
        logger.info("Executing trades...")
        
        # Deploy trading agent if not already deployed
        agent_address = await self.deployer.deploy_trading_agent(
            model_name="quantum_trading_v1",
            min_confidence=0.8,
            max_trade_amount=Decimal("1.0")  # 1 ETH max per trade
        )
        
        # Execute portfolio rebalancing
        if portfolio:
            for asset, weight in portfolio['weights'].items():
                trade_tx = await self.deployer.execute_trade(
                    agent_address=agent_address,
                    token_address=asset,
                    amount=weight,
                    side='BUY' if weight > 0 else 'SELL'
                )
                logger.info(f"Portfolio trade executed: {trade_tx.hex()}")
        
        # Execute arbitrage trades
        if opportunities:
            for opp in opportunities:
                if opp['expected_profit'] > 0.01:  # 1% profit threshold
                    arb_tx = await self.deployer.execute_arbitrage(
                        path=opp['path'],
                        amounts=opp['amounts']
                    )
                    logger.info(f"Arbitrage trade executed: {arb_tx.hex()}")

    async def monitor_performance(self):
        """Monitor and analyze trading performance"""
        logger.info("Analyzing performance...")
        
        # Get historical decisions and performance
        decisions = await self.audit_logger.get_model_decisions(
            model_id='quantum_ml_v1',
            start_time=datetime.now().timestamp() - 86400,  # Last 24 hours
            end_time=datetime.now().timestamp()
        )
        
        # Analyze model performance
        performance_metrics = self.audit_logger.analyze_model_performance(decisions)
        
        # Generate performance visualizations
        self.audit_logger.plot_model_metrics(
            metrics=performance_metrics,
            save_path='results/performance_analysis.png'
        )
        
        return performance_metrics

async def main():
    # Initialize trading system
    system = TradingSystem()
    
    try:
        # 1. Analyze market and generate predictions
        predictions = await system.analyze_market()
        
        # 2. Optimize portfolio
        optimal_portfolio = await system.optimize_portfolio(predictions)
        
        # 3. Find arbitrage opportunities
        arbitrage_opportunities = await system.find_arbitrage()
        
        # 4. Execute trades
        await system.execute_trades(optimal_portfolio, arbitrage_opportunities)
        
        # 5. Monitor and analyze performance
        performance = await system.monitor_performance()
        
        logger.info("Trading cycle completed successfully")
        logger.info(f"Performance metrics: {performance}")
        
    except Exception as e:
        logger.error(f"Error in trading cycle: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 