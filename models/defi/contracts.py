from typing import Dict, List, Optional, Tuple
import json
import logging
from web3 import Web3
from eth_account.account import Account
from eth_account.signers.local import LocalAccount
import asyncio
from decimal import Decimal

logger = logging.getLogger("billieverse.defi.contracts")

# ABIs
UNISWAP_V2_ROUTER_ABI = json.loads('''[
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"reserveIn","type":"uint256"},{"internalType":"uint256","name":"reserveOut","type":"uint256"}],"name":"getAmountOut","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"pure","type":"function"},
    {"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountAMin","type":"uint256"},{"internalType":"uint256","name":"amountBMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"removeLiquidity","outputs":[{"internalType":"uint256","name":"amountA","type":"uint256"},{"internalType":"uint256","name":"amountB","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactETHForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"payable","type":"function"}
]''')

UNISWAP_V2_PAIR_ABI = json.loads('''[
    {"constant":true,"inputs":[],"name":"getReserves","outputs":[{"internalType":"uint112","name":"_reserve0","type":"uint112"},{"internalType":"uint112","name":"_reserve1","type":"uint112"},{"internalType":"uint32","name":"_blockTimestampLast","type":"uint32"}],"payable":false,"stateMutability":"view","type":"function"}
]''')

ERC20_ABI = json.loads('''[
    {"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},
    {"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},
    {"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"}
]''')

class DeFiContract:
    """Base class for DeFi contract interactions"""
    
    def __init__(
        self,
        web3: Web3,
        address: str,
        abi: List[Dict],
        private_key: Optional[str] = None
    ):
        self.web3 = web3
        self.address = Web3.to_checksum_address(address)
        self.contract = web3.eth.contract(address=self.address, abi=abi)
        
        # Set up account if private key provided
        self.account: Optional[LocalAccount] = None
        if private_key:
            self.account = Account.from_key(private_key)
    
    async def _send_transaction(
        self,
        func,
        value: int = 0,
        gas_limit: Optional[int] = None
    ) -> str:
        """Send transaction to contract"""
        if not self.account:
            raise ValueError("No account configured")
            
        # Build transaction
        nonce = await self.web3.eth.get_transaction_count(
            self.account.address
        )
        
        tx = func.build_transaction({
            'from': self.account.address,
            'value': value,
            'nonce': nonce,
            'gas': gas_limit or 2000000,
            'gasPrice': await self.web3.eth.gas_price
        })
        
        # Sign and send
        signed = self.account.sign_transaction(tx)
        tx_hash = await self.web3.eth.send_raw_transaction(
            signed.rawTransaction
        )
        
        # Wait for receipt
        receipt = await self.web3.eth.wait_for_transaction_receipt(
            tx_hash
        )
        
        return receipt

class UniswapV2Router(DeFiContract):
    """Uniswap V2 Router contract interactions"""
    
    def __init__(
        self,
        web3: Web3,
        router_address: str,
        private_key: Optional[str] = None
    ):
        super().__init__(
            web3,
            router_address,
            UNISWAP_V2_ROUTER_ABI,
            private_key
        )
    
    async def get_amounts_out(
        self,
        amount_in: int,
        path: List[str]
    ) -> List[int]:
        """Calculate output amounts for swap"""
        return await self.contract.functions.getAmountsOut(
            amount_in,
            [Web3.to_checksum_address(addr) for addr in path]
        ).call()
    
    async def swap_exact_eth_for_tokens(
        self,
        eth_amount: int,
        min_tokens: int,
        token_path: List[str],
        to_address: str,
        deadline: int
    ) -> str:
        """Swap exact ETH for tokens"""
        func = self.contract.functions.swapExactETHForTokens(
            min_tokens,
            [Web3.to_checksum_address(addr) for addr in token_path],
            Web3.to_checksum_address(to_address),
            deadline
        )
        
        receipt = await self._send_transaction(
            func,
            value=eth_amount
        )
        return receipt['transactionHash'].hex()
    
    async def remove_liquidity(
        self,
        token_a: str,
        token_b: str,
        liquidity: int,
        min_a: int,
        min_b: int,
        to_address: str,
        deadline: int
    ) -> Tuple[int, int]:
        """Remove liquidity from pool"""
        func = self.contract.functions.removeLiquidity(
            Web3.to_checksum_address(token_a),
            Web3.to_checksum_address(token_b),
            liquidity,
            min_a,
            min_b,
            Web3.to_checksum_address(to_address),
            deadline
        )
        
        receipt = await self._send_transaction(func)
        return (
            receipt['events']['RemoveLiquidity']['returnValues']['amount0'],
            receipt['events']['RemoveLiquidity']['returnValues']['amount1']
        )

class UniswapV2Pair(DeFiContract):
    """Uniswap V2 Pair contract interactions"""
    
    def __init__(
        self,
        web3: Web3,
        pair_address: str
    ):
        super().__init__(web3, pair_address, UNISWAP_V2_PAIR_ABI)
    
    async def get_reserves(self) -> Tuple[int, int, int]:
        """Get pool reserves"""
        return await self.contract.functions.getReserves().call()

class ERC20Token(DeFiContract):
    """ERC20 token contract interactions"""
    
    def __init__(
        self,
        web3: Web3,
        token_address: str,
        private_key: Optional[str] = None
    ):
        super().__init__(
            web3,
            token_address,
            ERC20_ABI,
            private_key
        )
        
    async def decimals(self) -> int:
        """Get token decimals"""
        return await self.contract.functions.decimals().call()
    
    async def balance_of(
        self,
        address: str
    ) -> int:
        """Get token balance"""
        return await self.contract.functions.balanceOf(
            Web3.to_checksum_address(address)
        ).call()
    
    async def approve(
        self,
        spender: str,
        amount: int
    ) -> bool:
        """Approve spender"""
        func = self.contract.functions.approve(
            Web3.to_checksum_address(spender),
            amount
        )
        receipt = await self._send_transaction(func)
        return bool(receipt['status']) 