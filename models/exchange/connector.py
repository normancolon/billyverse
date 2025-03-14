import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime
import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class OrderResponse:
    order_id: str
    symbol: str
    side: str
    amount: float
    price: float
    status: str
    timestamp: datetime

class ExchangeConnector:
    def __init__(self,
                 exchange_id: str,
                 api_key: str,
                 api_secret: str,
                 sandbox: bool = True):
        """
        Exchange connector for real-time trading
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'coinbase', 'alpaca')
            api_key: API key for authentication
            api_secret: API secret for authentication
            sandbox: Whether to use sandbox/paper trading
        """
        self.exchange_id = exchange_id
        self.sandbox = sandbox
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        if sandbox:
            self.exchange.set_sandbox_mode(True)
            
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize markets
        self.markets = {}
        self.update_markets()
        
    def update_markets(self):
        """Updates market information"""
        try:
            self.markets = self.exchange.load_markets()
        except Exception as e:
            self.logger.error(f"Failed to update markets: {str(e)}")
            raise
            
    async def get_ticker(self, symbol: str) -> Dict:
        """Gets current ticker data for a symbol"""
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return {
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'timestamp': ticker['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise
            
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """Gets current orderbook for a symbol"""
        try:
            orderbook = await self.exchange.fetch_order_book(symbol, limit)
            return {
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': orderbook['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch orderbook for {symbol}: {str(e)}")
            raise
            
    async def place_order(self,
                         symbol: str,
                         side: str,
                         amount: float,
                         price: Optional[float] = None,
                         order_type: str = 'limit') -> OrderResponse:
        """Places an order on the exchange"""
        try:
            # Validate inputs
            if symbol not in self.markets:
                raise ValueError(f"Invalid symbol: {symbol}")
            if side not in ['buy', 'sell']:
                raise ValueError(f"Invalid side: {side}")
                
            # Format order parameters
            params = {
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount
            }
            
            if order_type == 'limit':
                if price is None:
                    raise ValueError("Price required for limit orders")
                params['price'] = price
                
            # Place order
            response = await self.exchange.create_order(**params)
            
            return OrderResponse(
                order_id=response['id'],
                symbol=response['symbol'],
                side=response['side'],
                amount=float(response['amount']),
                price=float(response['price']) if 'price' in response else None,
                status=response['status'],
                timestamp=datetime.fromtimestamp(response['timestamp'] / 1000)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {str(e)}")
            raise
            
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancels an existing order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise
            
    async def get_order_status(self, order_id: str, symbol: str) -> Dict:
        """Gets the current status of an order"""
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return {
                'status': order['status'],
                'filled': order['filled'],
                'remaining': order['remaining'],
                'average_price': order['average']
            }
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            raise
            
    async def get_account_balance(self) -> Dict[str, float]:
        """Gets account balances"""
        try:
            balance = await self.exchange.fetch_balance()
            return {
                currency: float(data['free'])
                for currency, data in balance['free'].items()
                if float(data['free']) > 0
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {str(e)}")
            raise
            
    async def get_historical_data(self,
                                symbol: str,
                                timeframe: str = '1h',
                                limit: int = 1000) -> pd.DataFrame:
        """Gets historical OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            raise
            
    def calculate_position_value(self, symbol: str, amount: float, price: float) -> float:
        """Calculates position value in quote currency"""
        market = self.markets[symbol]
        return amount * price * (1 if market['quote'] == 'USD' else self.get_usd_rate(market['quote']))
        
    def get_usd_rate(self, currency: str) -> float:
        """Gets USD conversion rate for a currency"""
        if currency == 'USD':
            return 1.0
        try:
            ticker = self.exchange.fetch_ticker(f"{currency}/USD")
            return ticker['last']
        except:
            return 1.0  # Default to 1.0 if rate not available 