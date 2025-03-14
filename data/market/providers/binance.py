from typing import Dict, Any, List, Optional, AsyncGenerator
import pandas as pd
from datetime import datetime
import aiohttp
import asyncio
import json
from data.market.base import MarketDataProvider
from core.config import settings

class BinanceProvider(MarketDataProvider):
    """Binance market data provider implementation"""
    
    def __init__(self, trading_pairs: List[str]):
        super().__init__(trading_pairs)
        self.base_url = "https://api.binance.com"
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.api_key = settings.api_keys.BINANCE_API_KEY
        self.api_secret = settings.api_keys.BINANCE_SECRET_KEY
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[aiohttp.ClientWebSocketResponse] = None
    
    async def connect(self) -> bool:
        """Establish connection to Binance"""
        try:
            self.session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
            self.is_connected = True
            self.logger.info("Connected to Binance REST API")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Binance: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Binance"""
        try:
            if self.session:
                await self.session.close()
            if self.ws:
                await self.ws.close()
            self.is_connected = False
            self.logger.info("Disconnected from Binance")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from Binance: {str(e)}")
            return False
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data from Binance"""
        if not self.session:
            raise ConnectionError("Not connected to Binance")
        
        symbol = self.normalize_symbol(symbol)
        url = f"{self.base_url}/api/v3/ticker/24hr"
        
        async with self.session.get(url, params={"symbol": symbol}) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "symbol": symbol,
                    "price": float(data["lastPrice"]),
                    "volume": float(data["volume"]),
                    "high": float(data["highPrice"]),
                    "low": float(data["lowPrice"]),
                    "timestamp": datetime.fromtimestamp(data["closeTime"] / 1000)
                }
            else:
                raise Exception(f"Error fetching ticker: {await response.text()}")
    
    async def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get current order book from Binance"""
        if not self.session:
            raise ConnectionError("Not connected to Binance")
        
        symbol = self.normalize_symbol(symbol)
        url = f"{self.base_url}/api/v3/depth"
        
        async with self.session.get(url, params={"symbol": symbol, "limit": depth}) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    "symbol": symbol,
                    "timestamp": datetime.now(),
                    "bids": [[float(price), float(qty)] for price, qty in data["bids"]],
                    "asks": [[float(price), float(qty)] for price, qty in data["asks"]]
                }
            else:
                raise Exception(f"Error fetching orderbook: {await response.text()}")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Get historical market data from Binance"""
        if not self.session:
            raise ConnectionError("Not connected to Binance")
        
        symbol = self.normalize_symbol(symbol)
        url = f"{self.base_url}/api/v3/klines"
        
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(start_time.timestamp() * 1000),
            "limit": 1000
        }
        if end_time:
            params["endTime"] = int(end_time.timestamp() * 1000)
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data, columns=[
                    "timestamp", "open", "high", "low", "close",
                    "volume", "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df = df.set_index("timestamp")
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = df[col].astype(float)
                return df
            else:
                raise Exception(f"Error fetching historical data: {await response.text()}")
    
    async def subscribe_ticker(self, symbol: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time ticker updates from Binance"""
        symbol = self.normalize_symbol(symbol).lower()
        ws = await self._get_websocket()
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@ticker"],
            "id": 1
        }
        await ws.send_str(json.dumps(subscribe_message))
        
        try:
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if "e" in data and data["e"] == "24hrTicker":
                        yield {
                            "symbol": symbol.upper(),
                            "price": float(data["c"]),
                            "volume": float(data["v"]),
                            "timestamp": datetime.fromtimestamp(data["E"] / 1000)
                        }
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
        finally:
            await self._close_websocket()
    
    async def subscribe_orderbook(
        self,
        symbol: str,
        depth: int = 10
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time order book updates from Binance"""
        symbol = self.normalize_symbol(symbol).lower()
        ws = await self._get_websocket()
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [f"{symbol}@depth{depth}@100ms"],
            "id": 2
        }
        await ws.send_str(json.dumps(subscribe_message))
        
        try:
            while True:
                msg = await ws.receive()
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if "e" in data and data["e"] == "depthUpdate":
                        yield {
                            "symbol": symbol.upper(),
                            "timestamp": datetime.fromtimestamp(data["E"] / 1000),
                            "bids": [[float(price), float(qty)] for price, qty in data["b"]],
                            "asks": [[float(price), float(qty)] for price, qty in data["a"]]
                        }
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            self.logger.error(f"WebSocket error: {str(e)}")
        finally:
            await self._close_websocket()
    
    async def _get_websocket(self) -> aiohttp.ClientWebSocketResponse:
        """Get or create WebSocket connection"""
        if not self.ws or self.ws.closed:
            self.ws = await aiohttp.ClientSession().ws_connect(self.ws_url)
        return self.ws
    
    async def _close_websocket(self) -> None:
        """Close WebSocket connection"""
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None 