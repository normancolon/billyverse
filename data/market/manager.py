from typing import Dict, List, Any, Optional, Type
import asyncio
from datetime import datetime
import pandas as pd
from data.market.base import MarketDataProvider
from data.market.providers.binance import BinanceProvider
from data.processing.feature_engineering import MarketDataProcessor
import logging
from core.config import settings

logger = logging.getLogger("billieverse.market")

class MarketDataManager:
    """Manages market data providers and coordinates data processing"""
    
    def __init__(self):
        self.providers: Dict[str, MarketDataProvider] = {}
        self.processor = MarketDataProcessor()
        self.logger = logger
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize market data providers"""
        trading_pairs = settings.trading.TRADING_PAIRS
        
        # Initialize Binance provider
        if settings.api_keys.BINANCE_API_KEY and settings.api_keys.BINANCE_SECRET_KEY:
            self.providers['binance'] = BinanceProvider(trading_pairs)
    
    async def connect_all(self) -> bool:
        """Connect to all configured providers"""
        try:
            results = await asyncio.gather(
                *[provider.connect() for provider in self.providers.values()],
                return_exceptions=True
            )
            return all(isinstance(r, bool) and r for r in results)
        except Exception as e:
            self.logger.error(f"Error connecting to providers: {str(e)}")
            return False
    
    async def disconnect_all(self) -> bool:
        """Disconnect from all providers"""
        try:
            results = await asyncio.gather(
                *[provider.disconnect() for provider in self.providers.values()],
                return_exceptions=True
            )
            return all(isinstance(r, bool) and r for r in results)
        except Exception as e:
            self.logger.error(f"Error disconnecting from providers: {str(e)}")
            return False
    
    async def get_ticker_all_providers(
        self,
        symbol: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get ticker data from all providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                ticker = await provider.get_ticker(symbol)
                results[name] = ticker
            except Exception as e:
                self.logger.error(f"Error getting ticker from {name}: {str(e)}")
        return results
    
    async def get_orderbook_all_providers(
        self,
        symbol: str,
        depth: int = 10
    ) -> Dict[str, Dict[str, Any]]:
        """Get order book data from all providers"""
        results = {}
        for name, provider in self.providers.items():
            try:
                orderbook = await provider.get_orderbook(symbol, depth)
                processed = self.processor.process_orderbook(orderbook)
                results[name] = processed
            except Exception as e:
                self.logger.error(f"Error getting orderbook from {name}: {str(e)}")
        return results
    
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        interval: str = "1m",
        provider: str = "binance"
    ) -> Optional[pd.DataFrame]:
        """Get and process historical market data"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not configured")
            
            # Get raw data
            df = await self.providers[provider].get_historical_data(
                symbol, start_time, end_time, interval
            )
            
            # Process data
            df = self.processor.process_ohlcv(df)
            df = self.processor.normalize_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    async def stream_market_data(
        self,
        symbol: str,
        provider: str = "binance"
    ):
        """Stream real-time market data"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not configured")
            
            provider_instance = self.providers[provider]
            
            # Start ticker and orderbook streams
            ticker_stream = provider_instance.subscribe_ticker(symbol)
            orderbook_stream = provider_instance.subscribe_orderbook(symbol)
            
            # Process streams
            async def process_ticker():
                async for ticker in ticker_stream:
                    yield {
                        "type": "ticker",
                        "provider": provider,
                        "data": ticker
                    }
            
            async def process_orderbook():
                async for orderbook in orderbook_stream:
                    processed = self.processor.process_orderbook(orderbook)
                    yield {
                        "type": "orderbook",
                        "provider": provider,
                        "data": processed
                    }
            
            # Merge streams
            async def merge_streams():
                ticker_task = asyncio.create_task(process_ticker())
                orderbook_task = asyncio.create_task(process_orderbook())
                
                try:
                    while True:
                        done, pending = await asyncio.wait(
                            [ticker_task, orderbook_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        for task in done:
                            if task == ticker_task:
                                ticker_data = await task
                                yield ticker_data
                                ticker_task = asyncio.create_task(process_ticker())
                            elif task == orderbook_task:
                                orderbook_data = await task
                                yield orderbook_data
                                orderbook_task = asyncio.create_task(process_orderbook())
                
                except Exception as e:
                    self.logger.error(f"Error in stream processing: {str(e)}")
                finally:
                    for task in [ticker_task, orderbook_task]:
                        if not task.done():
                            task.cancel()
            
            return merge_streams()
            
        except Exception as e:
            self.logger.error(f"Error setting up market data stream: {str(e)}")
            raise
    
    def calculate_market_impact(
        self,
        orderbook: Dict[str, Any],
        trade_size: float
    ) -> Dict[str, float]:
        """Calculate market impact for a trade"""
        return self.processor.calculate_market_impact(orderbook, trade_size) 