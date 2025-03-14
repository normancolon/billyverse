from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
import pandas as pd
from datetime import datetime
import logging
from core.config import settings

logger = logging.getLogger("billieverse.market")

class MarketDataProvider(ABC):
    """Base class for market data providers"""
    
    def __init__(self, trading_pairs: List[str]):
        self.trading_pairs = trading_pairs
        self.is_connected = False
        self.logger = logger
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the market data provider"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the market data provider"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker data for a symbol"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get current order book for a symbol"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """Get historical market data"""
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time ticker updates"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(
        self,
        symbol: str,
        depth: int = 10
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time order book updates"""
        pass
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol format for the specific provider"""
        return symbol.upper().replace("/", "")
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported by the provider"""
        return symbol in self.trading_pairs 