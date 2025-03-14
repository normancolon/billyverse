import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import logging
from models.events.base import BaseEventDetector, Event

logger = logging.getLogger("billieverse.models.events.news")

class NewsEventDetector(BaseEventDetector):
    """Real-time financial news event detector"""
    
    def __init__(
        self,
        api_keys: Dict[str, str],
        update_interval: int = 60,  # seconds
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.api_keys = api_keys
        self.update_interval = update_interval
        self.monitoring = False
        self.seen_articles: Set[str] = set()  # Track processed articles
        self.monitoring_task = None
    
    async def fetch_news_api(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Fetch news from NewsAPI"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'apiKey': self.api_keys['newsapi'],
                'q': '(crypto OR cryptocurrency OR bitcoin OR ethereum OR finance OR market OR stock)',
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 100
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('articles', [])
                else:
                    self.logger.error(f"NewsAPI error: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error fetching from NewsAPI: {str(e)}")
            return []
    
    async def fetch_alpha_vantage(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Fetch news from Alpha Vantage"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'NEWS_SENTIMENT',
                'apikey': self.api_keys['alphavantage'],
                'topics': 'blockchain,earnings,ipo,economy_macro,forex,merger_and_acquisition',
                'sort': 'LATEST',
                'limit': 100
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('feed', [])
                else:
                    self.logger.error(f"Alpha Vantage error: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error fetching from Alpha Vantage: {str(e)}")
            return []
    
    def process_event(self, raw_event: Dict) -> Optional[Event]:
        """Process a raw news event"""
        try:
            # Extract basic information
            source = raw_event.get('source', {}).get('name', 'Unknown')
            title = raw_event.get('title', '')
            description = raw_event.get('description', '')
            
            # Skip if already processed
            article_id = f"{source}_{title}"
            if article_id in self.seen_articles:
                return None
            self.seen_articles.add(article_id)
            
            # Combine title and description for analysis
            full_text = f"{title} {description}"
            
            # Extract entities
            entities = self.extract_entities(full_text)
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(full_text)
            
            # Identify affected assets
            affected_assets = self.identify_affected_assets(entities, full_text)
            
            # Determine event type
            event_type = self.classify_event_type(entities, full_text)
            
            # Estimate impact
            impact_score = self.estimate_impact(event_type, entities, sentiment)
            
            # Create event if impact is significant
            if impact_score >= self.impact_threshold:
                return Event(
                    timestamp=datetime.now(),
                    event_type=event_type,
                    source=source,
                    title=title,
                    description=description,
                    entities=entities,
                    sentiment=sentiment,
                    impact_score=impact_score,
                    affected_assets=affected_assets,
                    raw_data=raw_event
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing event: {str(e)}")
            return None
    
    def classify_event_type(
        self,
        entities: Dict[str, List[str]],
        text: str
    ) -> str:
        """Classify event type based on content"""
        try:
            # Keywords for classification
            geopolitical_keywords = [
                'war', 'conflict', 'sanctions', 'election',
                'government', 'policy', 'regulation'
            ]
            
            macro_keywords = [
                'gdp', 'inflation', 'interest rate', 'fed',
                'economy', 'recession', 'employment'
            ]
            
            # Check for geopolitical events
            if any(keyword in text.lower() for keyword in geopolitical_keywords):
                return 'geopolitical'
            
            # Check for macroeconomic events
            if any(keyword in text.lower() for keyword in macro_keywords):
                return 'macro'
            
            # Default to news
            return 'news'
            
        except Exception as e:
            self.logger.error(f"Error classifying event type: {str(e)}")
            return 'news'
    
    async def monitor_events(self) -> None:
        """Monitor news sources for events"""
        try:
            async with aiohttp.ClientSession() as session:
                while self.monitoring:
                    # Fetch from multiple sources
                    news_api_articles = await self.fetch_news_api(session)
                    alpha_vantage_articles = await self.fetch_alpha_vantage(session)
                    
                    # Process events
                    for article in news_api_articles + alpha_vantage_articles:
                        event = self.process_event(article)
                        if event:
                            self.add_event(event)
                            self.logger.info(
                                f"New event detected: {event.title} "
                                f"(Impact: {event.impact_score:.2f})"
                            )
                    
                    # Update metrics
                    self.calculate_metrics()
                    
                    # Wait for next update
                    await asyncio.sleep(self.update_interval)
                    
        except Exception as e:
            self.logger.error(f"Error in event monitoring: {str(e)}")
            self.monitoring = False
    
    def start_monitoring(self) -> None:
        """Start monitoring for events"""
        try:
            if not self.monitoring:
                self.monitoring = True
                loop = asyncio.get_event_loop()
                self.monitoring_task = loop.create_task(self.monitor_events())
                self.logger.info("Started event monitoring")
                
        except Exception as e:
            self.logger.error(f"Error starting monitoring: {str(e)}")
            raise
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for events"""
        try:
            if self.monitoring:
                self.monitoring = False
                if self.monitoring_task:
                    self.monitoring_task.cancel()
                self.logger.info("Stopped event monitoring")
                
        except Exception as e:
            self.logger.error(f"Error stopping monitoring: {str(e)}")
            raise
    
    def get_recent_events(
        self,
        minutes: int = 60,
        event_type: Optional[str] = None
    ) -> List[Event]:
        """Get recent events within specified time window"""
        try:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            events = [
                event for event in self.events
                if event.timestamp >= cutoff and
                (event_type is None or event.event_type == event_type)
            ]
            return sorted(events, key=lambda x: x.impact_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting recent events: {str(e)}")
            return []
    
    def get_event_summary(self) -> pd.DataFrame:
        """Get summary of recent events"""
        try:
            if not self.events:
                return pd.DataFrame()
            
            # Convert events to DataFrame
            data = []
            for event in self.events:
                data.append({
                    'timestamp': event.timestamp,
                    'type': event.event_type,
                    'source': event.source,
                    'title': event.title,
                    'sentiment': event.sentiment,
                    'impact_score': event.impact_score,
                    'affected_assets': ', '.join(event.affected_assets)
                })
            
            df = pd.DataFrame(data)
            return df.sort_values('timestamp', ascending=False)
            
        except Exception as e:
            self.logger.error(f"Error creating event summary: {str(e)}")
            return pd.DataFrame() 