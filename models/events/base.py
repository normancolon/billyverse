import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from transformers import pipeline

logger = logging.getLogger("billieverse.models.events")

@dataclass
class Event:
    """Container for detected events"""
    timestamp: datetime
    event_type: str  # e.g., 'news', 'geopolitical', 'macro'
    source: str  # Source of the event
    title: str  # Event title/headline
    description: str  # Event description
    entities: Dict[str, List[str]]  # Named entities (e.g., {'ORG': ['Fed', 'ECB'], 'PERSON': ['Powell']})
    sentiment: float  # Sentiment score (-1 to 1)
    impact_score: float  # Estimated impact on markets (0 to 1)
    affected_assets: List[str]  # List of affected assets/markets
    raw_data: Dict  # Raw event data

@dataclass
class EventMetrics:
    """Container for event detection metrics"""
    num_events: int  # Total number of events detected
    event_types: Dict[str, int]  # Count of events by type
    avg_impact_score: float  # Average impact score
    top_entities: Dict[str, List[Tuple[str, int]]]  # Most frequent entities by type
    sentiment_distribution: Dict[str, float]  # Distribution of sentiments

class BaseEventDetector(ABC):
    """Base class for event detection"""
    
    def __init__(
        self,
        ner_model: str = "finbert-ner",
        sentiment_model: str = "finbert-sentiment",
        impact_threshold: float = 0.5,
        max_events: int = 1000,
        cache_dir: Optional[str] = None
    ):
        self.impact_threshold = impact_threshold
        self.max_events = max_events
        self.logger = logger
        
        # Initialize NLP pipelines
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=ner_model,
                aggregation_strategy="simple",
                device=-1  # Use CPU by default
            )
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=sentiment_model,
                device=-1
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP pipelines: {str(e)}")
            raise
        
        # Initialize event storage
        self.events = []
        self.event_metrics = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        try:
            entities = self.ner_pipeline(text)
            
            # Group entities by type
            grouped = {}
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                
                if entity_type not in grouped:
                    grouped[entity_type] = []
                if entity_text not in grouped[entity_type]:
                    grouped[entity_type].append(entity_text)
            
            return grouped
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Convert to score between -1 and 1
            if result['label'] == 'POSITIVE':
                return result['score']
            elif result['label'] == 'NEGATIVE':
                return -result['score']
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            raise
    
    def estimate_impact(
        self,
        event_type: str,
        entities: Dict[str, List[str]],
        sentiment: float
    ) -> float:
        """Estimate market impact of event"""
        try:
            # Base impact factors
            type_weights = {
                'news': 0.5,
                'geopolitical': 0.8,
                'macro': 0.9
            }
            
            entity_weights = {
                'ORG': 0.4,
                'PERSON': 0.3,
                'LOC': 0.2,
                'MISC': 0.1
            }
            
            # Calculate base impact from event type
            impact = type_weights.get(event_type, 0.5)
            
            # Adjust for entities
            entity_impact = 0
            for entity_type, entities_list in entities.items():
                weight = entity_weights.get(entity_type, 0.1)
                entity_impact += weight * len(entities_list)
            impact *= (1 + min(entity_impact, 1))
            
            # Adjust for sentiment strength
            impact *= abs(sentiment)
            
            return min(max(impact, 0), 1)  # Ensure between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error estimating impact: {str(e)}")
            raise
    
    def identify_affected_assets(
        self,
        entities: Dict[str, List[str]],
        text: str
    ) -> List[str]:
        """Identify assets affected by the event"""
        try:
            # Extract company names and tickers
            affected = []
            
            # Add organizations as potential assets
            if 'ORG' in entities:
                affected.extend(entities['ORG'])
            
            # Add any mentioned cryptocurrencies
            crypto_keywords = ['BTC', 'ETH', 'Bitcoin', 'Ethereum']
            for keyword in crypto_keywords:
                if keyword in text:
                    affected.append(keyword)
            
            # Add any mentioned forex pairs
            forex_pairs = ['EUR/USD', 'USD/JPY', 'GBP/USD']
            for pair in forex_pairs:
                if pair in text:
                    affected.append(pair)
            
            return list(set(affected))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Error identifying affected assets: {str(e)}")
            raise
    
    def add_event(self, event: Event) -> None:
        """Add event to storage"""
        try:
            self.events.append(event)
            
            # Remove oldest events if exceeding max_events
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
                
        except Exception as e:
            self.logger.error(f"Error adding event: {str(e)}")
            raise
    
    def calculate_metrics(self) -> EventMetrics:
        """Calculate event detection metrics"""
        try:
            if not self.events:
                return EventMetrics(
                    num_events=0,
                    event_types={},
                    avg_impact_score=0.0,
                    top_entities={},
                    sentiment_distribution={}
                )
            
            # Count events by type
            event_types = {}
            for event in self.events:
                if event.event_type not in event_types:
                    event_types[event.event_type] = 0
                event_types[event.event_type] += 1
            
            # Calculate average impact score
            avg_impact = np.mean([event.impact_score for event in self.events])
            
            # Count entity frequencies
            entity_counts = {}
            for event in self.events:
                for entity_type, entities in event.entities.items():
                    if entity_type not in entity_counts:
                        entity_counts[entity_type] = {}
                    for entity in entities:
                        if entity not in entity_counts[entity_type]:
                            entity_counts[entity_type][entity] = 0
                        entity_counts[entity_type][entity] += 1
            
            # Get top entities
            top_entities = {}
            for entity_type, counts in entity_counts.items():
                sorted_entities = sorted(
                    counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]  # Top 10
                top_entities[entity_type] = sorted_entities
            
            # Calculate sentiment distribution
            sentiments = [event.sentiment for event in self.events]
            sentiment_dist = {
                'positive': np.mean([s > 0.2 for s in sentiments]),
                'neutral': np.mean([abs(s) <= 0.2 for s in sentiments]),
                'negative': np.mean([s < -0.2 for s in sentiments])
            }
            
            self.event_metrics = EventMetrics(
                num_events=len(self.events),
                event_types=event_types,
                avg_impact_score=avg_impact,
                top_entities=top_entities,
                sentiment_distribution=sentiment_dist
            )
            
            return self.event_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    @abstractmethod
    def process_event(self, raw_event: Dict) -> Optional[Event]:
        """Process a raw event and convert it to Event object"""
        pass
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start monitoring for events"""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop monitoring for events"""
        pass 