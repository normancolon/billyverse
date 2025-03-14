import asyncio
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json
from typing import Dict, List
import os
from dotenv import load_dotenv

from models.events.base import Event, EventMetrics
from models.events.news import NewsEventDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("billieverse.examples.events")

def plot_event_metrics(
    metrics: EventMetrics,
    save_dir: Path
) -> None:
    """Plot event detection metrics"""
    try:
        plots_dir = save_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Plot event type distribution
        plt.figure(figsize=(10, 6))
        event_types = pd.Series(metrics.event_types)
        event_types.plot(kind='bar')
        plt.title("Event Type Distribution")
        plt.xlabel("Event Type")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(plots_dir / "event_types.png")
        plt.close()
        
        # Plot sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiments = pd.Series(metrics.sentiment_distribution)
        sentiments.plot(kind='pie', autopct='%1.1f%%')
        plt.title("Sentiment Distribution")
        plt.tight_layout()
        plt.savefig(plots_dir / "sentiment_distribution.png")
        plt.close()
        
        # Plot top entities
        plt.figure(figsize=(12, 6))
        for entity_type, entities in metrics.top_entities.items():
            if entities:  # Check if there are entities of this type
                entity_df = pd.DataFrame(entities, columns=['Entity', 'Count'])
                plt.figure(figsize=(10, 6))
                sns.barplot(data=entity_df, x='Count', y='Entity')
                plt.title(f"Top {entity_type} Entities")
                plt.tight_layout()
                plt.savefig(plots_dir / f"top_{entity_type.lower()}_entities.png")
                plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting metrics: {str(e)}")
        raise

def save_events(
    events: List[Event],
    save_dir: Path
) -> None:
    """Save events to JSON file"""
    try:
        # Convert events to dictionaries
        event_dicts = []
        for event in events:
            event_dict = {
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type,
                'source': event.source,
                'title': event.title,
                'description': event.description,
                'entities': event.entities,
                'sentiment': event.sentiment,
                'impact_score': event.impact_score,
                'affected_assets': event.affected_assets
            }
            event_dicts.append(event_dict)
        
        # Save to file
        with open(save_dir / "events.json", "w") as f:
            json.dump(event_dicts, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving events: {str(e)}")
        raise

async def main():
    try:
        # Set up directories
        results_dir = Path("results/event_detection")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get API keys from environment
        api_keys = {
            'newsapi': os.getenv('NEWS_API_KEY'),
            'alphavantage': os.getenv('ALPHA_VANTAGE_KEY')
        }
        
        if not all(api_keys.values()):
            raise ValueError("Missing required API keys in environment variables")
        
        # Initialize event detector
        detector = NewsEventDetector(
            api_keys=api_keys,
            update_interval=60,  # 1 minute
            impact_threshold=0.5,
            ner_model="ProsusAI/finbert",
            sentiment_model="ProsusAI/finbert"
        )
        
        logger.info("Starting event monitoring...")
        
        try:
            # Start monitoring
            detector.start_monitoring()
            
            # Run for 1 hour
            await asyncio.sleep(3600)
            
        finally:
            # Stop monitoring
            detector.stop_monitoring()
        
        # Get event summary
        summary_df = detector.get_event_summary()
        if not summary_df.empty:
            summary_df.to_csv(results_dir / "event_summary.csv", index=False)
        
        # Get recent high-impact events
        recent_events = detector.get_recent_events(minutes=60)
        if recent_events:
            save_events(recent_events, results_dir)
        
        # Calculate and plot metrics
        metrics = detector.calculate_metrics()
        if metrics.num_events > 0:
            plot_event_metrics(metrics, results_dir)
            
            # Save metrics summary
            with open(results_dir / "metrics_summary.txt", "w") as f:
                f.write(f"Event Detection Metrics Summary\n")
                f.write(f"{'='*30}\n\n")
                f.write(f"Total Events: {metrics.num_events}\n")
                f.write(f"Average Impact Score: {metrics.avg_impact_score:.2f}\n\n")
                
                f.write("Event Type Distribution:\n")
                for event_type, count in metrics.event_types.items():
                    f.write(f"- {event_type}: {count}\n")
                
                f.write("\nSentiment Distribution:\n")
                for sentiment, ratio in metrics.sentiment_distribution.items():
                    f.write(f"- {sentiment}: {ratio:.1%}\n")
        
        logger.info("Event detection completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in event detection: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 