import pandas as pd
import praw
import tweepy
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import asyncio
import aiohttp
from core.config import settings

logger = logging.getLogger("billieverse.data.scraping")

class MarketDataCollector:
    """Collector for financial news and social media data"""
    
    def __init__(
        self,
        twitter_api_key: Optional[str] = None,
        twitter_api_secret: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        news_api_key: Optional[str] = None
    ):
        self.logger = logger
        
        # Initialize API clients
        self.twitter_client = self._init_twitter(twitter_api_key, twitter_api_secret)
        self.reddit_client = self._init_reddit(reddit_client_id, reddit_client_secret)
        self.news_api_key = news_api_key
        
        # Initialize async session
        self.session = None
    
    async def __aenter__(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close async session"""
        if self.session:
            await self.session.close()
    
    def _init_twitter(
        self,
        api_key: Optional[str],
        api_secret: Optional[str]
    ) -> Optional[tweepy.Client]:
        """Initialize Twitter client"""
        try:
            if api_key and api_secret:
                auth = tweepy.OAuthHandler(api_key, api_secret)
                return tweepy.API(auth)
            return None
        except Exception as e:
            self.logger.error(f"Error initializing Twitter client: {str(e)}")
            return None
    
    def _init_reddit(
        self,
        client_id: Optional[str],
        client_secret: Optional[str]
    ) -> Optional[praw.Reddit]:
        """Initialize Reddit client"""
        try:
            if client_id and client_secret:
                return praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent="BillieVerse/1.0"
                )
            return None
        except Exception as e:
            self.logger.error(f"Error initializing Reddit client: {str(e)}")
            return None
    
    async def get_financial_news(
        self,
        keywords: List[str],
        days: int = 1
    ) -> List[Dict[str, Any]]:
        """Get financial news articles"""
        try:
            if not self.news_api_key:
                raise ValueError("News API key not provided")
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Build query
            query = ' OR '.join(keywords)
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"from={from_date}&"
                f"to={to_date}&"
                f"language=en&"
                f"sortBy=publishedAt"
            )
            
            headers = {'Authorization': f'Bearer {self.news_api_key}'}
            
            async with self.session.get(url, headers=headers) as response:
                data = await response.json()
                
                if data.get('status') != 'ok':
                    raise ValueError(f"API Error: {data.get('message')}")
                
                articles = []
                for article in data.get('articles', []):
                    articles.append({
                        'source': article.get('source', {}).get('name'),
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'content': article.get('content'),
                        'url': article.get('url'),
                        'published_at': article.get('publishedAt')
                    })
                
                return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching news: {str(e)}")
            return []
    
    async def get_twitter_sentiment(
        self,
        keywords: List[str],
        max_tweets: int = 100
    ) -> List[Dict[str, Any]]:
        """Get Twitter posts"""
        try:
            if not self.twitter_client:
                raise ValueError("Twitter client not initialized")
            
            tweets = []
            query = ' OR '.join(keywords)
            
            # Search tweets
            cursor = tweepy.Cursor(
                self.twitter_client.search_tweets,
                q=query,
                lang='en',
                tweet_mode='extended'
            ).items(max_tweets)
            
            for tweet in cursor:
                tweets.append({
                    'text': tweet.full_text,
                    'created_at': tweet.created_at.isoformat(),
                    'user': tweet.user.screen_name,
                    'retweets': tweet.retweet_count,
                    'likes': tweet.favorite_count
                })
            
            return tweets
            
        except Exception as e:
            self.logger.error(f"Error fetching tweets: {str(e)}")
            return []
    
    async def get_reddit_sentiment(
        self,
        subreddits: List[str],
        keywords: List[str],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get Reddit posts and comments"""
        try:
            if not self.reddit_client:
                raise ValueError("Reddit client not initialized")
            
            posts = []
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search posts
                for keyword in keywords:
                    for post in subreddit.search(
                        keyword,
                        limit=limit,
                        sort='new'
                    ):
                        posts.append({
                            'title': post.title,
                            'text': post.selftext,
                            'created_at': datetime.fromtimestamp(post.created_utc).isoformat(),
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'url': f"https://reddit.com{post.permalink}"
                        })
            
            return posts
            
        except Exception as e:
            self.logger.error(f"Error fetching Reddit posts: {str(e)}")
            return []
    
    async def collect_all_sources(
        self,
        keywords: List[str],
        subreddits: List[str],
        days: int = 1,
        max_tweets: int = 100,
        reddit_limit: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from all sources"""
        try:
            # Collect data concurrently
            news_task = self.get_financial_news(keywords, days)
            twitter_task = self.get_twitter_sentiment(keywords, max_tweets)
            reddit_task = self.get_reddit_sentiment(subreddits, keywords, reddit_limit)
            
            news, tweets, reddit_posts = await asyncio.gather(
                news_task,
                twitter_task,
                reddit_task
            )
            
            return {
                'news': news,
                'tweets': tweets,
                'reddit': reddit_posts
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting data: {str(e)}")
            return {
                'news': [],
                'tweets': [],
                'reddit': []
            } 