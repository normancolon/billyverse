from typing import Dict, List, Optional, Callable
import json
import asyncio
from datetime import datetime
import logging
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from infrastructure.config import config

logger = logging.getLogger("billieverse.infrastructure.streaming")

class KafkaStream:
    """Kafka-based market data streaming"""
    
    def __init__(
        self,
        topic: str,
        bootstrap_servers: List[str] = config.kafka.bootstrap_servers,
        client_id: Optional[str] = None
    ):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id or f"client_{topic}"
        
        # Initialize clients
        self.producer = None
        self.consumer = None
        
        # Track statistics
        self.message_count = 0
        self.last_timestamp = None
    
    async def start_producer(self):
        """Start Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=f"{self.client_id}_producer",
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self.producer.start()
            logger.info(f"Started Kafka producer for topic {self.topic}")
            
        except Exception as e:
            logger.error(f"Error starting Kafka producer: {str(e)}")
            raise
    
    async def start_consumer(
        self,
        group_id: str = config.kafka.consumer_group,
        auto_offset_reset: str = config.kafka.auto_offset_reset
    ):
        """Start Kafka consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                client_id=f"{self.client_id}_consumer",
                auto_offset_reset=auto_offset_reset,
                value_deserializer=lambda v: json.loads(v.decode('utf-8'))
            )
            await self.consumer.start()
            logger.info(f"Started Kafka consumer for topic {self.topic}")
            
        except Exception as e:
            logger.error(f"Error starting Kafka consumer: {str(e)}")
            raise
    
    async def publish(
        self,
        message: Dict,
        partition: Optional[int] = None
    ):
        """Publish message to Kafka topic"""
        try:
            if not self.producer:
                await self.start_producer()
            
            # Add timestamp
            message['timestamp'] = datetime.utcnow().isoformat()
            
            # Send message
            await self.producer.send_and_wait(
                self.topic,
                message,
                partition=partition
            )
            
            # Update statistics
            self.message_count += 1
            self.last_timestamp = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error publishing to Kafka: {str(e)}")
            raise
    
    async def consume(
        self,
        handler: Callable,
        batch_size: int = 100,
        timeout: float = 1.0
    ):
        """Consume messages from Kafka topic"""
        try:
            if not self.consumer:
                await self.start_consumer()
            
            while True:
                messages = []
                async for message in self.consumer:
                    messages.append(message.value)
                    if len(messages) >= batch_size:
                        break
                
                if messages:
                    await handler(messages)
                else:
                    await asyncio.sleep(timeout)
                
        except Exception as e:
            logger.error(f"Error consuming from Kafka: {str(e)}")
            raise
    
    async def close(self):
        """Close Kafka connections"""
        try:
            if self.producer:
                await self.producer.stop()
            if self.consumer:
                await self.consumer.stop()
            
        except Exception as e:
            logger.error(f"Error closing Kafka connections: {str(e)}")
            raise

class StreamManager:
    """Manager for multiple Kafka streams"""
    
    def __init__(self):
        self.streams: Dict[str, KafkaStream] = {}
        
        # Initialize streams for each topic
        for topic in config.kafka.topics:
            self.streams[topic] = KafkaStream(topic)
    
    def get_stream(
        self,
        topic: str
    ) -> KafkaStream:
        """Get or create stream"""
        if topic not in self.streams:
            self.streams[topic] = KafkaStream(topic)
        return self.streams[topic]
    
    async def start_producers(
        self,
        topics: Optional[List[str]] = None
    ):
        """Start producers for specified topics"""
        topics = topics or list(self.streams.keys())
        
        for topic in topics:
            stream = self.get_stream(topic)
            await stream.start_producer()
    
    async def start_consumers(
        self,
        handlers: Dict[str, Callable]
    ):
        """Start consumers with handlers"""
        tasks = []
        
        for topic, handler in handlers.items():
            stream = self.get_stream(topic)
            task = asyncio.create_task(
                stream.consume(handler)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    async def close_all(self):
        """Close all streams"""
        for stream in self.streams.values():
            await stream.close()

# Create global stream manager instance
stream_manager = StreamManager() 