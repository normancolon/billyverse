import redis
import json
from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime
import logging
from infrastructure.config import config

logger = logging.getLogger("billieverse.infrastructure.queues")

class RedisQueue:
    """Redis-based message queue for real-time trading"""
    
    def __init__(
        self,
        queue_name: str,
        host: str = config.redis.host,
        port: int = config.redis.port,
        db: int = config.redis.db,
        password: Optional[str] = config.redis.password,
        ssl: bool = config.redis.ssl
    ):
        self.queue_name = queue_name
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            ssl=ssl,
            decode_responses=True
        )
        self.pubsub = self.redis_client.pubsub()
    
    def push(
        self,
        data: Union[Dict, List],
        priority: int = 1
    ) -> bool:
        """Push data to queue with priority"""
        try:
            # Add timestamp
            message = {
                'data': data,
                'timestamp': datetime.utcnow().isoformat(),
                'priority': priority
            }
            
            # Push to sorted set with priority score
            score = priority * 1000000000 + int(datetime.utcnow().timestamp())
            self.redis_client.zadd(
                self.queue_name,
                {json.dumps(message): score}
            )
            
            # Publish notification
            self.redis_client.publish(
                f"{self.queue_name}_notifications",
                "new_message"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error pushing to queue {self.queue_name}: {str(e)}")
            return False
    
    def pop(
        self,
        timeout: int = 1
    ) -> Optional[Dict]:
        """Pop highest priority message from queue"""
        try:
            # Get highest scored message
            result = self.redis_client.zpopmax(self.queue_name)
            
            if not result:
                return None
            
            # Parse message
            message = json.loads(result[0][0])
            return message
            
        except Exception as e:
            logger.error(f"Error popping from queue {self.queue_name}: {str(e)}")
            return None
    
    async def subscribe(
        self,
        callback: callable,
        batch_size: int = 10,
        interval: float = 0.1
    ):
        """Subscribe to queue and process messages asynchronously"""
        try:
            # Subscribe to notifications
            self.pubsub.subscribe(f"{self.queue_name}_notifications")
            
            while True:
                # Process batch of messages
                messages = []
                for _ in range(batch_size):
                    message = self.pop()
                    if message:
                        messages.append(message)
                    else:
                        break
                
                if messages:
                    # Process batch
                    await callback(messages)
                else:
                    # Wait for notification
                    message = self.pubsub.get_message(
                        timeout=interval
                    )
                    if not message:
                        await asyncio.sleep(interval)
                
        except Exception as e:
            logger.error(
                f"Error in subscription to {self.queue_name}: {str(e)}"
            )
            raise
    
    def clear(self):
        """Clear queue"""
        try:
            self.redis_client.delete(self.queue_name)
            return True
        except Exception as e:
            logger.error(f"Error clearing queue {self.queue_name}: {str(e)}")
            return False

class QueueManager:
    """Manager for multiple Redis queues"""
    
    def __init__(self):
        self.queues: Dict[str, RedisQueue] = {}
        
        # Initialize standard queues
        for queue_name in config.redis.queue_names:
            self.queues[queue_name] = RedisQueue(queue_name)
    
    def get_queue(
        self,
        queue_name: str
    ) -> RedisQueue:
        """Get or create queue"""
        if queue_name not in self.queues:
            self.queues[queue_name] = RedisQueue(queue_name)
        return self.queues[queue_name]
    
    async def start_consumers(
        self,
        handlers: Dict[str, callable]
    ):
        """Start consumers for all queues"""
        tasks = []
        
        for queue_name, handler in handlers.items():
            queue = self.get_queue(queue_name)
            task = asyncio.create_task(
                queue.subscribe(handler)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)
    
    def clear_all(self):
        """Clear all queues"""
        for queue in self.queues.values():
            queue.clear()

# Create global queue manager instance
queue_manager = QueueManager() 