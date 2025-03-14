import asyncio
import sys
import os

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import init_db, close_db
import logging
from pythonjsonlogger import jsonlogger

# Configure logging
logger = logging.getLogger("billieverse.scripts")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

async def main():
    """Initialize the database"""
    try:
        logger.info("Initializing database...")
        await init_db()
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        await close_db()

if __name__ == "__main__":
    asyncio.run(main()) 