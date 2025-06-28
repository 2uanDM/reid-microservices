import asyncio

from src.core import StreamingService
from src.utils import Logger

logger = Logger(__name__)

if __name__ == "__main__":
    service = StreamingService()
    try:
        asyncio.run(service.run())
    except KeyboardInterrupt:
        logger.info("Shutting down streaming service...")
        service.stop()
