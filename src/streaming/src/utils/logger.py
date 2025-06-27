import logging
import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)


class Logger:
    _instances = {}

    def __new__(cls, name: str = "default"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
            cls._instances[name]._setup_logger(name)
        return cls._instances[name]

    def _setup_logger(self, name: str):
        self.logger = logging.getLogger(f"thesis-{name}")
        self.logger.setLevel(logging.INFO)

        # File handler
        format_handler = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(
            f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
        )

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(format_handler)
        self.logger.addHandler(file_handler)

        # Stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(format_handler)
        self.logger.addHandler(stream_handler)

    def info(self, msg, extra=None):
        self.logger.info(msg=msg, extra=extra)

    def error(self, msg, exc_info=False, extra=None):
        self.logger.error(msg=msg, extra=extra, exc_info=exc_info)

    def warning(self, msg, extra=None):
        self.logger.warning(msg=msg, extra=extra)

    def debug(self, msg, extra=None):
        self.logger.debug(msg=msg, extra=extra)
