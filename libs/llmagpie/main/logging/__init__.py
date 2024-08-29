import sys
import logging

# logging.basicConfig(level=logging.INFO)
# class LoggingFormatter(logging.Formatter):
    # def formatTime(self, record, datefmt=None):

LOGGING_FORMAT = '[%(asctime)s.%(msecs)03d] name="%(name)s" level=%(levelname)s filename=%(filename)s lineno=%(lineno)d msg=%(message)s'
Formatter = logging.Formatter(LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(Formatter)

logger = logging.getLogger("llmagpie")
logger.setLevel(logging.INFO)

logger.addHandler(stream_handler)
logger.info("GGG")
