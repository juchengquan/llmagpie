import os
import pytz
import logging
import datetime
import pathlib
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, cast


class CustomFormatter(logging.Formatter):
    """
    Class to format log record timestamp to Singapore time and utc format.
    """
    def converter(self, timestamp: int):
        """Method to convert epoch time to Singapore timezone.

        Args:
            timestamp (int): epoch time.

        Returns:
            datetime: datetime formatted to Singapore timezone.
        """
        dt = datetime.datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        return dt.astimezone(pytz.timezone('Singapore'))

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None):
        """Method to format log record timestamp to provided format if specified else iso format.

        Args:
            record (logging.LogRecord): log record.
            datefmt (str, optional): date format. Defaults to None.

        Returns:
            str: formatted timestamp.
        """
        dt = self.converter( cast(int, record.created) )
        if datefmt:
            formatted_timestamp = dt.strftime(datefmt)
        else:
            try:
                formatted_timestamp = dt.isoformat(timespec='milliseconds')
            except TypeError:
                formatted_timestamp = dt.isoformat()
        return formatted_timestamp


def DefaultLogger(name: str, handlers: list = [], *args, **kwargs):
    """
    Function to initialize CustomLogger object. Calls logging.getLogger underneath the hood.

    Args:
        name (str): logger name.

    Returns:
        CustomLogger: custom logger object.
    """
    flag = name in logging.root.manager.loggerDict
    logger = logging.getLogger(name, *args, **kwargs)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        for handler in handlers:
            logger.addHandler(handler)
    if not flag:
        logger.debug(f"Logger initialized: {name}")

    return logger


LOGGING_FORMAT = '[%(asctime)s] name="%(name)s" level=%(levelname)s filename="%(filename)s" lineno=%(lineno)d msg="%(message)s"'
Formatter = CustomFormatter(LOGGING_FORMAT, datefmt="%Y-%m-%d %H:%M:%S,%f")


def get_or_create_logger(logger_name: str = "default", file_path: Optional[str] = None):
    """Get the logger with name or create if it does not exist.
    """
    if not file_path:
        log_path = os.getenv("LOG_DIR", pathlib.Path().resolve().parent / "logs")
        os.makedirs(log_path, exist_ok=True)
        # file_path = "/home/cdsw/logs/info.log"
        file_path = f'{log_path}/info.log'
    # Setup streamhandler which outputs to console
    streamhandler = logging.StreamHandler()
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(Formatter)

    filehandler = TimedRotatingFileHandler(
        filename=file_path,
        interval=1,
        when="d"
    )
    filehandler.setLevel(logging.INFO)
    filehandler.setFormatter(Formatter)

    logger = DefaultLogger(name=logger_name, handlers=[streamhandler, filehandler])

    return logger
