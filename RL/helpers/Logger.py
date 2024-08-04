import logging
import os
from logging import handlers
from sys import stdout


class Logger:
    LOGGING_DEFAULT_LOGGER_NAME = "5GNR-FLEX"
    LOGS_FOLDER = "../logs"
    DEFAULT_LOG_LEVEL = 'DEBUG'

    def __init__(
            self,
            caller=None,
            file_name=None,
            print_logs=True,
            write_logs=True,
            level=DEFAULT_LOG_LEVEL
    ):
        log_level = level if level in [
            'CRITICAL', 'ERROR', 'WARN', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
        ] else Logger.DEFAULT_LOG_LEVEL

        self.__name = Logger.LOGGING_DEFAULT_LOGGER_NAME + \
            ("" if caller is None else ("." + caller))
        self.__file = "{}/{}".format(
            Logger.LOGS_FOLDER,
            (file_name if file_name is not None else self.__name).replace(
                ".log",
                "") + ".log")
        # The formatter
        log_format = '%(asctime)s %(levelname)s %(name)s %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%b %d %H:%M:%S')
        # Create and clear the logger:
        self.__logger = logging.getLogger(self.__name)
        self.__logger.setLevel(log_level)

        self.__logger.propagate = False
        # Clear all handlers:
        while len(self.__logger.handlers) > 0:
            h = self.__logger.handlers[0]
            self.__logger.removeHandler(h)
        # Set level
        # Add handlers accordingly
        if print_logs:
            # Set Handler:
            self.__console_handler = logging.StreamHandler(stdout)
            self.__logger.addHandler(self.__console_handler)
            # Set the formatter
            self.__console_handler.setFormatter(formatter)
        if write_logs:
            # Create Logging folder if it doesn't exist:
            if not os.path.exists(os.path.dirname(self.__file)):
                os.makedirs(os.path.dirname(self.__file))
            # Set the Handler:
            self.__file_handler = handlers.RotatingFileHandler(
                self.__file, maxBytes=10 * 1024 * 1024, backupCount=5)
            self.__logger.addHandler(self.__file_handler)
            # Set the formatter
            self.__file_handler.setFormatter(formatter)

    @property
    def logger(self):
        return self.__logger
