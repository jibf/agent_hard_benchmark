import logging


class Logger:
    def __init__(self, name='my_logger', log_file='test.log', level=logging.INFO, console_output=False):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create handlers
        self.file_handler = logging.FileHandler(log_file)
        
        # Configure file handler
        self.file_handler.setLevel(level)
        
        # Create a logging format
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Add the formatter to handlers
        self.file_handler.setFormatter(self.formatter)
        
        # Add handlers to the logger
        if not self.logger.handlers:
            self.logger.addHandler(self.file_handler)
            if console_output:
                self.console_handler = logging.StreamHandler()
                self.console_handler.setLevel(level)
                self.console_handler.setFormatter(self.formatter)
                self.logger.addHandler(self.console_handler)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

# Example usage
if __name__ == "__main__":
    log = Logger(name="test_logger", log_file="logs/test.log", level=logging.DEBUG)
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")