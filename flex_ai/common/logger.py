import logging
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Create a custom formatter to add colors
class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    grey = Style.DIM + Fore.WHITE
    yellow = Style.BRIGHT + Fore.YELLOW
    green = Style.BRIGHT + Fore.GREEN
    format = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format,
        logging.INFO: green + format,
        logging.WARNING: yellow + format,
        logging.ERROR: Fore.RED + format,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Function to set up and return a logger
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Create formatter and add it to the handler
        ch.setFormatter(CustomFormatter())

        # Add the handler to the logger
        logger.addHandler(ch)

    return logger