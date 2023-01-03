"""Module for logger configuration."""
import logging
import logging.handlers

LOG_FORMAT = "%(asctime)s --> %(filename)-18s %(levelname)-8s %(message)s"

logger = logging.getLogger("COVID-19 Classifier using X-ray Images")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(LOG_FORMAT)

# Handler for logging to console.
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.set_name("console_handler")
logger.addHandler(console_handler)
