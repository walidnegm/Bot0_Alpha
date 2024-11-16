import logging
import logging_config  # Automatically configures logging

# Setup logger
logger = logging.getLogger(__name__)
logger.info("Logging has been configured.")
logger.debug("This is a debug message.")
