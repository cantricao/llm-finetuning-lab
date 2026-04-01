import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standard logger for the enterprise application.
    
    Args:
        name (str): The name of the module requesting the logger.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        
        # Create console handler with formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
    return logger
