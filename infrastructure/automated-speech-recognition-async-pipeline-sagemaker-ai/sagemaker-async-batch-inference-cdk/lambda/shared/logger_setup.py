"""
Shared Logging Configuration

This module provides centralized logging setup and configuration for all Lambda functions.
"""

import logging
import os


class LoggerSetup:
    """Handles logging configuration and setup for Lambda functions."""
    
    @staticmethod
    def setup_logging(service_name: str = None) -> logging.Logger:
        """
        Set up comprehensive logging with configurable log levels.
        
        Args:
            service_name (str, optional): Name of the service for logger identification
            
        Returns:
            logging.Logger: Configured logger instance
        """
        log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
        
        # Create logger with service-specific name if provided
        logger_name = service_name if service_name else __name__
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level, logging.INFO))
        
        # Clear any existing handlers to avoid duplicate logs
        logger.handlers.clear()
        
        # Create console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    @staticmethod
    def get_logger(service_name: str) -> logging.Logger:
        """
        Get a logger instance for a specific service.
        
        Args:
            service_name (str): Name of the service
            
        Returns:
            logging.Logger: Logger instance for the service
        """
        return LoggerSetup.setup_logging(service_name)