#!/usr/bin/env python
"""
Logging Setup - Unified logging configuration

v4.0 key features:
- Unified Python logging configuration
- Console and file output
- Module-level log level control
"""

import logging
import logging.config
import os
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def setup_logging(config_path: str = None, log_level: int = logging.INFO):
    """
    Unified logging initialization

    Args:
        config_path: Path to logging.yaml config file
        log_level: Default log level (if no config file)
    """
    if config_path and Path(config_path).exists() and HAS_YAML:
        try:
            with open(config_path) as f:
                log_config = yaml.safe_load(f)

            # Dynamically set output path
            handlers = log_config.get('handlers', {})
            if 'file' in handlers:
                log_file = handlers['file'].get('filename')
                if log_file and not os.path.isabs(log_file):
                    # Ensure log directory exists
                    log_dir = Path(log_file).parent
                    log_dir.mkdir(parents=True, exist_ok=True)

            logging.config.dictConfig(log_config)
            return
        except Exception as e:
            print(f"Failed to load logging config from {config_path}: {e}")

    # Default configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get the module logger

    Args:
        name: Module name (typically __name__)

    Returns:
        logging.Logger
    """
    return logging.getLogger(name)
