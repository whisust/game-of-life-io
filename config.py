import logging
import os


CONFIG = {
    'LOG_LEVEL': os.getenv('LOG_LEVEL', logging.DEBUG),
    'PORT': os.getenv('PORT', 8000),
    'MAX_PLAYERS': int(os.getenv('MAX_PLAYERS', 50)),
    'TICK_RATE_MS': int(os.getenv('TICK_RATE_MS', 250)),
    'GRID_SIZE': 256
}

# Configuration du logging
logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
logger = logging.getLogger("gameoflife")