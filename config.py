import logging
import os


CONFIG = {
    'LOG_LEVEL': os.getenv('LOG_LEVEL', logging.DEBUG),
    'PORT': int(os.getenv('PORT', 8000)),
    'HOST': os.getenv('HOST', "0.0.0.0"),
    'MAX_PLAYERS': int(os.getenv('MAX_PLAYERS', 50)),
    'TICK_RATE_MS': int(os.getenv('TICK_RATE_MS', 100)),
    'GRID_SIZE': 256,
    'ADMIN_NAME': os.getenv('ADMIN_NAME', "Tony"),
}

# Configuration du logging
logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s'
)
logger = logging.getLogger("gameoflife")