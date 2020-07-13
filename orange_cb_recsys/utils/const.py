from pathlib import Path
import logging

home_path = str(Path.home())
DEVELOPING = False

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('logger')

logger.setLevel(logging.INFO)
