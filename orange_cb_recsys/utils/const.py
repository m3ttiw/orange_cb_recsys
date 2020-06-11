from pathlib import Path
import logging

home_path = str(Path.home())
DEVELOPING = True

logger = logging.getLogger('logger')

logger.setLevel(logging.INFO)
