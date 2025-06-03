import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
LOGS_PATH = os.path.join(BASE_DIR, LOGS_DIR)

os.makedirs(LOGS_PATH, exist_ok=True)

LOG_FILE = os.path.join(LOGS_PATH, "mlops_training.log")

logger = logging.getLogger("mlops_logger")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)

def log_warning(message):
    logger.warning(message)
