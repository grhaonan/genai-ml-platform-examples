import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("ArchitectureWorkflow")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    '[%(levelname)s] %(asctime)s - %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)