import logging
from pathlib import Path

# Configure logging
log_dir = Path("working")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "safety_report.log"

logging.basicConfig(
    filename=log_file,
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def log_safety_event(event: str):
    logging.warning(event)
    print(f"[SAFETY LOG] {event}")