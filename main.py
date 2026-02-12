import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import logger
from utils.global_utils import (
    load_json,
    read_yaml, 
    save_json,
    load_json, 
    get_size, 
    read_video, 
    save_evidence
)
from Trackers.bike_tracker import BikeTracker
from ensure import ensure_annotations

Video_path = "ATV/videos/vid1.mp4"

@ensure_annotations
def main():
    logger.info("Starting ATV application...")

    # Load the global configuration

    

    # Read the Video frames 
    video_frames = read_video(Video_path) 


    # Bike Tracking 
    bike_tracker = BikeTracker(config_path="ATV/config/global_config.yaml")










if __name__ == "__main__":
    logger.info("Starting the ATV application...")
    main()