from ultralytics import YOLO
from ATV.utils.global_utils import read_yaml
from pathlib import Path
import numpy as np
import cv2
import supervision as sv
from ATV.logger import logger


class BikeTracker:
    def __init__(self, config_path: str | Path):
        """Initializes the BikeTracker with the given configuration path.
        
        Args:
            config_path (str): The file path to the configuration YAML file.
        """
        self.config = read_yaml(Path(config_path))
        self.model = YOLO(self.config["models"]["Detector_model_path"])
        self.tracker = sv.ByteTrack()
        logger.info("BikeTracker initialized successfully with the provided configuration.")

    def detect_bikes(self, frame: np.ndarray):
        """Detects and tracks bikes in the given frame.
        
        Args:
            frame (np.ndarray) = Matlike: The input video frame as a NumPy array."""
        
        batch_size = 20
        detections = []
        
        
        for i in range(0, len(frame), batch_size):
            batch_frames = frame[i:i+batch_size]
            results = self.model.predict(batch_frames)
            detections += results
            
        return detections
    

    def track_bikes(self, video_frames):
        """Tracks bikes across the given video frames.
        
        Args:
            video_frames (list): A list of video frames to process for bike tracking.
        """
        detections = self.detect_bikes(video_frames)
        tracks = []

        for frame_num , detection in enumerate(detections):
            class_names = detection.names
            class_name_inv = {v : k  for k, v in class_names.items()}

            supervision_detections = sv.Detections.from_ultralytics(detection)
            detection_with_tracking = self.tracker.update_with_detections(supervision_detections)
             
            
            tracks.append({})
            
            
            for track in detection_with_tracking:
                bbox = track[0].tolist()
                class_id = track[3]
                track_id = track[4]
            
                if class_id == class_name_inv["bike"]:
                    tracks[frame_num][track_id] = {
                        "bbox": bbox
                    }
        return tracks
        