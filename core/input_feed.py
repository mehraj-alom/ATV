import cv2
import numpy as np
import time
from logging import getLogger
from pathlib import Path
logger = getLogger(__name__)



class InputFeed:
    def __init__(self, source : str | int = 0, input_width: int = 640, input_height: int = 640):
        """
        Initializes the InputFeed with the given source, input width, and input height.
        Args:
            source (str | int): The video source (file path or camera index).
            input_width (int): The desired input width for the frames.
            input_height (int): The desired input height for the frames.
        """
        self.source = source
        self.input_width = input_width
        self.input_height = input_height
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video source: {self.source}")
            raise ValueError(f"Failed to open video source: {self.source}")
        logger.info(f"Video source {self.source} opened successfully.")
    
    def get_frame(self):
        """
        Reads a frame from the video source, resizes it to the desired input size, and returns it.
        Returns:
            np.ndarray: The resized frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("No more frames to read or error reading frame.")
            return None
        frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
        return frame_resized
    
    def release(self):
        """
        Releases the video capture object.
        """
        self.cap.release()
        logger.info(f"Video source {self.source} released.")

    


class Mock_input_feed:
    def __init__(self, source : str | int , logger = logger):
        self.source = source
        self.logger = logger
        if isinstance(self.source, str):
            self.source = Path(self.source)
        if not self.source:
            self.logger.error(f"Invalid video source: {self.source}")
            raise ValueError(f"Invalid video source: {self.source}")
        self.logger.info(f"Mock input feed initialized with source: {self.source}")

        self.frames = None

        if self.source:
            self.frames = cv2.VideoCapture(self.source)
            if not self.frames or self.frames.get(cv2.CAP_PROP_FRAME_COUNT) == 0:
                self.logger.error(f"Failed to open video source: {self.source}")
                raise ValueError(f"Failed to open video source: {self.source}")
            self.logger.info(f"Video source {self.source} opened successfully.")
        else:
            self.logger.error(f"Video source {self.source} is empty.")
            raise ValueError(f"Video source {self.source} is empty.")
        


    def get_frame(self):
        if self.frames is None:
            self.logger.error("Video source is not initialized.")
            raise ValueError("Video source is not initialized.")
        
        ret, frame = self.frames.read()
        if not ret:
            self.logger.warning("No more frames to read or error reading frame.")
            return None
        return frame 


    def release(self):
        if self.frames is not None:
            self.frames.release()
            self.logger.info(f"Video source {self.source} released.")
                

        
