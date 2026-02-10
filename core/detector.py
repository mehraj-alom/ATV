from ATV.config.config import DetectionConfig
import cv2
import numpy as np
from ATV.core.core_utils import load_model, load_labels

class Detector:
    def __init__(self, config: DetectionConfig , model_path: str, labels_path: str):
        """"
        Initializes the Detector with the given configuration, model path, and labels path. 
        Args:
            config (DetectionConfig): The configuration for the detector.
            model_path (str): The file path to the model.
            labels_path (str): The file path to the labels.
        """
        self.config = config if config else DetectionConfig(
                confidence_threshold=0.5, 
                probability_threshold=0.5,
                nms_threshold=0.4,
                input_width = 640,
                input_height = 640
        )
        self.model = load_model(model_path)
        self.labels = load_labels(labels_path)
    


