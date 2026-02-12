from typing import List
from ATV.config.config import DetectionConfig
import cv2
import numpy as np
from ATV.core.core_models import Detection
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
    
    def detect(self, frame) : 
        """
        Perform object detection on the input frame and return a list of Detection objects.
        Args:
            frame (np.ndarray): The input frame for detection.
        Returns:
            List[Detection]: A list of Detection objects containing bounding boxes, class IDs, class names, confidence scores, and tracker IDs.
        """ 
        pass



    def detect_to_objects(self, frame) -> List[Detection]:
        """
        Convert supervision.Detections to List[Detection] objects
        Args:
            frame (np.ndarray): The input frame for detection.
        Returns:
            List[Detection]: A list of Detection objects containing bounding boxes, class IDs, class names, confidence scores, and tracker IDs.
        """
        sv_detections = self.model(frame)
        
        detections = []
        for i in range(len(sv_detections)):
            det = Detection(
                bbox=tuple(sv_detections.xyxy[i].astype(int)),
                class_id=int(sv_detections.class_id[i]),
                class_name=self.labels[sv_detections.class_id[i]],
                confidence=float(sv_detections.confidence[i]),
                tracker_id=None
            )
            detections.append(det)
        
        return detections    
    


