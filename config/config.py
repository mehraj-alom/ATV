from pathlib import Path
from typing import Dict, Any
import yaml
from dataclasses import dataclass


class DetectionConfig:
    def __init__(self, confidence_threshold=0.5, 
                 probability_threshold=0.5,
                 nms_threshold=0.4,
                 input_width = 640,
                input_height = 640):
        self.confidence_threshold = confidence_threshold
        self.probability_threshold = probability_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height


class Config:
    """Global configuration manager"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    WEIGHTS_DIR = PROJECT_ROOT /"training/train5/weights"
    EVIDENCE_DIR = PROJECT_ROOT / "database/evidence"
    
    # Model Settings
    MODEL_PATH = WEIGHTS_DIR / 'best.onnx'
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    INPUT_SIZE = (640, 640)
    
    #Onnx Runtime Settings
    PROVIDERS = ['CPUExecutionProvider']  # PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] <- to be changed for GPU 


    # Classes
    CLASSES = ['bike', 'helmet', 'no-helmet', 'number-plate']
    
    # Class Inverse Mapping
    CLASS_ID_TO_NAME = {name : idx for idx , name in enumerate(CLASSES)}

    # Tracking
    TRACKER_CACHE_TTL = 10.0  # seconds
    
    # Database
    DB_PATH = PROJECT_ROOT / "database/violations.db"
    DB_TABLE_NAME = "ATV_Base"
    
