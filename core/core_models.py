from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class Detection:
    bbox : Tuple[int , int , int , int ] # (x1, y1, x2, y2)
    class_id : int
    class_name : str
    confidence : float
    tracker_id : Optional[int] = None

@dataclass
class ViolationEvent:
    timestamp : float
    tracker_id : int    
    detection : list[Detection]
    violation_type : str 
    frame : np.ndarray


@dataclass 
class DetectionConfig:
    model_path : str
    confidence_threshold : float = 0.5
    nms_threshold : float = 0.4
    input_size : tuple[int , int] = (640, 640)
    device : Optional[str] = 'cpu'

