# Notebook to Modular Code: Implementation Plan

## Overview
This guide will help you transition from notebook-based testing to production-ready modular code for your Traffic Violation Detection System. The system uses YOLOv8 for detection, ByteTrack for tracking, and LangGraph for workflow orchestration.

---

## 🎯 Implementation Order & File Creation Sequence

### Phase 1: Foundation Layer (Data Classes & Utilities)
### Phase 2: Core Detection & Processing
### Phase 3: Services Layer
### Phase 4: Workflow Orchestration
### Phase 5: API & Integration
### Phase 6: Testing & Validation

---

## 📋 Phase 1: Foundation Layer

### 1.1 Create Data Models (`core/models.py`)

**Priority**: ⭐⭐⭐ **First File to Create**

**Purpose**: Define all data structures used throughout the system

**What to Create**:
```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None

@dataclass
class ViolationEvent:
    """Violation event data"""
    timestamp: float
    frame: np.ndarray
    detections: List[Detection]
    tracker_id: int
    violation_type: str  # 'no-helmet', 'signal-violation', etc.
    
@dataclass
class DetectionConfig:
    """Configuration for detector"""
    model_path: str
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    img_size: int = 640
```

**Why First**: 
- All other modules will import these classes
- Provides type safety and structure
- Easy to test in notebook with sample data

**How to Test in Notebook**:
```python
# In notebook cell:
from core.models import Detection, ViolationEvent
import time

# Create a sample detection
det = Detection(
    bbox=(100, 200, 300, 400),
    class_id=2,
    class_name='no-helmet',
    confidence=0.85,
    tracker_id=5
)
print(det)
```

---

### 1.2 Create Configuration Manager (`core/config.py`)

**Priority**: ⭐⭐⭐

**Purpose**: Centralize all configuration settings

**What to Create**:
```python
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    """Global configuration manager"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    WEIGHTS_DIR = PROJECT_ROOT / 'weights'
    EVIDENCE_DIR = PROJECT_ROOT / 'evidence'
    
    # Model Settings
    MODEL_PATH = WEIGHTS_DIR / 'best.onnx'
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    
    # Classes
    CLASSES = ['bike', 'helmet', 'no-helmet', 'number-plate']
    
    # Tracking
    TRACKER_CACHE_TTL = 10.0  # seconds
    
    # Database
    DB_PATH = PROJECT_ROOT / 'smart_traffic.db'
    
    @classmethod
    def load_from_yaml(cls, path: str) -> Dict[str, Any]:
        """Load additional config from YAML"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
```

**What to Take from Notebook**:
- Hardcoded paths
- Threshold values
- Class names list

**How to Test**:
```python
# In notebook:
from core.config import Config

print(Config.MODEL_PATH)
print(Config.CLASSES)
```

---

## 📋 Phase 2: Core Detection & Processing

### 2.1 Refactor Detector ([core/detector.py](file:///home/machine/atv_dem/core/detector.py)) ✅ Already Done

**Status**: Already implemented, but needs enhancement

**What to Add**:
```python
# Add to existing detector.py
from core.models import Detection
from core.config import Config

class YOLOv8Detector:
    # ... existing code ...
    
    def detect_to_objects(self, frame) -> List[Detection]:
        """
        Convert supervision.Detections to List[Detection] objects
        """
        sv_detections = self.detect(frame)
        
        detections = []
        for i in range(len(sv_detections)):
            det = Detection(
                bbox=tuple(sv_detections.xyxy[i].astype(int)),
                class_id=int(sv_detections.class_id[i]),
                class_name=self.classes[sv_detections.class_id[i]],
                confidence=float(sv_detections.confidence[i]),
                tracker_id=None
            )
            detections.append(det)
        
        return detections
```

**What to Take from Notebook**:
- Detection logic (already done)
- Preprocessing steps (already done)
- Postprocessing logic (already done)

**What to Return**:
- [detect()](file:///home/machine/atv_dem/core/detector.py#54-115) → Returns `supervision.Detections` (for tracking)
- `detect_to_objects()` → Returns `List[Detection]` (for business logic)

**How to Test in Notebook**:
```python
# In notebook:
from core.detector import YOLOv8Detector
import cv2

detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
frame = cv2.imread('test_image.jpg')

# Test supervision format
sv_dets = detector.detect(frame)
print(f"Found {len(sv_dets)} detections")

# Test object format
obj_dets = detector.detect_to_objects(frame)
for det in obj_dets:
    print(f"{det.class_name}: {det.confidence:.2f}")
```

---

### 2.2 Create Tracking Service (`core/tracker.py`)

**Priority**: ⭐⭐

**Purpose**: Wrap ByteTrack with deduplication logic

**What to Create**:
```python
import supervision as sv
from typing import List, Dict
import time
from core.models import Detection
from core.config import Config

class ViolationTracker:
    """
    Manages object tracking and violation deduplication
    """
    
    def __init__(self, cache_ttl: float = Config.TRACKER_CACHE_TTL):
        self.byte_tracker = sv.ByteTrack()
        self.violation_cache: Dict[int, float] = {}
        self.cache_ttl = cache_ttl
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections
        """
        return self.byte_tracker.update_with_detections(detections)
    
    def is_new_violation(self, tracker_id: int) -> bool:
        """
        Check if this tracker_id represents a new violation
        """
        current_time = time.time()
        last_logged = self.violation_cache.get(tracker_id, 0)
        
        if current_time - last_logged > self.cache_ttl:
            self.violation_cache[tracker_id] = current_time
            return True
        
        return False
    
    def cleanup_old_tracks(self):
        """
        Remove expired entries from cache
        """
        current_time = time.time()
        expired = [
            tid for tid, ts in self.violation_cache.items()
            if current_time - ts > self.cache_ttl * 2
        ]
        for tid in expired:
            del self.violation_cache[tid]
```

**What to Take from [graph.py](file:///home/machine/atv_dem/core/graph.py)**:
- `byte_tracker` instance
- `violation_log_cache` dictionary
- `CACHE_TTL` logic

**What to Return**:
- Updated detections with tracker IDs
- Boolean indicating if violation is new

**How to Test in Notebook**:
```python
# In notebook (multi-frame test):
from core.tracker import ViolationTracker
from core.detector import YOLOv8Detector
import cv2

detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
tracker = ViolationTracker(cache_ttl=5.0)

cap = cv2.VideoCapture('vid1.mp4')
for i in range(30):  # Test 30 frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    detections = detector.detect(frame)
    
    # Track
    tracked_detections = tracker.update(detections)
    
    # Check for violations
    if tracked_detections.tracker_id is not None:
        for tid in tracked_detections.tracker_id:
            if tracker.is_new_violation(tid):
                print(f"Frame {i}: New violation from track ID {tid}")
```

---

### 2.3 Enhance Visualizer ([core/visualizer.py](file:///home/machine/atv_dem/core/visualizer.py))

**Priority**: ⭐

**Purpose**: Add support for tracked detections

**What to Add**:
```python
import cv2
import supervision as sv
from typing import List
from core.models import Detection

class Visualizer:
    def __init__(self):
        # Existing color dict
        self.colors = {
            'helmet': (0, 255, 0),
            'no-helmet': (0, 0, 255),
            'number-plate': (255, 0, 0),
            'bike': (255, 255, 0)
        }
        
        # Supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
    
    def draw_tracked(self, frame, detections: sv.Detections, class_names: List[str]):
        """
        Draw detections with tracker IDs
        """
        labels = []
        for i in range(len(detections)):
            class_name = class_names[detections.class_id[i]]
            conf = detections.confidence[i]
            
            if detections.tracker_id is not None:
                tid = detections.tracker_id[i]
                label = f"#{tid} {class_name} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"
            
            labels.append(label)
        
        annotated = self.box_annotator.annotate(
            scene=frame.copy(), 
            detections=detections
        )
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=detections,
            labels=labels
        )
        
        return annotated
    
    # Keep existing draw_boxes() for compatibility
```

**What to Take from [graph.py](file:///home/machine/atv_dem/core/graph.py)**:
- BoxAnnotator usage
- LabelAnnotator usage
- Label formatting logic

**How to Test**:
```python
# In notebook:
from core.visualizer import Visualizer
from core.detector import YOLOv8Detector
from core.tracker import ViolationTracker
import cv2

detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
tracker = ViolationTracker()
viz = Visualizer()

frame = cv2.imread('test.jpg')
detections = detector.detect(frame)
tracked = tracker.update(detections)

annotated = viz.draw_tracked(frame, tracked, detector.classes)
cv2.imwrite('output_tracked.jpg', annotated)
```

---

## 📋 Phase 3: Services Layer

### 3.1 Enhance Evidence Manager ([services/evidence_manager.py](file:///home/machine/atv_dem/services/evidence_manager.py))

**Priority**: ⭐⭐

**Current**: Already exists, but enhance it

**What to Add**:
```python
from pathlib import Path
from datetime import datetime
import cv2
from typing import Tuple, List, Optional
from core.models import ViolationEvent, Detection

class EvidenceManager:
    # ... existing code ...
    
    def log_violation_with_metadata(
        self, 
        violation: ViolationEvent
    ) -> Tuple[str, float]:
        """
        Save violation with full metadata
        """
        timestamp = violation.timestamp
        dt = datetime.fromtimestamp(timestamp)
        
        # Directory structure
        date_dir = self.base_dir / f"{dt.year}-{dt.month:02d}-{dt.day:02d}"
        hour_dir = date_dir / f"{dt.hour:02d}"
        hour_dir.mkdir(parents=True, exist_ok=True)
        
        # Save image
        filename = f"violation_{timestamp}_{violation.tracker_id}.jpg"
        filepath = hour_dir / filename
        cv2.imwrite(str(filepath), violation.frame)
        
        # Save metadata JSON
        metadata = {
            'timestamp': timestamp,
            'tracker_id': violation.tracker_id,
            'violation_type': violation.violation_type,
            'detections': [
                {
                    'class': det.class_name,
                    'confidence': det.confidence,
                    'bbox': det.bbox
                }
                for det in violation.detections
            ]
        }
        
        import json
        meta_path = hour_dir / f"violation_{timestamp}_{violation.tracker_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(filepath), timestamp
```

**What to Take from Notebook**:
- Directory creation logic
- File naming conventions
- Timestamp formatting

**What to Return**:
- File path to saved evidence
- Timestamp

---

### 3.2 Create OCR Service ([services/ocr_service.py](file:///home/machine/atv_dem/services/ocr_service.py))

**Priority**: ⭐⭐

**Purpose**: Replace mock OCR with real EasyOCR

**What to Create**:
```python
import easyocr
import cv2
import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

class OCRService:
    """
    License plate text extraction using EasyOCR
    """
    
    def __init__(self, languages=['en'], gpu=False):
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def extract_plate_text(
        self, 
        image: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> str:
        """
        Extract text from license plate region
        
        Args:
            image: Full frame
            bbox: Bounding box (x1, y1, x2, y2) of plate. If None, use full image
        
        Returns:
            Extracted text
        """
        if bbox:
            x1, y1, x2, y2 = bbox
            plate_crop = image[y1:y2, x1:x2]
        else:
            plate_crop = image
        
        # Preprocess
        plate_crop = self._preprocess_plate(plate_crop)
        
        # OCR
        results = self.reader.readtext(plate_crop)
        
        if not results:
            return ""
        
        # Combine all text
        text = " ".join([res[1] for res in results])
        
        # Clean up
        text = self._clean_plate_text(text)
        
        return text
    
    def _preprocess_plate(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance plate image for better OCR
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def _clean_plate_text(self, text: str) -> str:
        """
        Clean OCR output
        """
        # Remove special characters, keep alphanumeric
        import re
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return text
    
    def extract_from_detections(
        self, 
        frame: np.ndarray,
        detections: List,
        class_name: str = 'number-plate'
    ) -> List[Tuple[str, Tuple]]:
        """
        Extract text from all plate detections
        
        Returns:
            List of (text, bbox) tuples
        """
        results = []
        
        for det in detections:
            if det.class_name == class_name:
                text = self.extract_plate_text(frame, det.bbox)
                if text:
                    results.append((text, det.bbox))
        
        return results
```

**What to Take from Notebook**:
- EasyOCR initialization code
- Preprocessing steps (grayscale, CLAHE, etc.)
- Text cleaning regex

**What to Return**:
- Extracted text string
- Confidence score (optional)

**How to Test in Notebook**:
```python
# In notebook:
from services.ocr_service import OCRService
import cv2

ocr = OCRService(languages=['en'], gpu=False)

# Test on full image
plate_img = cv2.imread('plate_sample.jpg')
text = ocr.extract_plate_text(plate_img)
print(f"Detected: {text}")

# Test with detector
from core.detector import YOLOv8Detector
detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')

frame = cv2.imread('test.jpg')
detections = detector.detect_to_objects(frame)

results = ocr.extract_from_detections(frame, detections)
for text, bbox in results:
    print(f"Plate: {text} at {bbox}")
```

---

### 3.3 Database Service ([services/database.py](file:///home/machine/atv_dem/services/database.py)) ✅ Already Done

**Status**: Already implemented

**What to Verify**:
- `insert_incident()` method exists
- Table schema supports your needs
- Add index on timestamp for faster queries

---

## 📋 Phase 4: Workflow Orchestration

### 4.1 Refactor LangGraph Workflow ([core/graph.py](file:///home/machine/atv_dem/core/graph.py))

**Priority**: ⭐⭐⭐ **Critical**

**What to Change**:

**Current Issues**:
- Global instances (not reusable)
- Tightly coupled
- Hard to test

**New Structure**:
```python
from typing import TypedDict, Any, Optional
from langgraph.graph import StateGraph, END
import time
from core.detector import YOLOv8Detector
from core.tracker import ViolationTracker
from core.visualizer import Visualizer
from core.models import ViolationEvent, Detection
from services.evidence_manager import EvidenceManager
from services.database import Database
from services.ocr_service import OCRService
from core.config import Config

# State Definition
class AgentState(TypedDict):
    frame: Any
    detections: Any  # sv.Detections
    violation_detected: bool
    evidence_path: Optional[str]
    timestamp: float
    plate_text: Optional[str]

class TrafficViolationGraph:
    """
    LangGraph workflow for traffic violation detection
    """
    
    def __init__(
        self,
        detector: YOLOv8Detector,
        tracker: ViolationTracker,
        visualizer: Visualizer,
        evidence_manager: EvidenceManager,
        database: Database,
        ocr_service: Optional[OCRService] = None
    ):
        self.detector = detector
        self.tracker = tracker
        self.visualizer = visualizer
        self.evidence_manager = evidence_manager
        self.database = database
        self.ocr_service = ocr_service
        
        self.graph = self._build_graph()
    
    def _detect_node(self, state: AgentState):
        """Node: Run object detection"""
        frame = state["frame"]
        detections = self.detector.detect(frame)
        return {"detections": detections}
    
    def _track_node(self, state: AgentState):
        """Node: Update object tracking"""
        detections = state["detections"]
        tracked = self.tracker.update(detections)
        return {"detections": tracked}
    
    def _visualize_node(self, state: AgentState):
        """Node: Annotate frame with detections"""
        frame = state["frame"]
        detections = state["detections"]
        
        annotated = self.visualizer.draw_tracked(
            frame, detections, self.detector.classes
        )
        
        return {"frame": annotated}
    
    def _filter_node(self, state: AgentState):
        """Node: Check for violations"""
        detections = state["detections"]
        is_violation = False
        
        no_helmet_id = self.detector.classes.index('no-helmet')
        
        if detections.tracker_id is not None:
            for class_id, tracker_id in zip(
                detections.class_id, 
                detections.tracker_id
            ):
                if class_id == no_helmet_id:
                    if self.tracker.is_new_violation(tracker_id):
                        is_violation = True
                        break
        
        return {"violation_detected": is_violation}
    
    def _evidence_node(self, state: AgentState):
        """Node: Save evidence"""
        frame = state["frame"]
        
        path, ts = self.evidence_manager.log_violation(frame, [])
        
        if path:
            self.database.insert_incident(path, ts, "NEW")
        
        return {"evidence_path": path, "timestamp": ts}
    
    def _ocr_node(self, state: AgentState):
        """Node: Extract plate text (optional)"""
        if not self.ocr_service:
            return {}
        
        frame = state["frame"]
        detections_list = self.detector.detect_to_objects(frame)
        
        results = self.ocr_service.extract_from_detections(
            frame, detections_list
        )
        
        plate_text = results[0][0] if results else None
        
        return {"plate_text": plate_text}
    
    def _analytics_node(self, state: AgentState):
        """Node: Analytics for non-violations"""
        return {}
    
    def _router(self, state: AgentState):
        """Conditional router"""
        if state["violation_detected"]:
            return "save_evidence"
        else:
            return "analytics"
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        builder = StateGraph(AgentState)
        
        # Add nodes
        builder.add_node("detect", self._detect_node)
        builder.add_node("track", self._track_node)
        builder.add_node("visualize", self._visualize_node)
        builder.add_node("filter", self._filter_node)
        builder.add_node("save_evidence", self._evidence_node)
        builder.add_node("analytics", self._analytics_node)
        
        # Add OCR node if service available
        if self.ocr_service:
            builder.add_node("ocr", self._ocr_node)
        
        # Set entry
        builder.set_entry_point("detect")
        
        # Add edges
        builder.add_edge("detect", "track")
        builder.add_edge("track", "visualize")
        builder.add_edge("visualize", "filter")
        
        # Conditional routing
        builder.add_conditional_edges(
            "filter",
            self._router,
            {
                "save_evidence": "save_evidence",
                "analytics": "analytics"
            }
        )
        
        # Terminal edges
        if self.ocr_service:
            builder.add_edge("save_evidence", "ocr")
            builder.add_edge("ocr", END)
        else:
            builder.add_edge("save_evidence", END)
        
        builder.add_edge("analytics", END)
        
        return builder.compile()
    
    def process_frame(self, frame, timestamp: float = None):
        """
        Process a single frame through the workflow
        """
        if timestamp is None:
            timestamp = time.time()
        
        initial_state = {
            "frame": frame,
            "detections": None,
            "violation_detected": False,
            "evidence_path": None,
            "timestamp": timestamp,
            "plate_text": None
        }
        
        result = self.graph.invoke(initial_state)
        
        return result
```

**What to Take from Current [graph.py](file:///home/machine/atv_dem/core/graph.py)**:
- All node functions
- State definition
- Graph structure

**What's Different**:
- Class-based (not global)
- Dependency injection
- Easier to test
- Configurable OCR

**How to Test in Notebook**:
```python
# In notebook:
from core.graph import TrafficViolationGraph
from core.detector import YOLOv8Detector
from core.tracker import ViolationTracker
from core.visualizer import Visualizer
from services.evidence_manager import EvidenceManager
from services.database import Database
from services.ocr_service import OCRService
import cv2

# Initialize components
detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
tracker = ViolationTracker()
visualizer = Visualizer()
evidence_mgr = EvidenceManager()
db = Database()
ocr = OCRService()

# Create workflow
workflow = TrafficViolationGraph(
    detector=detector,
    tracker=tracker,
    visualizer=visualizer,
    evidence_manager=evidence_mgr,
    database=db,
    ocr_service=ocr
)

# Test on video
cap = cv2.VideoCapture('vid1.mp4')
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    
    result = workflow.process_frame(frame)
    
    if result['violation_detected']:
        print(f"Frame {i}: Violation detected!")
        if result['plate_text']:
            print(f"  Plate: {result['plate_text']}")
    
    # Display annotated frame
    cv2.imshow('Frame', result['frame'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📋 Phase 5: API & Integration

### 5.1 Update Main API ([api/main.py](file:///home/machine/atv_dem/api/main.py))

**Priority**: ⭐⭐

**What to Change**:

**Old Approach**:
```python
# Global graph
from core.graph import graph

result = graph.invoke({"frame": frame, ...})
```

**New Approach**:
```python
from fastapi import FastAPI
from core.graph import TrafficViolationGraph
from core.detector import YOLOv8Detector
from core.tracker import ViolationTracker
# ... other imports

app = FastAPI()

# Initialize workflow at startup
@app.on_event("startup")
async def startup_event():
    global workflow
    
    detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
    tracker = ViolationTracker()
    visualizer = Visualizer()
    evidence_mgr = EvidenceManager()
    db = Database()
    ocr = OCRService()
    
    workflow = TrafficViolationGraph(
        detector=detector,
        tracker=tracker,
        visualizer=visualizer,
        evidence_manager=evidence_mgr,
        database=db,
        ocr_service=ocr
    )

@app.get("/video_feed")
async def video_feed():
    """Stream processed video"""
    from core.camera import VideoLoader
    
    camera = VideoLoader(source='vid1.mp4')
    
    def generate():
        while True:
            frame = camera.get_frame()
            if frame is None:
                break
            
            result = workflow.process_frame(frame)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', result['frame'])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
```

---

### 5.2 Create Main Entry Point (`main.py`)

**Priority**: ⭐

**Purpose**: Provide CLI for testing without API

**What to Create**:
```python
#!/usr/bin/env python3
"""
Main entry point for traffic violation detection
Can run in CLI mode or API mode
"""

import argparse
import cv2
import time
from pathlib import Path

from core.detector import YOLOv8Detector
from core.tracker import ViolationTracker
from core.visualizer import Visualizer
from services.evidence_manager import EvidenceManager
from services.database import Database
from services.ocr_service import OCRService
from core.graph import TrafficViolationGraph
from core.camera import VideoLoader

def run_cli(video_source: str, show_display: bool = True):
    """
    Run in CLI mode with video display
    """
    print("Initializing components...")
    
    # Initialize all components
    detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')
    tracker = ViolationTracker()
    visualizer = Visualizer()
    evidence_mgr = EvidenceManager()
    db = Database()
    ocr = OCRService()
    
    # Create workflow
    workflow = TrafficViolationGraph(
        detector=detector,
        tracker=tracker,
        visualizer=visualizer,
        evidence_manager=evidence_mgr,
        database=db,
        ocr_service=ocr
    )
    
    # Load video
    camera = VideoLoader(source=video_source)
    
    print(f"Processing video from: {video_source}")
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("End of video stream")
                break
            
            # Process frame
            result = workflow.process_frame(frame)
            
            frame_count += 1
            
            # Log violations
            if result['violation_detected']:
                print(f"[Frame {frame_count}] Violation detected!")
                if result['plate_text']:
                    print(f"  License Plate: {result['plate_text']}")
                print(f"  Evidence saved: {result['evidence_path']}")
            
            # Display
            if show_display:
                cv2.imshow('Traffic Violation Detection', result['frame'])
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User quit")
                    break
            
            # FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Processed {frame_count} frames @ {fps:.2f} FPS")
    
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        # Stats
        total_time = time.time() - start_time
        print(f"\nTotal frames: {frame_count}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average FPS: {frame_count/total_time:.2f}")

def run_api():
    """
    Run in API mode
    """
    import uvicorn
    from api.main import app
    
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection System"
    )
    parser.add_argument(
        '--mode',
        choices=['cli', 'api'],
        default='cli',
        help='Run mode: cli or api'
    )
    parser.add_argument(
        '--source',
        default='vid1.mp4',
        help='Video source (file path or camera index)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable video display (headless mode)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli(args.source, show_display=not args.no_display)
    else:
        run_api()
```

**How to Test**:
```bash
# Test CLI mode
python main.py --mode cli --source vid1.mp4

# Test headless
python main.py --mode cli --source 0 --no-display

# Test API mode
python main.py --mode api
```

---

## 📋 Phase 6: Testing & Validation

### 6.1 Create Unit Tests (`tests/test_detector.py`)

**Priority**: ⭐

```python
import pytest
import cv2
import numpy as np
from core.detector import YOLOv8Detector

@pytest.fixture
def detector():
    return YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')

@pytest.fixture
def sample_frame():
    # Create a dummy frame
    return np.zeros((640, 640, 3), dtype=np.uint8)

def test_detector_initialization(detector):
    assert detector is not None
    assert len(detector.classes) == 4

def test_detect_returns_detections(detector, sample_frame):
    detections = detector.detect(sample_frame)
    assert detections is not None

def test_detect_to_objects(detector, sample_frame):
    # Add a test image with actual detections
    frame = cv2.imread('/home/machine/atv_dem/train5/val_batch0_labels.jpg')
    objects = detector.detect_to_objects(frame)
    assert isinstance(objects, list)
```

---

## 📊 Summary: Order of Implementation

| Step | File | Purpose | Dependencies |
|------|------|---------|--------------|
| 1 | `core/models.py` | Data structures | None |
| 2 | `core/config.py` | Configuration | None |
| 3 | [core/detector.py](file:///home/machine/atv_dem/core/detector.py) | Enhancement | models, config |
| 4 | `core/tracker.py` | Tracking service | models, config |
| 5 | [core/visualizer.py](file:///home/machine/atv_dem/core/visualizer.py) | Visualization | models |
| 6 | [services/ocr_service.py](file:///home/machine/atv_dem/services/ocr_service.py) | OCR functionality | models |
| 7 | [services/evidence_manager.py](file:///home/machine/atv_dem/services/evidence_manager.py) | Enhancement | models |
| 8 | [core/graph.py](file:///home/machine/atv_dem/core/graph.py) | Workflow refactor | All core + services |
| 9 | `main.py` | CLI entry point | graph |
| 10 | [api/main.py](file:///home/machine/atv_dem/api/main.py) | API updates | graph |
| 11 | `tests/` | Unit tests | All modules |

---

## 🧪 Testing Strategy

### Notebook Testing Approach

For each module you create, test it immediately in a notebook:

```python
# Standard notebook test template
# Cell 1: Imports
from core.models import Detection
from core.detector import YOLOv8Detector
import cv2

# Cell 2: Initialize
detector = YOLOv8Detector('/home/machine/atv_dem/weights/best.onnx')

# Cell 3: Load test data
frame = cv2.imread('test_image.jpg')

# Cell 4: Run and inspect
detections = detector.detect_to_objects(frame)
for det in detections:
    print(det)

# Cell 5: Visualize
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
for det in detections:
    x1, y1, x2, y2 = det.bbox
    plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        fill=False, color='red'))
plt.show()
```

---

## 🚀 Quick Start Guide

Start implementing in this exact order:

1. **Day 1**: Create `core/models.py` and `core/config.py`, test in notebook
2. **Day 2**: Enhance [core/detector.py](file:///home/machine/atv_dem/core/detector.py) with new methods, test in notebook
3. **Day 3**: Create `core/tracker.py`, test with video in notebook
4. **Day 4**: Create [services/ocr_service.py](file:///home/machine/atv_dem/services/ocr_service.py), test with plate images
5. **Day 5**: Refactor [core/graph.py](file:///home/machine/atv_dem/core/graph.py) to use class-based approach
6. **Day 6**: Create `main.py` CLI tool, test end-to-end
7. **Day 7**: Update [api/main.py](file:///home/machine/atv_dem/api/main.py), test web interface

---

## 📝 Migration Checklist

- [ ] Phase 1: Foundation Layer
  - [ ] Create `core/models.py`
  - [ ] Create `core/config.py`
  - [ ] Test in notebook

- [ ] Phase 2: Core Processing
  - [ ] Enhance [core/detector.py](file:///home/machine/atv_dem/core/detector.py)
  - [ ] Create `core/tracker.py`
  - [ ] Enhance [core/visualizer.py](file:///home/machine/atv_dem/core/visualizer.py)
  - [ ] Test all modules in notebook

- [ ] Phase 3: Services
  - [ ] Create [services/ocr_service.py](file:///home/machine/atv_dem/services/ocr_service.py)
  - [ ] Enhance [services/evidence_manager.py](file:///home/machine/atv_dem/services/evidence_manager.py)
  - [ ] Test each service in notebook

- [ ] Phase 4: Workflow
  - [ ] Refactor [core/graph.py](file:///home/machine/atv_dem/core/graph.py)
  - [ ] Test workflow with sample video

- [ ] Phase 5: Integration
  - [ ] Create `main.py`
  - [ ] Update [api/main.py](file:///home/machine/atv_dem/api/main.py)
  - [ ] Test CLI and API modes

- [ ] Phase 6: Testing
  - [ ] Create unit tests
  - [ ] Create integration tests
  - [ ] Performance testing

---

## 💡 Key Principles

1. **Test Each Module Immediately**: Don't move forward until current module works
2. **Keep Notebooks for Experimentation**: Use notebooks to prototype, then move to modules
3. **Start Simple**: Get basic version working, then add features
4. **Use Type Hints**: Makes debugging easier
5. **Dependency Injection**: Pass dependencies, don't use globals
6. **Logging**: Add logging to every module for debugging

---

## 🔗 References

- Current Architecture: [architecture_map.md](file:///home/machine/atv_dem/architecture_map.md)
- Existing Detector: [core/detector.py](file:///home/machine/atv_dem/core/detector.py)
- Existing Graph: [core/graph.py](file:///home/machine/atv_dem/core/graph.py)
- Existing Tracker Logic: Lines 44-51 in [core/graph.py](file:///home/machine/atv_dem/core/graph.py#L44-L51)
