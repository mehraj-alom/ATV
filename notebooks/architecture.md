# System Architecture & Linkage Map

This document outlines how files and functions in the `atv_dem` project interact with each other, implementing the FlowChart architecture.

## 1. Core Detection & Logic Flow

### `api/main.py`
- **Role**: Entry point. initializes the FastAPI app and the LangGraph workflow.
- **Links**:
  - Code: `app.include_router(dashboard_router)` -> Connects to UI.
  - Code: `graph_runner.run(frame)` -> Invokes `core/graph.py`.

### `core/graph.py`
- **Role**: Defines the LangGraph agentic workflow (The "Brain").
- **Links**:
  - **Node: Detect**: Calls `core/detector.py:YOLOv8Detector.detect()`.
  - **Node: Track**: Calls `supervision.ByteTrack`.
  - **Node: Visualize**: Annotates frame with Bounding Boxes & Tracking IDs.
  - **Node: Filter**: Logic check (if "no-helmet" in detections & new Track ID).
  - **Node: Save Evidence**: Calls `services/evidence_manager.py`.
  - **Node: Human Review**: Pauses execution or flags state for `api/main.py`.

### `core/detector.py`
- **Role**: Wraps ONNX Runtime for YOLOv8 inference.
- **Function**: `detect(frame)`
  - **Input**: Raw image frame (numpy array).
  - **Output**: List of detections `[{class, conf, bbox}]`.
  - **Dependency**: Loads `<project_root>/weights/best.onnx`.

### `core/visualizer.py`
- **Role**: Draws bounding boxes on images.
- **Function**: `draw_boxes(frame, detections)`
  - **Used By**: `core/graph.py` (before logging evidence) and `api/main.py` (for live feed display).

## 2. Evidence & Data Services

### `services/evidence_manager.py`
- **Role**: Manages file storage for violations.
- **Function**: `log_violation(frame, metadata)`
  - **Action**: Saves file to `/evidence/{YYYY-MM-DD}/{HH}/violation_{ts}.jpg`.
  - **Links**: data passed to `services/database.py`.

### `services/database.py`
- **Role**: SQLite interaction.
- **Function**: `insert_incident(path, timestamp, status)`
  - **Used By**: `services/evidence_manager.py` and `services/ocr_service.py`.

### `services/ocr_service.py`
- **Role**: Extracts text from license plates.
- **Trigger**: Called when User clicks "Verify" on Dashboard.
- **Function**: `extract_text(image_path)` -> Returns string.

## 3. Human-in-the-Loop (UI)

### `templates/dashboard.html` & `static/style.css`
- **Role**: Frontend interface.
- **Link**: Fetches data from `api/main.py` endpoints (`/incidents/review`).
- **Action**: User clicks sends POST to `/incidents/{id}/verify`.

## 4. External Simulation

### `services/parivahan_mock.py`
- **Role**: Simulates Government Database.
- **Function**: `get_vehicle_details(plate_number)`
  - **Used By**: `services/notifier.py` or `api/main.py` after OCR.

## Data Flow Summary

1. **Camera** -> `api/main.py`
2. `api/main.py` -> `core/graph.py` (LangGraph)
3. `core/graph.py` -> `core/detector.py` (ONNX)
4. IF Violation: `core/graph.py` -> `services/evidence_manager.py` (Disk Save)
5. `api/main.py` sets status to "NEEDS_REVIEW"
6. **User** (Dashboard) -> "Verify" detected
7. `api/main.py` -> `services/ocr_service.py`
8. `services/ocr_service.py` -> `services/parivahan_mock.py`
9. Result -> `services/database.py`
