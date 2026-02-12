# Project File Map

This document lists project files, shows which files reference or depend on which others, and explains why those connections exist.

---

- **File:** [ATV/constants.py](ATV/constants.py)
  - **Connected to:** [ATV/utils/global_utils.py](ATV/utils/global_utils.py)
  - **Why:** Defines global paths (evidence and config paths) used by utilities to read/write files.

- **File:** [ATV/logger.py](ATV/logger.py)
  - **Connected to:** used across the project by `main`, `core`, `utils`, and `Trackers` (e.g. [ATV/main.py](ATV/main.py), [ATV/utils/global_utils.py](ATV/utils/global_utils.py), [ATV/Trackers/bike_tracker.py](ATV/Trackers/bike_tracker.py), [ATV/core/core_utils.py](ATV/core/core_utils.py)).
  - **Why:** Centralized logging; modules import the `logger` to record events and errors.

- **File:** [ATV/utils/global_utils.py](ATV/utils/global_utils.py)
  - **Connected to:** [ATV/logger.py](ATV/logger.py), [ATV/constants.py](ATV/constants.py), [ATV/config/config.py](ATV/config/config.py) (indirectly via config usage), and called by application entry points like [ATV/main.py](ATV/main.py) and trackers.
  - **Why:** Provides common helpers (read_yaml, read_video, save/load JSON, save_evidence) that many components rely on for I/O and configuration access.

- **File:** [ATV/main.py](ATV/main.py)
  - **Connected to:** [ATV/logger.py](ATV/logger.py), [ATV/utils/global_utils.py](ATV/utils/global_utils.py), [ATV/Trackers/bike_tracker.py](ATV/Trackers/bike_tracker.py), and the global config file at `ATV/config/global_config.yaml`.
  - **Why:** Orchestrator/entry point — reads video, loads config, constructs trackers, and starts the processing pipeline.

- **File:** [ATV/Trackers/bike_tracker.py](ATV/Trackers/bike_tracker.py)
  - **Connected to:** [ATV/utils/global_utils.py](ATV/utils/global_utils.py) (uses `read_yaml`), [ATV/logger.py](ATV/logger.py), external libs (`ultralytics.YOLO`, `supervision`), and model files referenced in config.
  - **Why:** Implements detection + tracking logic for bikes; it loads the detector model path from the YAML config and uses utilities and logger for config and diagnostics.

- **File:** [ATV/Trackers/helmet_tracker.py](ATV/Trackers/helmet_tracker.py)
  - **Connected to:** (currently empty) intended to connect to `utils`, `logger`, and detectors similar to `bike_tracker`.
  - **Why:** Placeholder for helmet-specific tracking logic.

- **File:** [ATV/Trackers/no_plate_tracker.py](ATV/Trackers/no_plate_tracker.py)
  - **Connected to:** (currently empty) intended to connect to OCR/detector modules and `utils`.
  - **Why:** Placeholder for license-plate tracking/processing.

- **File:** [ATV/drawer/utils.py](ATV/drawer/utils.py)
  - **Connected to:** OpenCV and NumPy (internal), and intended to be used by drawer modules responsible for drawing tracks on frames.
  - **Why:** Low-level drawing helpers used by drawer modules to annotate frames with bounding boxes and labels.

- **File:** [ATV/drawer/bike_track_drawer.py](ATV/drawer/bike_track_drawer.py)
  - **Connected to:** (file exists but empty) expected to import drawing utilities and possibly tracker output structures from `Trackers`.
  - **Why:** Renders bike tracks and annotations onto frames for visualization or saving evidence.

- **File:** [ATV/drawer/helmet_track_drawer.py](ATV/drawer/helmet_track_drawer.py)
  - **Connected to:** (file exists but empty) intended to rely on `drawer/utils.py` and track data.
  - **Why:** Renders helmet-related visualizations.

- **File:** [ATV/drawer/no_plate_track_drawer.py](ATV/drawer/no_plate_track_drawer.py)
  - **Connected to:** (file exists but empty) intended to render license-plate tracks for review/evidence.
  - **Why:** Visualization of plate tracking results.

- **File:** [ATV/core/core_utils.py](ATV/core/core_utils.py)
  - **Connected to:** [ATV/logger.py](ATV/logger.py), PyTorch (torch), YAML utilities.
  - **Why:** Utility functions to load models and labels; consumed by detector wrappers and other core components.

- **File:** [ATV/core/detector.py](ATV/core/detector.py)
  - **Connected to:** [ATV/config/config.py](ATV/config/config.py) for `DetectionConfig`, [ATV/core/core_utils.py](ATV/core/core_utils.py) for `load_model`/`load_labels`.
  - **Why:** Provides a `Detector` abstraction that wraps model and label loading and applies detection; used by higher-level trackers or modules that need detections.

- **File:** [ATV/core/ocr_engine.py](ATV/core/ocr_engine.py)
  - **Connected to:** (empty) expected to connect to OCR libraries (e.g., EasyOCR) and be used by plate trackers or post-processing.
  - **Why:** OCR processing for detected license-plate crops.

- **File:** [ATV/core/api_mock.py](ATV/core/api_mock.py)
  - **Connected to:** (empty) placeholder for API stubs or mocked endpoints for testing.
  - **Why:** Allows offline testing or interface simulation without external services.

- **File:** [ATV/core/app.py](ATV/core/app.py)
  - **Connected to:** Demonstrative use of OpenCV; not integrated deeply yet.
  - **Why:** Small sandbox for camera/video testing and prototyping.

- **File:** [ATV/config/config.py](ATV/config/config.py)
  - **Connected to:** [ATV/core/detector.py](ATV/core/detector.py) (provides `DetectionConfig`) and read by utilities/trackers via YAML config files.
  - **Why:** Central location for configuration classes and defaults used when creating detectors and trackers.

- **File:** [ATV/UI/components.py](ATV/UI/components.py)
  - **Connected to:** (empty) intended to provide UI components; will import `logger` and possibly tracker interfaces.
  - **Why:** Frontend pieces for the app UI.

- **File:** [ATV/UI/app.py](ATV/UI/app.py)
  - **Connected to:** (empty) intended to be the UI entry point, connecting components to app logic.
  - **Why:** Hosts the UI that displays video, controls, and results.

- **File:** [ATV/Testing/local_testing.py](ATV/Testing/local_testing.py)
  - **Connected to:** imports `main` to run local tests; serves as a small harness.
  - **Why:** Quick local test runner for the application pipeline.

---

## Summary & next steps
- The main glue points are `ATV/main.py` (orchestrator), `ATV/utils/global_utils.py` (I/O/helpers), `ATV/logger.py` (logging), and tracker modules under `ATV/Trackers/` which perform detection/tracking.
- Several files are placeholders/empty (`Trackers/helmet_tracker.py`, `drawer/*_drawer.py`, `core/ocr_engine.py`, `core/api_mock.py`, `UI/*`) — they should be implemented to complete the pipeline.

If you want, I can:
- expand the map to include line-level references for each import, or
- generate a visual graph (DOT / PNG) of these connections, or
- auto-fill the empty placeholders with minimal scaffolding.
