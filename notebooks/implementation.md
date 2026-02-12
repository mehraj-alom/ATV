# How to Run the Traffic Violation Detection System

## 🚀 Quick Start

### 1. Activate Virtual Environment & Start Server

```bash
cd /home/machine/atv_dem
source venv/bin/activate
PYTHONPATH=/home/machine/atv_dem:$PYTHONPATH python api/main.py
```

### 2. Access the Application

Once the server starts, you'll see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Open your browser and navigate to:
- **Dashboard:** http://localhost:8000
- **Live Video Feed:** http://localhost:8000/video_feed
- **API - Pending Incidents:** http://localhost:8000/api/incidents/pending

---

## 📋 What Happens When You Run

1. **Video Processing**: The system loads `vid1.mp4` and starts processing frames
2. **LangGraph Workflow**: Each frame goes through:
   - **Detect** → YOLOv8 ONNX detection (Helmet/No-Helmet/Number-Plate/Motorcycle)
   - **Track** → ByteTrack assigns unique IDs to tracked objects
   - **Visualize** → Annotates frame with bounding boxes and IDs
   - **Filter** → Checks for "No-Helmet" violations
   - **Save Evidence** → If violation detected, saves to `/evidence/` and logs to database
3. **Web Dashboard**: Shows live video feed and pending violations for review
4. **OCR & Verification**: Click "Verify" on incidents to extract license plate numbers

---

## 🔧 Alternative Run Methods

### Using Uvicorn Directly

```bash
cd /home/machine/atv_dem
source venv/bin/activate
PYTHONPATH=/home/machine/atv_dem:$PYTHONPATH uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-restart on code changes (useful for development).

---

## 📹 Video Source Options

The system tries to load video sources in this order:

1. **vid1.mp4** (default) - Pre-recorded video
2. **Camera 0** - USB/Built-in webcam
3. **MockVideoLoader** - Fallback dummy frames

To change the video source, edit `api/main.py` line 38:

```python
video_source = "/home/machine/atv_dem/vid1.mp4"  # Change path here
# OR use camera: video_source = 0
```

---

## 🛑 How to Stop

Press **Ctrl+C** in the terminal to stop the server.

---

## 📊 API Endpoints

### GET /
Dashboard HTML interface

### GET /video_feed
MJPEG stream of processed video with annotations

### GET /api/incidents/pending
Returns JSON list of pending violations

### POST /api/incidents/{id}/verify
Runs OCR on violation image, updates status to "VERIFIED"

### POST /api/incidents/{id}/reject
Marks violation as "FALSE_POSITIVE"

---

## 🗂️ System Architecture

```
Camera/Video → FastAPI → LangGraph Workflow
                            ↓
                   [Detect → Track → Visualize → Filter]
                            ↓
                   Save Evidence (if violation)
                            ↓
                   Database (SQLite)
                            ↓
                   Web Dashboard (Human Review)
                            ↓
                   OCR → Parivahan Mock API → Notification
```

See [architecture_map.md](architecture_map.md) for detailed component linkage.

---

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'core'
**Solution:** Make sure to set `PYTHONPATH`:
```bash
PYTHONPATH=/home/machine/atv_dem:$PYTHONPATH python api/main.py
```

### Port Already in Use
**Solution:** Change port or kill existing process:
```bash
lsof -ti:8000 | xargs kill -9
```

### Missing Dependencies
**Solution:** Install requirements:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### ONNX Model Not Found
**Solution:** Ensure `weights/best.onnx` exists (✓ verified present)

---

## 📝 Development Notes

- **Database:** SQLite at `smart_traffic.db`
- **Evidence Storage:** `/evidence/{YYYY-MM-DD}/{HH}/violation_*.jpg`
- **YOLO Classes:** `['Helmet', 'No-Helmet', 'Number-Plate', 'Motorcycle']`
- **Deduplication:** Same track ID won't be logged within 10 seconds
