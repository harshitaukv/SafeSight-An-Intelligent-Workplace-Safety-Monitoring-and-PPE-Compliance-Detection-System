# 🦺 AI-Based PPE Compliance Monitoring — Module 1

An end-to-end computer vision system that detects whether construction workers are wearing proper **Personal Protective Equipment (PPE)** and flags safety violations in real time.

| | |
|---|---|
| **Model** | YOLOv8m |
| **mAP50** | 0.841 |
| **mAP50-95** | 0.467 |
| **Backend** | FastAPI |
| **Frontend** | React + Tailwind CSS |
| **Status** | ✅ Model inference + frontend integration completed |

---

## 📌 Overview

PPE detected by the system:

- Helmet
- Gloves
- Vest
- Boots
- Goggles

The pipeline detects each PPE item, associates it with the correct worker, applies rule-based violation checks, and logs every result — giving a complete safety-compliance engine for construction sites.

---

## 🏗️ Architecture

```
Input Image
    ↓
YOLOv8 Detection
    ↓
PPE Association Logic
    ↓
Rule-Based Violation Engine
    ↓
FastAPI API Layer
    ↓
React Frontend
    ↓
JSON Logging
```

---

## 📂 Project Structure

```
PPE/
│── detect.py             # YOLO inference
│── associate.py          # Maps PPE items to workers
│── violation_checker.py  # Rule-based violation engine
│── api.py                # FastAPI controller (main entry point)
│── app.py                # Old Streamlit prototype
│── violations.json       # Historical detection logs
│── uploads/               # Uploaded input images
│── outputs/               # Annotated YOLO output images
│── runs/                  # YOLO training/inference runs
│── frontend/               # React + Tailwind UI
```

---

## ⚙️ How It Works

### 1. Dataset Collection & Cleaning
- Sourced from Kaggle: construction worker images with bounding-box PPE annotations.
- Final classes: `Helmet`, `Gloves`, `Vest`, `Boots`, `Goggles`, `Person`.
- Removed logical-condition classes (`no_helmet`, `no_gloves`, `no_boots`) since they aren't physical objects.
- Remapped inconsistent label indices (e.g. `person: 6 → 5`) for continuous YOLO indexing.

### 2. Model Training (YOLOv8)
Configured via `data.yaml` (image paths, class count, class names):

| Parameter | Value |
|---|---|
| Model | YOLOv8m |
| Epochs | 20 |
| Image size | 640 |
| mAP50 | 0.841 |
| mAP50-95 | 0.467 |

### 3. Object Detection (`detect.py`)
Loads `best.pt` and returns raw bounding-box detections for person + each PPE class.

### 4. PPE Association (`associate.py`)
For every detected person, checks for nearby helmet / gloves / vest / boots / goggles and marks each `True`/`False`:

```json
[
  {
    "helmet": true,
    "gloves": false,
    "vest": true,
    "boots": false,
    "goggles": true
  }
]
```

### 5. Violation Engine (`violation_checker.py`)
Applies simple rules — any missing item becomes a violation:

```
If helmet missing  → "Missing helmet"
If boots missing   → "Missing boots"
...
```

### 6. API Layer (`api.py`)
Central controller — the bridge between frontend and backend:

```
Receive uploaded image
   → Save to uploads/
   → detect.py
   → associate.py
   → violation_checker.py
   → Save annotated image to outputs/
   → Log result to violations.json
   → Return results to frontend
```

### 7. Frontend (`frontend/`)
Built with React + Tailwind CSS. Features:
- Image upload
- Original image preview
- Annotated detection preview
- Grouped violation display

### 8. Logging (`violations.json`)
Every run is logged for future analytics:

```json
{
  "timestamp": "2026-07-03 11:34:11",
  "workers_detected": 1,
  "violations": ["Missing boots", "Missing goggles"]
}
```

---

## 🔗 File Interaction Map

| File | Purpose | Used By |
|---|---|---|
| `detect.py` | Runs YOLO model, returns raw detections | `api.py` |
| `associate.py` | Maps detected PPE to the correct worker | `api.py` |
| `violation_checker.py` | Flags missing PPE per worker | `api.py` |
| `api.py` | Orchestrates the full pipeline, main entry point | `frontend/App.jsx` |
| `frontend/` | User interface, talks to FastAPI | End user |
| `violations.json` | Historical detection log | `api.py` |
| `uploads/` | Stores input images | `api.py` |
| `outputs/` | Stores annotated result images | `detect.py`, frontend |

**Mental model:**
`Frontend` = interface · `api.py` = controller · `detect.py` = vision brain · `associate.py` = PPE mapper · `violation_checker.py` = rule engine · `violations.json` = evidence log · `outputs/` = visual proof

---

## ✅ Current Status

**Completed**
- [x] Dataset collection
- [x] Label cleaning
- [x] YOLO model training
- [x] Object detection
- [x] PPE association
- [x] Rule-based violation engine
- [x] FastAPI integration
- [x] React frontend
- [x] JSON logging

**Upcoming**
- [ ] Employee identity recognition (FaceNet)
- [ ] Database integration (PostgreSQL)

---

## 🚀 Setup & Run

### Backend
```bash
pip install ultralytics fastapi uvicorn python-multipart opencv-python numpy
```

### Frontend
```bash
npm install
npm install axios tailwindcss @tailwindcss/vite
```

### Run the project
```bash
run_project.bat
```
