# 🦺 SafeSight: AI-Based PPE Compliance & Workplace Safety Monitoring

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-61dafb?logo=react&logoColor=white)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-red?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **computer vision system** that detects PPE compliance in construction workers and flags safety violations in real-time. Powered by YOLOv8, FastAPI, and React.

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Workflow](#-workflow) • [Documentation](#-documentation)

</div>
---
https://drive.google.com/file/d/1iPd65zUjqJyKQoGVGnYFKQ0CZrW_H_n0/view?usp=sharing


## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Detection Model** | YOLOv8m |
| **mAP50** | 0.841 |
| **mAP50-95** | 0.467 |
| **Inference Backend** | FastAPI |
| **Frontend** | React + Tailwind CSS |
| **Status** | ✅ Production-Ready |

---

## ✨ Features

### 🎯 PPE Detection
Detects and validates the following safety equipment:
- **Helmet** - Hard hat protection
- **Gloves** - Hand protection
- **Safety Vest** - Visibility & protection
- **Boots** - Foot protection
- **Goggles** - Eye protection
- **Person** - Worker identification

### 🔍 Core Capabilities
- ✅ Real-time detection from images/video frames
- ✅ Worker-to-PPE association logic
- ✅ Automated violation detection
- ✅ Historical logging & audit trails
- ✅ REST API for seamless integration
- ✅ Responsive web dashboard

### 📈 Safety Insights
- Violation flagging with detailed reason codes
- Worker compliance statistics
- Historical violation trends
- Exportable compliance reports

---

## 🏗️ Architecture

### System Design
```
┌─────────────────────────────────────────────────────────┐
│                   SAFESIGHT PIPELINE                    │
└─────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   Input: Image/Video Frame            │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   YOLOv8 Object Detection             │
        │   (Detects Person + PPE Items)        │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   PPE Association Engine              │
        │   (Maps PPE → Workers)                │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   Violation Checker                   │
        │   (Rule-based compliance engine)      │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   FastAPI Controller                  │
        │   (REST API Layer)                    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   React Frontend Dashboard            │
        │   (User interface & visualization)    │
        └───────────────────────────────────────┘
                            ↓
        ┌───────────────────────────────────────┐
        │   JSON Logging & Database             │
        │   (Audit trail & analytics)           │
        └───────────────────────────────────────┘
```

### Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Detection Engine** | YOLOv8m | Latest |
| **Vision Library** | OpenCV | 4.5+ |
| **Backend Framework** | FastAPI | 0.95+ |
| **Server** | Uvicorn | 0.21+ |
| **Frontend Framework** | React | 18+ |
| **Styling** | Tailwind CSS | 3+ |
| **HTTP Client** | Axios | 1.3+ |
| **Language** | Python 3.9+ | - |

---

## 📂 Project Structure

```
SafeSight/
│
├── 📄 README.md                          # Project documentation
├── 📄 requirements.txt                   # Python dependencies
├── 🔧 run_project.bat                    # Windows batch runner
│
├── 🐍 Backend (Python/FastAPI)
│   ├── detect.py                         # YOLO inference engine
│   ├── associate.py                      # PPE-to-worker mapping
│   ├── violation_checker.py              # Compliance rule engine
│   ├── api.py                            # FastAPI main controller
│   ├── config.py                         # Configuration settings
│   ├── utils.py                          # Helper functions
│   │
│   ├── 📁 models/
│   │   └── best.pt                       # YOLOv8m trained weights
│   │
│   ├── 📁 data/
│   │   ├── data.yaml                     # Dataset configuration
│   │   └── [training data]               # Kaggle dataset
│   │
│   ├── 📁 uploads/                       # User-uploaded images
│   ├── 📁 outputs/                       # Annotated result images
│   ├── 📁 runs/                          # YOLO training runs
│   └── 📁 logs/
│       └── violations.json               # Historical detection log
│
├── ⚛️ Frontend (React)
│   ├── 📄 package.json                   # Node.js dependencies
│   ├── 📄 tailwind.config.js             # Tailwind CSS config
│   ├── 📄 vite.config.js                 # Vite build config
│   │
│   ├── 📁 src/
│   │   ├── main.jsx                      # React entry point
│   │   ├── App.jsx                       # Root component
│   │   ├── index.css                     # Global styles
│   │   │
│   │   ├── 📁 components/
│   │   │   ├── ImageUpload.jsx           # Upload component
│   │   │   ├── Dashboard.jsx             # Main dashboard
│   │   │   ├── ResultsDisplay.jsx        # Results viewer
│   │   │   └── ViolationsList.jsx        # Violations panel
│   │   │
│   │   ├── 📁 services/
│   │   │   └── api.js                    # API client (Axios)
│   │   │
│   │   └── 📁 utils/
│   │       └── helpers.js                # Utility functions
│   │
│   └── 📁 public/                        # Static assets
│
└── 📋 Workflow Documents
    ├── WORKFLOW.md                       # Development workflow
    ├── CONTRIBUTING.md                   # Contribution guidelines
    └── API_DOCS.md                       # API documentation
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9** or higher
- **Node.js 16** or higher
- **npm** or **yarn**
- **Git**

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System.git
cd SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System
```

### 2️⃣ Backend Setup

#### Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Required Packages
```bash
pip install ultralytics fastapi uvicorn python-multipart opencv-python numpy pillow
```

#### Verify Installation
```bash
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
```

### 3️⃣ Frontend Setup

```bash
cd frontend

# Install Node dependencies
npm install

# Install additional packages
npm install axios tailwindcss @tailwindcss/vite

# Verify installation
npm list
```

### 4️⃣ Run the Project

#### Option A: Automatic (Batch File - Windows Only)
```bash
# From project root
run_project.bat
```

#### Option B: Manual Setup

**Terminal 1 - Backend:**
```bash
# From project root
python api.py
```
Backend runs on `http://localhost:8000`

**Terminal 2 - Frontend:**
```bash
# From frontend directory
cd frontend
npm run dev
```
Frontend runs on `http://localhost:5173` (or specified port)

### 5️⃣ Verify Everything Works

- **Backend API:** Visit `http://localhost:8000/docs` (Swagger UI)
- **Frontend:** Visit `http://localhost:5173`
- Try uploading an image to test the pipeline

---

## 📊 Workflow

### Development Workflow

#### 1. **Image Upload Flow**
```
User Upload (Frontend)
        ↓
POST /api/detect (FastAPI)
        ↓
Save to uploads/ folder
        ↓
YOLO Inference (detect.py)
        ↓
Get bounding boxes + confidence scores
        ↓
PPE Association (associate.py)
        ↓
Map each PPE item to nearest worker
        ↓
Violation Checking (violation_checker.py)
        ↓
Generate compliance report
        ↓
Save annotated image to outputs/
        ↓
Log to violations.json
        ↓
Return JSON response to frontend
        ↓
Display results to user
```

#### 2. **Detection Pipeline**

**detect.py** - YOLO Inference
```python
def detect_objects(image_path, confidence=0.5):
    """
    Runs YOLOv8m on input image
    Returns: list of detections with coordinates
    """
    model = YOLO("models/best.pt")
    results = model.predict(image_path, conf=confidence)
    return parse_detections(results)
```

**associate.py** - Worker-PPE Mapping
```python
def associate_ppe_to_workers(persons, ppe_items):
    """
    For each person, checks proximity to PPE items
    Returns: dict mapping worker_id → {helmet: bool, gloves: bool, ...}
    """
    associations = []
    for person in persons:
        ppe_status = {
            "helmet": has_nearby_ppe(person, ppe_items, "helmet"),
            "gloves": has_nearby_ppe(person, ppe_items, "gloves"),
            "vest": has_nearby_ppe(person, ppe_items, "vest"),
            "boots": has_nearby_ppe(person, ppe_items, "boots"),
            "goggles": has_nearby_ppe(person, ppe_items, "goggles"),
        }
        associations.append(ppe_status)
    return associations
```

**violation_checker.py** - Compliance Engine
```python
def check_violations(ppe_associations):
    """
    Applies safety rules to worker PPE status
    Returns: list of violations per worker
    """
    violations = []
    required_ppe = ["helmet", "gloves", "vest", "boots"]
    
    for worker_idx, ppe_status in enumerate(ppe_associations):
        worker_violations = []
        for equipment in required_ppe:
            if not ppe_status.get(equipment, False):
                worker_violations.append(f"Missing {equipment}")
        violations.append(worker_violations)
    
    return violations
```

**api.py** - REST Endpoint
```python
@app.post("/api/detect")
async def detect_ppe(file: UploadFile = File(...)):
    """
    Main endpoint orchestrating the full pipeline
    """
    # Save uploaded image
    image_path = f"uploads/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    # Run detection pipeline
    detections = detect_objects(image_path)
    ppe_associations = associate_ppe_to_workers(detections)
    violations = check_violations(ppe_associations)
    
    # Save annotated image
    annotated_path = f"outputs/annotated_{file.filename}"
    save_annotated_image(image_path, detections, annotated_path)
    
    # Log results
    log_to_violations_file({
        "timestamp": datetime.now().isoformat(),
        "workers_detected": len(ppe_associations),
        "violations": violations
    })
    
    # Return response
    return {
        "status": "success",
        "workers_detected": len(ppe_associations),
        "violations": violations,
        "annotated_image_url": f"/outputs/annotated_{file.filename}"
    }
```

#### 3. **Data Flow - Sample JSON**

**Input PPE Association:**
```json
[
  {
    "worker_id": 0,
    "helmet": true,
    "gloves": false,
    "vest": true,
    "boots": false,
    "goggles": true,
    "confidence": 0.94
  },
  {
    "worker_id": 1,
    "helmet": true,
    "gloves": true,
    "vest": true,
    "boots": true,
    "goggles": false,
    "confidence": 0.87
  }
]
```

**Output Violations:**
```json
{
  "timestamp": "2026-07-05T10:30:45.123456",
  "image_filename": "construction_site_001.jpg",
  "workers_detected": 2,
  "violations": [
    {
      "worker_id": 0,
      "violations": ["Missing gloves", "Missing boots"],
      "compliance_score": 0.6
    },
    {
      "worker_id": 1,
      "violations": ["Missing goggles"],
      "compliance_score": 0.8
    }
  ],
  "overall_compliance": 0.7
}
```

#### 4. **Logging & Audit Trail**

All detections logged to `logs/violations.json`:
```json
{
  "detection_id": "det_20260705_103045",
  "timestamp": "2026-07-05T10:30:45",
  "location": "construction_site_001",
  "workers_detected": 2,
  "violations_found": 2,
  "violation_details": ["Missing gloves", "Missing boots", "Missing goggles"],
  "annotated_image": "outputs/annotated_construction_site_001.jpg",
  "processed_by": "YOLOv8m",
  "inference_time_ms": 1240
}
```

---

## 🔌 API Reference

### Base URL
```
http://localhost:8000
```

### Authentication
Not required for Module 1 (upcoming in Module 2)

### Endpoints

#### 1. **Health Check**
```http
GET /health
```
**Response:**
```json
{ "status": "healthy", "version": "1.0.0" }
```

#### 2. **Detect PPE Compliance** ⭐ Main Endpoint
```http
POST /api/detect
Content-Type: multipart/form-data

file: <image_file>
confidence: 0.5 (optional, default: 0.5)
```

**Request Example:**
```bash
curl -X POST "http://localhost:8000/api/detect" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

**Response (200 OK):**
```json
{
  "status": "success",
  "workers_detected": 2,
  "detections": [
    {
      "worker_id": 0,
      "helmet": true,
      "gloves": false,
      "vest": true,
      "boots": false,
      "goggles": true
    }
  ],
  "violations": [
    {
      "worker_id": 0,
      "violations": ["Missing gloves", "Missing boots"]
    }
  ],
  "annotated_image_url": "/outputs/annotated_image.jpg",
  "processing_time_ms": 1240
}
```

#### 3. **Get Detection History**
```http
GET /api/violations/history
```

**Response:**
```json
{
  "total_detections": 42,
  "detections": [...]
}
```

#### 4. **Get API Documentation**
```http
GET /docs
```
Auto-generated Swagger UI documentation

---

## 🛠️ Development Guide

### Adding New PPE Types

**Step 1:** Update `violation_checker.py`
```python
REQUIRED_PPE = ["helmet", "gloves", "vest", "boots", "goggles", "respirator"]
```

**Step 2:** Retrain YOLO model with new class
```bash
# Update data.yaml with new class
# Retrain model
yolo detect train data=data.yaml model=yolov8m.pt epochs=20 img=640
```

**Step 3:** Update frontend components
```javascript
// In frontend/src/components/ViolationsList.jsx
const PPE_NAMES = {
  helmet: "Safety Helmet",
  gloves: "Work Gloves",
  // ... add new PPE
};
```

### Modifying Violation Rules

Edit `violation_checker.py`:
```python
def check_violations(ppe_associations):
    violations = []
    
    # Example: Make goggles optional in some areas
    required_ppe = ["helmet", "gloves", "vest", "boots"]  # goggles optional
    optional_ppe = ["goggles"]
    
    for worker_idx, ppe_status in enumerate(ppe_associations):
        # Custom logic here
        pass
    
    return violations
```

### Improving Detection Accuracy

1. **Collect more training data** from construction sites
2. **Retrain YOLOv8 model:**
   ```bash
   yolo detect train data=data.yaml model=yolov8l.pt epochs=50 img=640
   ```
3. **Adjust confidence threshold** in `api.py`
4. **Fine-tune PPE association logic** in `associate.py`

---

## 📊 Model Training Details

### Dataset Information
- **Source:** Kaggle Construction Worker PPE Detection Dataset
- **Total Images:** 6,000+
- **Classes:** 6 (Person, Helmet, Gloves, Vest, Boots, Goggles)
- **Annotations:** YOLO format (normalized bbox + class)
- **Train/Val/Test Split:** 70% / 15% / 15%

### Training Configuration
```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
nc: 6
names: ['person', 'helmet', 'gloves', 'vest', 'boots', 'goggles']
```

### Training Command
```bash
yolo detect train data=data/data.yaml model=yolov8m.pt epochs=20 img=640 device=0
```

### Performance Metrics
| Metric | Value |
|--------|-------|
| mAP50 | 0.841 |
| mAP50-95 | 0.467 |
| Precision | 0.89 |
| Recall | 0.81 |
| Inference Time | ~30ms per image |

---

## 🐛 Troubleshooting

### Common Issues

#### 1. **Backend won't start: Module not found**
```
Error: ModuleNotFoundError: No module named 'ultralytics'
```
**Solution:**
```bash
pip install ultralytics opencv-python fastapi uvicorn
```

#### 2. **CUDA/GPU not detected**
```python
# api.py
# Force CPU usage
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

#### 3. **Frontend can't reach backend**
- Ensure backend is running on `http://localhost:8000`
- Check CORS settings in `api.py`
- Verify firewall allows localhost:8000

#### 4. **Image upload fails: Permission denied**
```bash
# Ensure write permissions
chmod -R 755 uploads/ outputs/ logs/
```

#### 5. **Model weights not found: FileNotFoundError**
```bash
# Download model weights
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** with clear messages: `git commit -m "feat: add your feature"`
4. **Push** to your fork: `git push origin feature/your-feature`
5. **Open** a Pull Request with description

### Code Style
- Follow PEP 8 for Python
- Use meaningful variable names
- Add docstrings to functions
- Test before submitting PR

---

## 📈 Roadmap

### ✅ Module 1: PPE Detection (Current)
- [x] YOLOv8 object detection
- [x] PPE association logic
- [x] Violation detection engine
- [x] FastAPI REST API
- [x] React frontend dashboard

### 📅 Module 2: Employee Recognition
- [ ] FaceNet for worker identification
- [ ] Employee database integration
- [ ] Individual compliance tracking
- [ ] Per-worker safety reports

### 📅 Module 3: Database & Analytics
- [ ] PostgreSQL integration
- [ ] Analytics dashboard
- [ ] Historical trend analysis
- [ ] Compliance reporting system

### 📅 Module 4: Video Processing
- [ ] Real-time video stream processing
- [ ] RTSP/RTMP support
- [ ] Multi-camera coordination
- [ ] Alert/notification system

---

## 📝 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Harshita Ukv**
- GitHub: [@harshitaukv](https://github.com/harshitaukv)
- Project: [SafeSight Repository](https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System)

---

## 💬 Support & Contact

- **Issues:** [GitHub Issues](https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System/issues)
- **Discussions:** [GitHub Discussions](https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System/discussions)
- **Email:** Available in GitHub profile

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Official Docs](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS Docs](https://tailwindcss.com/)
- [OpenCV Python Documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star on GitHub!**

Made with ❤️ for workplace safety

</div>
