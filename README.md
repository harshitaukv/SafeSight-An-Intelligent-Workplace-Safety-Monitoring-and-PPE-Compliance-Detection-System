# 🦺 SafeSight: AI-Based PPE Compliance & Workplace Safety Monitoring

<div align="center">

![SafeSight Banner](https://img.shields.io/badge/SafeSight-Workplace%20Safety-orange?style=for-the-badge&logo=university)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18%2B-61dafb?logo=react&logoColor=white)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-red?logo=yolo&logoColor=white)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end **computer vision system** that detects PPE compliance in construction workers and flags safety violations in real-time. Powered by YOLOv8, FastAPI, and React.

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Workflow](#-workflow) • [Documentation](#-documentation)

</div>


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

### System Design (flowchart)

```mermaid
flowchart TD
    A[Input: Image / Video Frame]
    B[YOLOv8 Object Detection\n(Detects Person + PPE Items)]
    C[PPE Association Engine\n(Maps PPE → Workers)]
    D[Violation Checker\n(Rule-based compliance engine)]
    E[FastAPI Controller\n(REST API Layer)]
    F[React Frontend Dashboard\n(User interface & visualization)]
    G[JSON Logging & Database\n(Audit trail & analytics)]

    A --> B --> C --> D --> E --> F
    E --> G
    D --> G
```

> The architecture flowchart above replaces the older ASCII diagram with a colorful, interactive mermaid flowchart (rendered by GitHub's markdown renderer). It makes the data/processing flow easier to scan.

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

#### 1. **Image Upload Flow** (flowchart)

```mermaid
flowchart TD
    U[User Upload (Frontend)] --> POST[POST /api/detect (FastAPI)]
    POST --> Save[Save to uploads/ folder]
    Save --> Inference[YOLO Inference (detect.py)]
    Inference --> BBoxes[Get bounding boxes + confidence scores]
    BBoxes --> Associate[PPE Association (associate.py)]
    Associate --> Map[Map each PPE item to nearest worker]
    Map --> Checker[Violation Checking (violation_checker.py)]
    Checker --> Report[Generate compliance report]
    Report --> Annotate[Save annotated image to outputs/]
    Annotate --> Log[Log to violations.json]
    Log --> Response[Return JSON response to frontend]
    Response --> UI[Display results to user]
```

The mermaid flowchart above visualizes the end-to-end processing pipeline and is easier to read than a text-only diagram.

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

---

(Other sections such as API Reference, Training, Troubleshooting, Contributing, Roadmap, License, Author, and References remain unchanged.)

---

<div align="center">

**⭐ If this project helped you, please consider giving it a star on GitHub!**

Made with ❤️ for workplace safety

</div>
