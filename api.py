from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import os
import json
import shutil
from datetime import datetime
from detect import detect_objects
from associate import associate_ppe
from violation_checker import check_violations

app = FastAPI(
    title="PPE Sentinel API",
    description="AI-Based Construction PPE Detection API",
    version="2.0.0"
)

# ----------------------------------------------------
# CORS
# ----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Create Required Folders
# ----------------------------------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------------------------------
# Serve Annotated Images
# ----------------------------------------------------
app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_FOLDER),
    name="outputs"
)

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads"
)

# ----------------------------------------------------
# Health Check
# ----------------------------------------------------
@app.get("/")
def home():
    return {
        "status": "running",
        "project": "PPE Sentinel",
        "version": "2.0.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# ----------------------------------------------------
# Multiple Image Detection Endpoint
# ----------------------------------------------------
@app.post("/detect")
async def detect(files: List[UploadFile] = File(...)):
    images = []
    summary = {
        "total_images": 0,
        "total_workers": 0,
        "safe_workers": 0,
        "unsafe_workers": 0,
        "helmet": 0,
        "vest": 0,
        "gloves": 0,
        "boots": 0,
        "goggles": 0,
        "compliance": 0
    }
    logs = []

    for file in files:
        # -----------------------------
        # Save Uploaded Image
        # -----------------------------
        upload_path = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # -----------------------------
        # Run YOLO Detection
        # -----------------------------
        results = detect_objects(upload_path)

        # -----------------------------
        # Save Annotated Image
        # -----------------------------
        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.filename
        )
        results[0].save(filename=output_path)

        # -----------------------------
        # Associate PPE with Workers
        # -----------------------------
        workers = associate_ppe(results)

        # -----------------------------
        # Check Violations / Build Worker Status
        # -----------------------------
        report = check_violations(workers)

        processed_workers = report["workers"]

        summary["total_workers"] += len(processed_workers)
        summary["safe_workers"] += report["summary"]["safe_workers"]
        summary["unsafe_workers"] += report["summary"]["unsafe_workers"]

        summary["helmet"] += report["summary"]["helmet"]
        summary["vest"] += report["summary"]["vest"]
        summary["gloves"] += report["summary"]["gloves"]
        summary["boots"] += report["summary"]["boots"]
        summary["goggles"] += report["summary"]["goggles"]

        summary["total_images"] += 1

        images.append(
            {
                "name": file.filename,
                "original": f"http://127.0.0.1:8000/uploads/{file.filename}",
                "annotated": f"http://127.0.0.1:8000/outputs/{file.filename}",
                "workers": processed_workers,
            }
        )

        logs.append(
            {
                "timestamp": str(datetime.now()),
                "image": file.filename,
                "workers": len(processed_workers),
                "unsafe": len(
                    [
                        w
                        for w in processed_workers
                        if w["status"] == "Unsafe"
                    ]
                ),
            }
        )

    # ----------------------------------------------------
    # Calculate Overall Compliance
    # ----------------------------------------------------
    total_required_ppe = summary["total_workers"] * 5
    total_missing_ppe = (
        summary["helmet"]
        + summary["vest"]
        + summary["gloves"]
        + summary["boots"]
        + summary["goggles"]
    )

    if total_required_ppe > 0:
        summary["compliance"] = round(
            ((total_required_ppe - total_missing_ppe)
             / total_required_ppe) * 100,
            2
        )
    else:
        summary["compliance"] = 100.0

    # ----------------------------------------------------
    # Save Detection History
    # ----------------------------------------------------
    history_file = "violations.json"

    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
    else:
        history = []

    history.append(
        {
            "timestamp": str(datetime.now()),
            "summary": summary,
            "images": logs
        }
    )

    with open(history_file, "w") as f:
        json.dump(history, f, indent=4)

    # ----------------------------------------------------
    # Return Response
    # ----------------------------------------------------
    return {
        "summary": summary,
        "images": images
    }