from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import json
from datetime import datetime

from detect import detect_objects
from associate import associate_ppe
from violation_checker import check_violations

app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create folders
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Serve output images
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO detection
    results = detect_objects(file_path)

    # Save annotated output image
    output_path = f"outputs/{file.filename}"
    results[0].save(filename=output_path)

    # Process workers and violations
    workers = associate_ppe(results)
    violations = check_violations(workers)

    # Log to violations.json
    log_entry = {
        "timestamp": str(datetime.now()),
        "workers_detected": len(workers),
        "violations": violations
    }

    if os.path.exists("violations.json"):
        with open("violations.json", "r") as f:
            try:
                logs = json.load(f)
            except:
                logs = []
    else:
        logs = []

    logs.append(log_entry)

    with open("violations.json", "w") as f:
        json.dump(logs, f, indent=4)

    return {
        "workers": workers,
        "violations": violations,
        "annotated_image": f"http://127.0.0.1:8000/{output_path}"
    }