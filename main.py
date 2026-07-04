import json
from datetime import datetime

# Extract detection details
detection_logs = []

for r in detections:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        coords = box.xyxy[0].tolist()

        detection_logs.append({
            "class_id": cls,
            "confidence": round(conf, 3),
            "bbox": coords
        })

# Save enhanced log
log_data = {
    "timestamp": str(datetime.now()),
    "image_path": image_path,
    "annotated_image_path": "runs/detect/predict/image1.jpg",
    "workers_detected": len(workers),
    "violation_count": len(violations) if violations != ["No violations"] else 0,
    "violations": violations,
    "detections": detection_logs
}

with open("violations.json", "a") as f:
    json.dump(log_data, f, indent=4)
    f.write("\n")