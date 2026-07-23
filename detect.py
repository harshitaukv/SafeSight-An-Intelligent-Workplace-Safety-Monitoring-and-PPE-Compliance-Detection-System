from ultralytics import YOLO
import torch
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "runs",
    "train_m",
    "weights",
    "best.pt"
)

CONFIDENCE = 0.50
IOU = 0.45

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Load Model (Only Once)
# --------------------------------------------------

model = YOLO(MODEL_PATH)

print(f"Model Loaded: {MODEL_PATH}")
print(f"Running on: {DEVICE}")

# --------------------------------------------------
# Detection Function
# --------------------------------------------------

def detect_objects(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    results = model.predict(
        source=image_path,
        conf=CONFIDENCE,
        iou=IOU,
        device=DEVICE,
        verbose=False
    )

    return results