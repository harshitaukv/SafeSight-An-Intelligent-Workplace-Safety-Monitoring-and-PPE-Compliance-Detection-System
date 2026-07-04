from ultralytics import YOLO

# Load trained model
model = YOLO(r"A:\PPE\runs\train_m\weights\best.pt")

def detect_objects(image_path):
    # Higher confidence threshold
    results = model(image_path, conf=0.5, save=True)

    # Print detections for debugging
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            print(f"Class: {cls}, Confidence: {round(conf, 2)}, Box: {coords}")

    return results