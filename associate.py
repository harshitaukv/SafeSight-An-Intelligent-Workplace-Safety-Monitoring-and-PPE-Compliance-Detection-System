def associate_ppe(detections):
    workers = []
    person_boxes = []
    ppe_boxes = []

    for r in detections:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].tolist()

            if cls == 5:
                person_boxes.append(coords)
            else:
                ppe_boxes.append((cls, coords))

    for idx, person in enumerate(person_boxes):
        px1, py1, px2, py2 = person

        worker = {
            "worker_id": idx + 1,
            "helmet": False,
            "gloves": False,
            "vest": False,
            "boots": False,
            "goggles": False
        }

        for cls, ppe in ppe_boxes:
            x1, y1, x2, y2 = ppe

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if px1 <= center_x <= px2 and py1 <= center_y <= py2:
                if cls == 0:
                    worker["helmet"] = True
                elif cls == 1:
                    worker["gloves"] = True
                elif cls == 2:
                    worker["vest"] = True
                elif cls == 3:
                    worker["boots"] = True
                elif cls == 4:
                    worker["goggles"] = True

        workers.append(worker)

    return workers