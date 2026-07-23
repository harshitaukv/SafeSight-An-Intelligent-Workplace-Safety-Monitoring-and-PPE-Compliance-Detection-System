import math


# Class IDs
HELMET = 0
GLOVES = 1
VEST = 2
BOOTS = 3
GOGGLES = 4
PERSON = 5


def box_center(box):
    x1, y1, x2, y2 = box
    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def inside_person(center, person_box):
    cx, cy = center
    x1, y1, x2, y2 = person_box

    return (
        x1 <= cx <= x2 and
        y1 <= cy <= y2
    )


def distance(c1, c2):
    return math.sqrt(
        (c1[0] - c2[0]) ** 2 +
        (c1[1] - c2[1]) ** 2
    )


def associate_ppe(results):

    people = []
    ppe_objects = []

    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])

            coords = box.xyxy[0].tolist()

            if cls == PERSON:

                people.append(coords)

            else:

                ppe_objects.append(
                    {
                        "class": cls,
                        "box": coords
                    }
                )

    workers = []

    for index, person in enumerate(people):

        workers.append(
            {
                "worker_id": index + 1,
                "person_box": person,
                "helmet": False,
                "vest": False,
                "gloves": False,
                "boots": False,
                "goggles": False,
            }
        )

    used = set()

    for ppe_index, obj in enumerate(ppe_objects):

        center = box_center(obj["box"])

        nearest_worker = None

        nearest_distance = float("inf")

        for worker in workers:

            if not inside_person(
                center,
                worker["person_box"]
            ):
                continue

            person_center = box_center(
                worker["person_box"]
            )

            d = distance(
                center,
                person_center
            )

            if d < nearest_distance:

                nearest_distance = d

                nearest_worker = worker

        if nearest_worker is None:
            continue

        if ppe_index in used:
            continue

        used.add(ppe_index)

        cls = obj["class"]

        if cls == HELMET:
            nearest_worker["helmet"] = True

        elif cls == VEST:
            nearest_worker["vest"] = True

        elif cls == GLOVES:
            nearest_worker["gloves"] = True

        elif cls == BOOTS:
            nearest_worker["boots"] = True

        elif cls == GOGGLES:
            nearest_worker["goggles"] = True

    for worker in workers:
        worker.pop("person_box")

    return workers