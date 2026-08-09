# ppe_geometry.py
"""
Plausibility rules for attaching a detected PPE box to a detected person.

The association step used to accept any PPE box whose *center* fell anywhere
inside a person's bounding box, at whatever confidence the detector emitted.
That is extremely permissive, and it is where most false positives came from:

  * a helmet sitting on a bench, a bin, or another worker's head still counts
    as long as it lands somewhere inside this person's box — and a person box
    is a large rectangle that includes plenty of background;
  * boots detected up at head height, or goggles down by the feet, are
    accepted without complaint, because vertical position is never checked;
  * a hi-vis vest being *held up* in front of the body scores exactly the
    same as one being worn;
  * a weak 0.50-confidence guess counts as much as a confident 0.95 one.

These rules add the three cheap checks a human reviewer applies instantly:
is the detection strong enough, is it actually on this person, and is it in
the right place on the body for what it claims to be. They are deliberately
independent of the model — no retraining required — and every threshold is
named and tunable here rather than buried in the association loop.

Trade-off worth stating plainly: every rule here can reject a real detection
(genuinely worn PPE in an odd pose, a crouching worker whose proportions
break the vertical bands). The bands are set wide enough to tolerate normal
posture, and any rejection is recoverable — the item simply needs to be
detected in a frame where the geometry is clearer.
"""

# Minimum detector confidence per class. Higher than the model-wide 0.50
# floor for the classes that produce the flimsiest boxes: gloves and goggles
# are small, low-texture, and easily hallucinated on hands/faces, whereas a
# helmet or vest that the model is only 55% sure about is usually still real.
MIN_CONFIDENCE = {
    "person": 0.55,
    "helmet": 0.55,
    "vest": 0.55,
    "gloves": 0.60,
    "boots": 0.60,
    "goggles": 0.60,
}

# Fraction of the PPE box that must fall inside the person box. A helmet
# actually on someone's head is almost entirely within their silhouette;
# an object merely *near* them clips the edge of it.
MIN_CONTAINMENT = 0.6

# Where each item is expected to sit on the body, as a fraction of the
# person box height measured from the top (0.0 = crown, 1.0 = feet), plus
# the largest that item can plausibly be relative to the person box.
#
# The bands are generous on purpose — bending, crouching, sitting and camera
# angle all shift these — but they still make "boots at head height" or
# "helmet at the knees" impossible.
PPE_ITEM_RULES = {
    "helmet":  {"band": (0.00, 0.35), "max_h": 0.35, "max_w": 0.90},
    "goggles": {"band": (0.00, 0.35), "max_h": 0.22, "max_w": 0.70},
    "vest":    {"band": (0.12, 0.75), "max_h": 0.75, "max_w": 1.00},
    "gloves":  {"band": (0.25, 0.95), "max_h": 0.35, "max_w": 0.60},
    "boots":   {"band": (0.60, 1.05), "max_h": 0.35, "max_w": 0.90},
}


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def containment(inner, outer):
    """Fraction of `inner` that lies inside `outer`."""
    area = box_area(inner)
    if area <= 0:
        return 0.0
    return intersection_area(inner, outer) / area


def relative_position(ppe_box, person_box):
    """
    Vertical position of the PPE box's center within the person box, as a
    fraction of the person's height (0.0 = top edge, 1.0 = bottom edge).
    Values outside 0–1 mean the center sits above/below the person box.
    """
    _, py1, _, py2 = person_box
    height = py2 - py1
    if height <= 0:
        return None
    center_y = (ppe_box[1] + ppe_box[3]) / 2
    return (center_y - py1) / height


def size_ratios(ppe_box, person_box):
    px1, py1, px2, py2 = person_box
    person_w = px2 - px1
    person_h = py2 - py1
    if person_w <= 0 or person_h <= 0:
        return None
    return (
        (ppe_box[2] - ppe_box[0]) / person_w,
        (ppe_box[3] - ppe_box[1]) / person_h,
    )


def association_score(item, ppe_box, confidence, person_box):
    """
    How well a PPE box fits being worn by this person.

    Returns a score in (0, 1] — higher is a better fit — or None when the
    pairing fails a hard check and should never be made. The score exists so
    that when several people are candidates for the same item, it goes to
    the one it actually sits on rather than whichever the loop reached first.
    """
    rules = PPE_ITEM_RULES.get(item)
    if rules is None:
        return None

    if confidence is not None and confidence < MIN_CONFIDENCE.get(item, 0.5):
        return None

    inside = containment(ppe_box, person_box)
    if inside < MIN_CONTAINMENT:
        return None

    position = relative_position(ppe_box, person_box)
    if position is None:
        return None

    low, high = rules["band"]
    if not (low <= position <= high):
        return None

    ratios = size_ratios(ppe_box, person_box)
    if ratios is None:
        return None
    width_ratio, height_ratio = ratios
    if width_ratio > rules["max_w"] or height_ratio > rules["max_h"]:
        return None

    # Within the band, fit degrades with distance from its center: a helmet
    # right at the crown beats one hovering at the band's lower edge.
    band_center = (low + high) / 2
    band_half = max(1e-6, (high - low) / 2)
    band_fit = 1.0 - min(1.0, abs(position - band_center) / band_half)

    conf_component = confidence if confidence is not None else 0.75

    # Containment and confidence carry the most weight; band fit breaks ties
    # between two people who both plausibly own the item.
    return round(0.45 * inside + 0.35 * conf_component + 0.20 * band_fit, 6)
