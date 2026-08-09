# risk_engine.py

def assess_risk(missing_ppe):
    """
    Determine risk level based on missing PPE.
    """

    missing = [x.lower() for x in missing_ppe]

    high = {"helmet"}
    medium = {"goggles", "gloves", "boots"}
    low = {"vest"}

    if any(item in high for item in missing):
        return "HIGH"

    if any(item in medium for item in missing):
        return "MEDIUM"

    if any(item in low for item in missing):
        return "LOW"

    return "SAFE"


def get_recommendations(missing_ppe):
    """
    Generate corrective actions.
    """

    recommendations = []

    for item in missing_ppe:

        item = item.lower()

        if item == "helmet":
            recommendations.append(
                "Provide an IS-certified safety helmet before allowing work."
            )

        elif item == "vest":
            recommendations.append(
                "Ensure the worker wears a high-visibility safety vest."
            )

        elif item == "gloves":
            recommendations.append(
                "Provide appropriate protective gloves."
            )

        elif item == "boots":
            recommendations.append(
                "Wear steel-toe safety boots."
            )

        elif item == "goggles":
            recommendations.append(
                "Wear protective safety goggles."
            )

    if not recommendations:
        recommendations.append(
            "No corrective action required."
        )

    return recommendations