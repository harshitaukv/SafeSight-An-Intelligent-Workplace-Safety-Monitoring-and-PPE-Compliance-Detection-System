import streamlit as st
from detect import detect_objects
from associate import associate_ppe
from violation_checker import check_violations
import json
from datetime import datetime

st.title("PPE Sentinel")

uploaded_file = st.file_uploader("Upload Construction Image")

if uploaded_file:
    image_path = "temp.jpg"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(image_path)

    if st.button("Run Detection"):
        # Detection
        detections = detect_objects(image_path)

        # PPE Association
        workers = associate_ppe(detections)

        # Violation Check
        violations = check_violations(workers)

        # Show violations
        st.write("Violations:")
        for v in violations:
            st.write(v)

        # Save logs to violations.json
        log_data = {
            "timestamp": str(datetime.now()),
            "image_path": image_path,
            "workers_detected": len(workers),
            "violations": violations
        }

        with open("violations.json", "a") as f:
            json.dump(log_data, f)
            f.write("\n")

        st.success("Violations logged successfully.")