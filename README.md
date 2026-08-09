# 🛡️ SafeSight

> An Intelligent Workplace Safety Monitoring and PPE Compliance Detection System

![safety](https://img.shields.io/badge/Project-SafeSight-blueviolet) ![python](https://img.shields.io/badge/Made%20with-Python-3776AB) ![status](https://img.shields.io/badge/Status-Experimental-yellow)

---

🌈 Welcome to SafeSight — where computer vision meets workplace safety. This repository contains a full-stack, research-to-deployment pipeline for detecting Personal Protective Equipment (PPE) compliance (helmets, vests, masks, gloves), unsafe behaviors, and potential hazards in workplace video feeds and images.

Why SafeSight?
- Reduce accidents with automated monitoring.
- Improve compliance reporting and audits.
- Enable real-time alerts to supervisors and linked systems.

✨ Highlights
- PPE detection models (helmet, vest, gloves, mask)
- Person detection + pose-aware heuristics for unsafe behavior
- Inference scripts for images and video streams (OpenCV)
- Dashboard-ready outputs (JSON, annotated frames)

---

Table of Contents
- Features
- Demo (Quick look)
- Quick Start
- Usage
- Project Structure
- Models & Data
- Contributing
- License & Contact

---

Features
- 🎯 Multi-class PPE detection (helmet, safety vest, mask, gloves)
- 🧭 Bounding-box + Confidence scores for each detection
- 🖼️ Image and video processing pipelines (support for recorded files and RTSP streams)
- 🔔 Simple alerting hooks (webhooks / console / logger)
- ⚡ Lightweight inference for edge deployment

Demo (Quick look)
- Run the inference script on a sample image and get an annotated image with detected PPE and compliance status.

Quick Start
1. Clone the repo

```bash
git clone https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System.git
cd SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System
```

2. Create a Python virtual environment and install requirements

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

3. Run inference on an example image

```bash
python run_inference.py --input assets/examples/site_photo.jpg --output out/annotated.jpg
```

Usage
- Training: If training scripts are included, follow scripts in `training/` (README inside training folder might exist).
- Inference (image): `python run_inference.py --input path/to/image.jpg --output out/annotated.jpg`
- Inference (video/stream): `python run_inference.py --input rtsp://<camera> --output out/video_out.mp4 --stream` (check flags)
- Configuration: Edit `config.yaml` to set thresholds, class names, and model paths.

Project Structure
- assets/ — example images, sample videos
- models/ — pretrained weights and exported artifacts (not committed due to size)
- src/ — core code: detection, utils, inference, visualization
- training/ — training configs, notebooks, augmentation scripts
- requirements.txt — Python dependencies

Models & Data
- This repo uses (or supports) common object-detection backbones (YOLO, Faster R-CNN, MobileNet-SSD). Provide your own weights in `models/` and set paths in `config.yaml`.
- If you need sample datasets or annotations, check `assets/` or contact the maintainer for links.

Tips for better results
- Use domain-specific finetuning (site images, angles, lighting).
- Balance datasets — include both compliant and non-compliant examples.
- Calibrate confidence and NMS thresholds in `config.yaml`.

Contributing
We welcome contributions! Please:
1. Fork the repo
2. Create a feature branch (feature/your-feature)
3. Open a PR with a clear description of changes
4. Add tests or examples for new features

Please follow the code style and include a short explanation of how to test your changes.

License
- MIT License — see LICENSE file for details

Contact & Acknowledgements
- Author: harshitaukv
- Repo: https://github.com/harshitaukv/SafeSight-An-Intelligent-Workplace-Safety-Monitoring-and-PPE-Compliance-Detection-System

If you'd like a themed logo, demo notebook, or automated CI for model validation added, tell me which you'd prefer and I'll add it next. Stay safe and happy coding! 🚧👷‍♀️👷‍♂️
