# DiceWatcher

Real-time dice recognition and Warhammer 40,000 mechanics engine

---

## Overview
DiceWatcher is a computer-vision application for recognizing dice faces in real time and applying tabletop game mechanics such as hit, wound, save, and damage resolution.

It consists of:
- YOLOv8-based detector that locates dice and classifies visible faces (1–6).
- FastAPI backend providing a `/predict` endpoint returning annotated frames and JSON results.
- Streamlit web application for live webcam streaming, rule selection, and phase management.
- Mechanics and rule engines reproducing core Warhammer 40,000 combat logic.

A short demonstration video is available here:  
[Demo video on Google Drive](https://drive.google.com/file/d/10BYVVmny9c7KtTDKZ7CavrLh3ipLILNw/view?usp=sharing)

---

## Features

- Live camera detection
  - Works with USB webcams (tested on Aoni A25).
  - Automatic frame stabilization to avoid double detections.
- Static image upload
  - Processes images directly through the backend endpoint.
- Phase-based workflow
  - Separate buffers for Hit, Wound, and Save dice.
  - Optional Re-roll hit 1s step.
- Mechanics evaluation
  - Calculates hits, wounds, failed saves, and damage using a JSON unit profile.
- Config upload
  - Upload custom JSON unit files describing Ballistic Skill, Strength, AP, Damage, etc.
- Dockerized architecture
  - Independent containers for API (`dice_api`) and web app (`dice_app`).

---

## Installation

### Requirements
- Python 3.10+
- Recommended: Linux environment with webcam support (v4l2).

### Local setup

```bash
git clone https://github.com/NikitaMensh/DiceWatcher.git
cd DiceWatcher

# Backend
pip install -r requirements_api.txt
uvicorn api.detection:app --reload --host 0.0.0.0 --port 8000

# Frontend
pip install -r requirements_app.txt
streamlit run app/streamlit_app.py
````

### Docker

```bash
docker compose up --build
```

Note: On Linux, grant webcam access:

```bash
docker compose run --device /dev/video0:/dev/video2 dice_app
```

---

## Example Unit Configuration

```json
{
  "unit": "Skitarii Rangers",
  "models": 10,
  "bs": 4,
  "weapons": [
    {
      "name": "Galvanic Rifle",
      "type": "Rapid Fire",
      "attacks_per_model": 1,
      "rapid_fire_multiplier": 2,
      "strength": 4,
      "ap": 0,
      "damage": 1
    }
  ],
  "abilities": {
    "reroll_hit_ones": false,
    "hit_mod": 0,
    "wound_mod": 0
  }
}
```

Upload this JSON through the web app sidebar to compute damage results.

---

## Current Results

* mAP@0.5 = 0.99 for opaque dice.
* Real-time processing at 15–20 FPS on CPU.
* Stable integration of live inference, rule engine, and game mechanics.

---

## Known Issues

* Transparent dice produce unstable reflections and low contrast; fine-tuning planned.
* When using Docker on macOS/Windows, direct webcam access is limited.
* Streamlit event loop may block when running long camera sessions on some systems.

---

## Planned Work

* Improve recognition of transparent dice.
* Add new mechanics: mortal wounds, exploding 6s, weapon-specific rules.
* Implement full battle simulation for multi-unit combat probability analysis.

---

## Author

Nikita Menshikov
Innopolis University
[n.menshikov@innopolis.university](mailto:n.menshikov@innopolis.university)

---

## License

MIT License © 2025 Nikita Menshikov
