# DiceWatcher

DiceWatcher is a computer vision system for real-time detection, counting, and evaluation of dice throws.  
It is designed for tabletop wargames, where players often roll dozens of dice and apply complex rules. The system automates dice recognition and rule evaluation, letting players focus on gameplay.

## Features
- Webcam input with real-time processing (≥25 FPS target).
- Detection and counting of multiple dice per frame.
- Pip recognition (1–6) on standard six-sided dice.
- Rule-based evaluation (e.g., thresholds for hits, conditional rerolls).
- Visualization with bounding boxes and overlays on the video stream.

## Installation
### Requirements
- Python 3.9+
- PyTorch
- OpenCV
- Ultralytics YOLOv8 (or another object detection framework)
- Other dependencies in `requirements.txt`

### Setup
```bash
git clone https://github.com/NikitaMensh/DiceWatcher.git
cd DiceWatcher
pip install -r requirements.txt
````

## Usage

Run live webcam detection:

```bash
python run.py --source 0
```

Run on a saved video:

```bash
python run.py --source path/to/video.mp4
```

Run evaluation with custom rules (example):

```bash
python run.py --source 0 --rules wh40k.json
```

## Dataset

Training uses a mix of public and custom datasets:

* [Six-sided Dice Dataset (Kaggle, Nell Byler)](https://www.kaggle.com/datasets/nellbyler/d6-dice)
* [Dice Detection Dataset (Roboflow)](https://universe.roboflow.com/yolo-hkw8z/dice-detection-3rsln)
* Self-recorded webcam dice throws for validation.
  
## Project Structure

```
DiceWatcher/
│
├── data/             # dataset samples and annotations
├── models/           # trained weights and configs
├── src/              # main source code
│   ├── detection/    # dice detection and recognition
│   ├── rules/        # rule engine for wargames
│   └── utils/        # helper functions
├── run.py            # entry point for detection + rules
├── requirements.txt
└── README.md
```

## References

* H. Wimsatt, *Using Machine Learning to Interpret Dice Rolls*, 2021.
* D. Jha et al., *Exploring Deep Learning Methods for Real-Time Surgical Instrument Segmentation*, 2021.
* [Six-sided Dice Dataset, Kaggle](https://www.kaggle.com/datasets/nellbyler/d6-dice).
* [Dice Detection Dataset, Roboflow](https://universe.roboflow.com/yolo-hkw8z/dice-detection-3rsln).

## License

MIT License. See [LICENSE](LICENSE).

```

---

Do you want me to also prepare a **requirements.txt** with the exact minimal dependencies (torch, opencv, ultralytics, etc.) so the repo is runnable out of the box?
```
