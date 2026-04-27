Nereus-LC: Event-Triggered Dog Waste Detection and Response
Nereus-LC is a computer-vision prototype for detecting dog activity in yard video and logging likely waste events. The current runtime combines YOLOv8 dog detection, MobileNetV2 poop-posture gating, CLIP-based pee / poop / neutral confirmation, EMA smoothing, ON/OFF thresholding, and event evidence logging.
Repository layout
```text
440-AI-Nereus-Beneventi-main/
├── CPT_S_540-AI-Final-Paper-Beneventi.pdf
├── Nereus-LC-540-Final.pptx
├── README.md
├── requirements.txt
└── Nereus-Captured/
    ├── Nereus-Captured.py
    ├── clip_event_scorer.py
    ├── train_dogpoop_mobilenetv2.py
    ├── tau.txt
    ├── class_indices.json
    └── Test Videos/
        └── Test1.mp4
```
Files needed to run inference
The runtime expects these files to be in the same folder as `Nereus-Captured.py`:
```text
Nereus-Captured.py
clip_event_scorer.py
mobilenetv2_dogpoop.pt      # trained MobileNetV2 posture weights
class_indices.json
tau.txt
```
`yolov8n.pt` is optional. If it is not present, Ultralytics can download the standard YOLOv8n weights automatically on first run.
Important: the trained file `mobilenetv2_dogpoop.pt` is required. If it is missing, the runtime cannot classify poop posture.
Setup
Windows PowerShell
```powershell
python -m venv env
.\env\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS / Linux
```bash
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If you need GPU-enabled PyTorch, install the correct PyTorch build for your CUDA version from the PyTorch website before installing the rest of the requirements.
Running the demo video
From the repository root:
```bash
cd Nereus-Captured
python Nereus-Captured.py --src "Test Videos/Test1.mp4" --record
```
On Windows PowerShell:
```powershell
cd Nereus-Captured
python Nereus-Captured.py --src "Test Videos\Test1.mp4" --record
```
Outputs are saved to:
```text
Nereus-Captured/nereus_out/
```
Typical outputs include an annotated MP4, clean snapshots, post-event snapshots, localized region crops, and console logs with MobileNet/CLIP scores.
Running from a webcam
```bash
python Nereus-Captured.py --src 0 --record
```
Press `q` in the preview window to quit.
Command-line options
```text
--src       Camera index or video path. Example: 0 or "Test Videos/Test1.mp4"
--tau       Optional override for the MobileNet posture threshold
--record    Save annotated MP4 output
--aim       Override butt-region crop direction: auto, left, or right
```
Metadata files
`tau.txt`
Stores the selected MobileNet posture threshold. Example:
```text
0.53
```
`class_indices.json`
Maps class IDs to labels. Example:
```json
{
  "0": "notpoop",
  "1": "poop"
}
```
Notes on CLIP
The CLIP stage scores candidate event crops against prompt groups for neutral dog behavior, poop-related behavior, and pee-related behavior. Current experiments showed that CLIP is sensitive to prompt wording and may bias toward the pee label. Treat CLIP results as exploratory confirmation evidence, not ground truth.
Training script
`train_dogpoop_mobilenetv2.py` is included for reference. It expects an ImageFolder-style dataset:
```text
dpd2024/
├── train/
│   ├── notpoop/
│   └── poop/
├── val/
│   ├── notpoop/
│   └── poop/
└── test/
    ├── notpoop/
    └── poop/
```
By default, place `dpd2024/` inside `Nereus-Captured/`. You can also set an environment variable named `NEREUS_DATA_DIR` to point to the dataset location.
Suggested cleanup before submission
Do not include temporary test files, empty placeholders, cache folders, or generated output folders. Keep the repository focused on source code, report/slides, metadata files, sample video, and documented model-weight requirements.
